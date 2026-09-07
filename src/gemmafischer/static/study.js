const $s = (id) => document.getElementById(id);
const pieces = {
  p: "♟", r: "♜", n: "♞", b: "♝", q: "♛", k: "♚",
  P: "♙", R: "♖", N: "♘", B: "♗", Q: "♕", K: "♔",
};
const pieceNames = {
  p: "black pawn", r: "black rook", n: "black knight", b: "black bishop", q: "black queen", k: "black king",
  P: "white pawn", R: "white rook", N: "white knight", B: "white bishop", Q: "white queen", K: "white king",
};
const terminalStudyStates = new Set(["ready", "cancelled", "failed", "paused_interrupted", "paused_storage"]);
const study = {
  job: null,
  moment: null,
  fen: null,
  phase: "original",
  selected: null,
  moves: [],
  focusSquare: "a8",
  locked: false,
  pollController: null,
};

class StudyApiError extends Error {
  constructor(payload, status) {
    super(payload?.message || `Request failed (${status})`);
    this.code = payload?.code || "REQUEST_FAILED";
  }
}

async function studyApi(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  const body = await response.json();
  if (!response.ok) throw new StudyApiError(body?.error, response.status);
  return body;
}

function showView(name) {
  document.querySelectorAll(".product-view").forEach((view) => {
    view.hidden = view.id !== `${name}-view`;
  });
  document.querySelectorAll(".nav-tab").forEach((tab) => {
    const active = tab.dataset.view === name;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-current", active ? "page" : "false");
  });
  if (name === "review") void loadReviews();
  if (name === "progress") void loadProgress();
  if (name === "lab") void window.gemmafischerEnsureLab?.();
}

function setStudyError(message = "") {
  $s("study-error").textContent = message;
  $s("study-error").hidden = !message;
}

function setImportCollapsed(collapsed) {
  $s("pgn-form").hidden = collapsed;
  $s("analyze-another").hidden = !collapsed;
}

function setSubmitBusy(busy) {
  $s("import-submit").disabled = busy;
  $s("import-submit").setAttribute("aria-busy", String(busy));
}

function studyStateLabel(job) {
  const labels = {
    queued: "Waiting for Stockfish…",
    parsing: "Reading and validating the game…",
    screening: "Screening every decision you made…",
    deep_analysis: "Verifying the strongest learning moments…",
    building_transfer: "Building transfer practice…",
    ready: job.moments.length ? "Your learning moments are ready" : "No material mistakes found",
    cancelled: "Analysis cancelled",
    paused_interrupted: "Analysis paused after interruption",
    paused_storage: "Analysis paused because storage is unavailable",
    failed: "This game could not be analyzed",
  };
  return labels[job.state] || job.state;
}

function momentAction(status) {
  if (status === "in_progress") return { label: "Continue this moment", phase: "retry" };
  if (status === "scheduled") return { label: "Practiced", phase: null };
  if (status === "mastered") return { label: "Mastered", phase: null };
  return { label: "Solve this moment", phase: "original" };
}

function nextUnsolvedMoment(currentId = null) {
  const moments = study.job?.moments || [];
  const after = currentId
    ? moments.slice(moments.findIndex((moment) => moment.moment_id === currentId) + 1)
    : moments;
  return after.concat(moments).find((moment) =>
    ["new", "in_progress"].includes(moment.practice_status || "new")
    && moment.moment_id !== currentId,
  ) || null;
}

function renderJob(job) {
  study.job = job;
  setImportCollapsed(true);
  setSubmitBusy(!terminalStudyStates.has(job.state));
  $s("study-work").hidden = !$s("practice-work").hidden;
  $s("study-state").textContent = studyStateLabel(job);
  const total = Math.max(job.progress.total_units, 1);
  $s("study-progress").max = total;
  $s("study-progress").value = job.progress.completed_units;
  $s("cancel-study").hidden = terminalStudyStates.has(job.state);
  if (job.error) setStudyError(job.error.message);
  else if (job.state !== "failed") setStudyError();
  const list = $s("moment-list");
  list.replaceChildren();
  job.moments.forEach((moment) => {
    const status = moment.practice_status || "new";
    const action = momentAction(status);
    const card = document.createElement("article");
    card.className = "moment-card";
    card.dataset.status = status;
    const rank = document.createElement("span");
    rank.className = "moment-rank";
    rank.textContent = `0${moment.rank}`;
    const content = document.createElement("div");
    const title = document.createElement("h3");
    title.textContent = `Move ${Math.ceil(moment.source_ply / 2)} · ${moment.played_move_san}`;
    const detail = document.createElement("p");
    const loss = moment.mate_loss ? "missed forced result" : `${moment.severity_cp} cp swing`;
    detail.textContent = `${loss}${moment.concept_keys.length ? ` · ${moment.concept_keys.join(", ")}` : ""}`;
    const statusLine = document.createElement("p");
    statusLine.className = "moment-status";
    statusLine.textContent = status.replaceAll("_", " ");
    content.append(title, detail, statusLine);
    if (action.phase) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "primary";
      button.textContent = action.label;
      button.addEventListener("click", () => openMoment(moment, action.phase));
      content.append(button);
    }
    card.append(rank, content);
    list.append(card);
  });
}

async function pollStudy(jobId) {
  study.pollController?.abort();
  const controller = new AbortController();
  study.pollController = controller;
  setSubmitBusy(true);
  try {
    while (!controller.signal.aborted) {
      const job = await studyApi(`/api/v1/study-jobs/${jobId}`, { signal: controller.signal });
      renderJob(job);
      if (terminalStudyStates.has(job.state)) return;
      await new Promise((resolve) => setTimeout(resolve, 500));
    }
  } finally {
    if (study.pollController === controller) setSubmitBusy(false);
  }
}

function parseStudyFen(fen) {
  const ranks = fen.split(" ")[0].split("/");
  const board = {};
  ranks.forEach((rank, row) => {
    let file = 0;
    for (const token of rank) {
      if (/\d/.test(token)) file += Number(token);
      else {
        board["abcdefgh"[file] + (8 - row)] = token;
        file += 1;
      }
    }
  });
  return board;
}

function renderStudyBoard() {
  const node = $s("study-board");
  const position = parseStudyFen(study.fen);
  const perspective = study.job?.game?.perspective || "white";
  const files = perspective === "white" ? [..."abcdefgh"] : [..."hgfedcba"];
  const ranks = perspective === "white" ? [8, 7, 6, 5, 4, 3, 2, 1] : [1, 2, 3, 4, 5, 6, 7, 8];
  node.classList.toggle("locked", study.locked);
  node.replaceChildren();
  ranks.flatMap((rank) => files.map((file) => file + rank)).forEach((square, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `square ${(Math.floor(index / 8) + index) % 2 ? "dark" : "light"}`;
    if (study.selected === square) button.classList.add("selected");
    if (!study.locked && study.moves.some((move) => move.slice(2, 4) === square)) {
      button.classList.add(position[square] ? "legal-capture" : "legal-target");
    }
    button.dataset.square = square;
    button.tabIndex = square === study.focusSquare ? 0 : -1;
    button.setAttribute("role", "gridcell");
    const piece = position[square];
    button.setAttribute(
      "aria-label",
      `${piece ? `${pieceNames[piece]} on ` : "Empty "}${square}${study.selected === square ? ", selected" : ""}`,
    );
    button.textContent = piece ? pieces[piece] : "";
    button.addEventListener("click", () => void selectStudySquare(square));
    button.addEventListener("focus", () => { study.focusSquare = square; });
    button.addEventListener("keydown", studyBoardKeydown);
    node.append(button);
  });
}

function studyBoardKeydown(event) {
  const buttons = [...$s("study-board").querySelectorAll("button")];
  const current = buttons.indexOf(event.currentTarget);
  const deltas = { ArrowLeft: -1, ArrowRight: 1, ArrowUp: -8, ArrowDown: 8 };
  if (event.key === "Escape") {
    event.preventDefault();
    study.selected = null;
    study.moves = [];
    renderStudyBoard();
    return;
  }
  if (!(event.key in deltas)) return;
  event.preventDefault();
  const target = buttons[Math.max(0, Math.min(63, current + deltas[event.key]))];
  study.focusSquare = target.dataset.square;
  buttons.forEach((button) => { button.tabIndex = button === target ? 0 : -1; });
  target.focus();
}

function requestStudyPromotion(candidates) {
  return new Promise((resolve, reject) => {
    const open = window.gemmafischerOpenPromotion;
    if (typeof open !== "function") {
      reject(new Error("Promotion choice is unavailable."));
      return;
    }
    open(candidates, resolve, () => {
      const error = new Error("Promotion cancelled");
      error.name = "AbortError";
      reject(error);
    });
  });
}

async function selectStudySquare(square) {
  if (!study.moment || study.locked) return;
  const matching = study.moves.filter(
    (move) => study.selected && move.startsWith(`${study.selected}${square}`),
  );
  if (matching.length) {
    if (matching.length > 1) {
      try {
        await submitAttempt(await requestStudyPromotion(matching));
      } catch (error) {
        if (error.name !== "AbortError") $s("study-board-status").textContent = error.message;
      }
      return;
    }
    await submitAttempt(matching[0]);
    return;
  }
  try {
    const result = await studyApi(
      `/api/v1/studies/${study.job.job_id}/moments/${study.moment.moment_id}/legal-moves?from_square=${square}`,
    );
    study.selected = square;
    study.moves = result.moves_uci;
    $s("study-board-status").textContent = result.moves_uci.length ? "Choose a marked destination." : "That piece has no legal moves.";
    renderStudyBoard();
  } catch (error) {
    $s("study-board-status").textContent = error.message;
  }
}

function focusStudyBoard() {
  $s("study-board").querySelector('[tabindex="0"]')?.focus();
}

function openMoment(moment, phase, fen = null) {
  study.moment = moment;
  study.phase = phase;
  study.fen = fen || moment.fen;
  study.selected = null;
  study.moves = [];
  study.locked = false;
  study.focusSquare = study.job?.game?.perspective === "black" ? "h1" : "a8";
  $s("study-work").hidden = true;
  $s("practice-work").hidden = false;
  $s("practice-phase").textContent = phase === "delayed_review" ? "Delayed review" : phase === "transfer" ? "Related position" : phase === "retry" ? "Retry before moving on" : "Solve before reveal";
  $s("practice-title").textContent = "What would you play here?";
  $s("moment-context").textContent = phase === "original"
    ? `In your game you played ${moment.played_move_san}. Find a stronger continuation without seeing the answer.`
    : phase === "delayed_review"
      ? "This idea is due. Play the stronger continuation again."
      : "Use the idea, not just your memory of the move.";
  $s("attempt-feedback").hidden = true;
  $s("retry-moment").hidden = true;
  $s("transfer-moment").hidden = true;
  $s("next-moment").hidden = true;
  $s("study-board-status").textContent = "Choose a piece, then a destination.";
  renderStudyBoard();
  focusStudyBoard();
}

function lockBoardForRetry() {
  study.locked = true;
  study.selected = null;
  study.moves = [];
  renderStudyBoard();
  $s("retry-moment").focus();
}

function showNextCta(attempt) {
  if (attempt.feedback?.next_fen) {
    $s("transfer-moment").hidden = false;
    $s("transfer-moment").dataset.fen = attempt.feedback.next_fen;
    return;
  }
  const next = nextUnsolvedMoment(study.moment.moment_id);
  if (next) {
    $s("next-moment").hidden = false;
    $s("next-moment").dataset.momentId = next.moment_id;
    $s("next-moment").dataset.phase = momentAction(next.practice_status || "new").phase || "original";
  }
}

async function submitAttempt(move) {
  $s("study-board").classList.add("busy");
  try {
    const attempt = await studyApi(
      `/api/v1/studies/${study.job.job_id}/moments/${study.moment.moment_id}/attempts`,
      {
        method: "POST",
        headers: { "Idempotency-Key": `attempt-${crypto.randomUUID()}` },
        body: JSON.stringify({
          expected_revision: study.job.revision,
          phase: study.phase,
          move_uci: move,
          hint_used: false,
        }),
      },
    );
    const feedback = $s("attempt-feedback");
    feedback.hidden = false;
    if (!attempt.feedback) {
      feedback.textContent = "Not quite. The answer is still hidden. Retry this position and calculate once more.";
      $s("retry-moment").hidden = false;
      lockBoardForRetry();
      return;
    }
    feedback.textContent = `${attempt.outcome === "incorrect" ? "The engine prefers" : "Good."} ${attempt.feedback.preferred_move_san}. ${attempt.feedback.message}`;
    study.locked = true;
    study.selected = null;
    study.moves = [];
    renderStudyBoard();
    await refreshCounts();
    if (study.job) {
      try {
        study.job = await studyApi(`/api/v1/study-jobs/${study.job.job_id}`);
      } catch {
        /* Moment list refreshes when the learner returns. */
      }
    }
    showNextCta(attempt);
  } catch (error) {
    $s("study-board-status").textContent = error.message;
  } finally {
    $s("study-board").classList.remove("busy");
  }
}

async function openDueReview(card) {
  study.job = await studyApi(`/api/v1/study-jobs/${card.job_id}`);
  showView("learn");
  setImportCollapsed(true);
  openMoment(card.moment, "delayed_review");
}

async function loadReviews() {
  const list = $s("review-list");
  try {
    const due = await studyApi("/api/v1/reviews/due");
    list.replaceChildren();
    due.items.forEach((card) => {
      const moment = card.moment;
      const button = document.createElement("button");
      button.className = "moment-card review-card";
      button.type = "button";
      button.textContent = `${card.concept_key} · move ${Math.ceil(moment.source_ply / 2)} · review now`;
      button.addEventListener("click", () => void openDueReview(card));
      list.append(button);
    });
    if (!list.children.length) {
      list.textContent = "Nothing is due. Practice a learning moment to start your review schedule.";
    }
  } catch (error) {
    list.replaceChildren();
    list.textContent = `Due reviews could not be loaded: ${error.message}`;
  }
}

async function loadProgress() {
  try {
    const value = await studyApi("/api/v1/progress");
    const metrics = [
      ["Due", value.due], ["Learning", value.learning], ["Retaining", value.retaining],
      ["Mastered", value.mastered], ["Attempts", value.attempts],
    ];
    if (value.attempts) {
      metrics.push(
        ["First-try accuracy", `${Math.round(value.original_accuracy * 100)}%`],
        ["Transfer accuracy", `${Math.round(value.transfer_accuracy * 100)}%`],
        ["Delayed accuracy", `${Math.round(value.delayed_accuracy * 100)}%`],
      );
    }
    const grid = $s("progress-grid");
    grid.replaceChildren();
    metrics.forEach(([label, metric]) => {
      const card = document.createElement("div");
      const strong = document.createElement("strong");
      const span = document.createElement("span");
      strong.textContent = String(metric);
      span.textContent = label;
      card.append(strong, span);
      grid.append(card);
    });
  } catch (error) {
    $s("progress-grid").textContent = `Progress could not be loaded: ${error.message}`;
  }
}

async function refreshCounts() {
  const badge = $s("due-count");
  const tab = document.querySelector('.nav-tab[data-view="review"]');
  try {
    const due = await studyApi("/api/v1/reviews/due");
    badge.textContent = String(due.count);
    badge.hidden = due.count === 0;
    tab?.setAttribute(
      "aria-label",
      due.count === 0 ? "Review, no cards due" : `Review, ${due.count} due`,
    );
  } catch {
    badge.textContent = "–";
    badge.hidden = false;
    tab?.setAttribute("aria-label", "Review, count unavailable");
  }
}

async function restoreLatestStudy() {
  try {
    const jobs = await studyApi("/api/v1/study-jobs?limit=1");
    if (!jobs.items.length) return;
    const latest = jobs.items[0];
    renderJob(latest);
    if (!terminalStudyStates.has(latest.state)) await pollStudy(latest.job_id);
  } catch (error) {
    setStudyError(`Saved study could not be restored: ${error.message}`);
  }
}

document.querySelectorAll(".nav-tab").forEach((tab) => tab.addEventListener("click", () => showView(tab.dataset.view)));
$s("study-perspective").addEventListener("change", () => { $s("player-name").disabled = $s("study-perspective").value !== "auto"; });
$s("analyze-another").addEventListener("click", () => {
  setImportCollapsed(false);
  $s("pgn").focus();
});
$s("pgn-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  setStudyError();
  $s("practice-work").hidden = true;
  try {
    const accepted = await studyApi("/api/v1/study-jobs", {
      method: "POST",
      headers: { "Idempotency-Key": `study-${crypto.randomUUID()}` },
      body: JSON.stringify({ pgn: $s("pgn").value, perspective: $s("study-perspective").value, player_name: $s("player-name").value || null, rating_bucket: $s("study-rating").value }),
    });
    setImportCollapsed(true);
    await pollStudy(accepted.job_id);
  } catch (error) {
    setSubmitBusy(false);
    if (error.name !== "AbortError") setStudyError(error.message);
  }
});
$s("cancel-study").addEventListener("click", async () => { if (study.job) renderJob(await studyApi(`/api/v1/study-jobs/${study.job.job_id}`, { method: "DELETE" })); });
$s("close-study-practice").addEventListener("click", () => {
  $s("practice-work").hidden = true;
  $s("study-work").hidden = false;
  if (study.job) renderJob(study.job);
});
$s("retry-moment").addEventListener("click", () => openMoment(study.moment, "retry"));
$s("transfer-moment").addEventListener("click", (event) => openMoment(study.moment, "transfer", event.currentTarget.dataset.fen));
$s("next-moment").addEventListener("click", () => {
  const next = (study.job?.moments || []).find((moment) => moment.moment_id === $s("next-moment").dataset.momentId);
  if (next) openMoment(next, $s("next-moment").dataset.phase || "original");
});
$s("delete-progress").addEventListener("click", async () => {
  if (window.confirm("Delete all practice attempts and review scheduling? Your imported studies will remain.")) {
    await studyApi("/api/v1/progress", { method: "DELETE" });
    await loadProgress();
    await refreshCounts();
    if (study.job) {
      try {
        renderJob(await studyApi(`/api/v1/study-jobs/${study.job.job_id}`));
      } catch {
        /* The job list is unchanged if this refresh fails. */
      }
    }
  }
});

void Promise.all([refreshCounts(), restoreLatestStudy()]);
