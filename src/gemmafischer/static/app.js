const PIECES = {
  p: "♟",
  r: "♜",
  n: "♞",
  b: "♝",
  q: "♛",
  k: "♚",
  P: "♙",
  R: "♖",
  N: "♘",
  B: "♗",
  Q: "♕",
  K: "♔",
};
const NAMES = {
  p: "black pawn",
  r: "black rook",
  n: "black knight",
  b: "black bishop",
  q: "black queen",
  k: "black king",
  P: "white pawn",
  R: "white rook",
  N: "white knight",
  B: "white bishop",
  Q: "white queen",
  K: "white king",
};
const START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const EXAMPLE_FEN =
  "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3";
const STORAGE_KEY = "gemmafischer.session.v2";
const $ = (id) => document.getElementById(id);
const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const state = {
  session: null,
  tutor: null,
  tutorMode: false,
  lastAnalysisId: null,
  squares: {},
  turn: "w",
  flipped: false,
  selected: null,
  legalMoves: [],
  legalTargets: [],
  lastMove: [],
  focusSquare: "a8",
  busy: false,
  busyToken: 0,
  sessionEpoch: 0,
  sessionTransition: false,
  legalEpoch: 0,
  legalController: null,
  reviewToken: 0,
  reviewController: null,
  analysisId: null,
  animationToken: 0,
  activeGhost: null,
  exhibitionToken: 0,
  exhibitionRunning: false,
  exhibitionTransition: false,
  exhibitionPromise: null,
  promotionMoves: [],
  promotionFocus: null,
};
const squareNodes = new Map();
class StaleIntentError extends Error {}
class ApiError extends Error {
  constructor(error, status) {
    super(error?.message || `Request failed (${status})`);
    this.name = "ApiError";
    this.code = error?.code || "REQUEST_FAILED";
    this.retryable = Boolean(error?.retryable);
    this.requestId = error?.request_id || null;
  }
}
const superseded = (error) =>
  error?.name === "AbortError" || error instanceof StaleIntentError;

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  const data = response.status === 204 ? null : await response.json();
  if (!response.ok) {
    if (["STORAGE_UNAVAILABLE", "STORAGE_CORRUPT"].includes(data?.error?.code))
      updateStorageBanner(
        data.error.code === "STORAGE_CORRUPT" ? "corrupt" : "degraded",
      );
    throw new ApiError(data?.error, response.status);
  }
  return data;
}
function idempotencyKey(prefix) {
  return `${prefix}-${crypto.randomUUID()}`;
}
function sessionContext() {
  return {
    sessionId: state.session?.session_id || null,
    revision: state.session?.revision ?? null,
    epoch: state.sessionEpoch,
  };
}
function contextIsCurrent(context) {
  return (
    context.epoch === state.sessionEpoch &&
    context.sessionId === state.session?.session_id
  );
}
function beginBusy() {
  const token = ++state.busyToken;
  state.busy = true;
  return token;
}
function endBusy(token) {
  if (token !== state.busyToken) return;
  state.busy = false;
  if (state.session) renderSession();
  else renderBoard();
}
function invalidateBusy() {
  state.busyToken += 1;
  state.busy = false;
}
function savePreferences() {
  try {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        schema: 2,
        sessionId: state.session?.session_id,
        flipped: state.flipped,
        mode: $("session-mode").value,
        difficulty: $("difficulty").value,
        rating: $("rating").value,
      }),
    );
  } catch {
    /* Storage can be unavailable. */
  }
}
function loadPreferences() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
    if (!saved || saved.schema !== 2) return {};
    state.flipped = Boolean(saved.flipped);
    if (["player", "exhibition"].includes(saved.mode))
      $("session-mode").value = saved.mode;
    if (["casual", "club", "strong"].includes(saved.difficulty))
      $("difficulty").value = saved.difficulty;
    if (
      ["1000-1199", "1200-1399", "1400-1599", "1600-1800"].includes(
        saved.rating,
      )
    )
      $("rating").value = saved.rating;
    return saved;
  } catch {
    localStorage.removeItem(STORAGE_KEY);
    return {};
  }
}

function parseFen(fen) {
  const fields = fen.trim().split(/\s+/);
  if (fields.length !== 6)
    throw new Error(`FEN requires six fields; received ${fields.length}.`);
  const ranks = fields[0].split("/");
  if (ranks.length !== 8) throw new Error("FEN board requires eight ranks.");
  const squares = {};
  ranks.forEach((rank, row) => {
    let file = 0;
    for (const char of rank) {
      if (/\d/.test(char)) file += Number(char);
      else {
        if (!PIECES[char] || file > 7)
          throw new Error("FEN contains invalid piece placement.");
        squares["abcdefgh"[file] + (8 - row)] = char;
        file += 1;
      }
    }
    if (file !== 8)
      throw new Error("Every FEN rank must describe eight squares.");
  });
  if (!["w", "b"].includes(fields[1]))
    throw new Error("FEN side to move must be w or b.");
  state.squares = squares;
  state.turn = fields[1];
}
function initializeBoard() {
  for (const rank of [8, 7, 6, 5, 4, 3, 2, 1])
    for (const file of "abcdefgh") {
      const square = file + rank,
        button = document.createElement("button"),
        piece = document.createElement("span"),
        coordinate = document.createElement("span");
      button.type = "button";
      button.dataset.square = square;
      button.setAttribute("role", "gridcell");
      piece.className = "piece";
      piece.setAttribute("aria-hidden", "true");
      coordinate.className = "coordinate";
      coordinate.setAttribute("aria-hidden", "true");
      coordinate.textContent = square;
      button.append(piece, coordinate);
      button.addEventListener("click", () => selectSquare(square));
      button.addEventListener("focus", () => {
        state.focusSquare = square;
      });
      button.addEventListener("keydown", boardKeydown);
      squareNodes.set(square, button);
    }
}
function boardOrder() {
  const files = state.flipped ? [..."hgfedcba"] : [..."abcdefgh"],
    ranks = state.flipped ? [1, 2, 3, 4, 5, 6, 7, 8] : [8, 7, 6, 5, 4, 3, 2, 1];
  return ranks.flatMap((rank) => files.map((file) => file + rank));
}
function renderBoard() {
  const board = $("board"),
    order = boardOrder();
  board.classList.toggle("busy", state.busy);
  board.classList.toggle("automated", state.exhibitionRunning);
  board.setAttribute("aria-busy", String(state.busy));
  const inputDisabled =
    state.busy || state.exhibitionRunning || state.sessionTransition;
  order.forEach((square, index) => {
    const button = squareNodes.get(square),
      piece = state.squares[square],
      legal = state.legalTargets.includes(square),
      row = Math.floor(index / 8),
      column = index % 8;
    button.className = `square ${(row + column) % 2 ? "dark" : "light"}`;
    if (state.selected === square) button.classList.add("selected");
    if (state.lastMove.includes(square)) button.classList.add("last-move");
    if (legal) button.classList.add(piece ? "legal-capture" : "legal-target");
    button.querySelector(".piece").textContent = piece ? PIECES[piece] : "";
    button.querySelector(".coordinate").textContent =
      row === 7 ? square[0] : column === 0 ? square[1] : "";
    button.setAttribute(
      "aria-label",
      `${piece ? NAMES[piece] + " on " : "Empty "}${square}${state.selected === square ? ", selected" : ""}${legal ? ", legal destination" : ""}${state.lastMove.includes(square) ? ", last move" : ""}`,
    );
    button.setAttribute("aria-selected", String(state.selected === square));
    button.disabled = inputDisabled;
    button.setAttribute("aria-disabled", String(inputDisabled));
    button.tabIndex = !inputDisabled && square === state.focusSquare ? 0 : -1;
  });
  const rows = [];
  for (let row = 0; row < 8; row += 1) {
    const rowNode = document.createElement("div");
    rowNode.className = "board-row";
    rowNode.setAttribute("role", "row");
    rowNode.append(...order.slice(row * 8, row * 8 + 8).map((square) => squareNodes.get(square)));
    rows.push(rowNode);
  }
  board.replaceChildren(...rows);
}
function boardKeydown(event) {
  const deltas = {
    ArrowLeft: [-1, 0],
    ArrowRight: [1, 0],
    ArrowUp: [0, -1],
    ArrowDown: [0, 1],
  };
  if (!(event.key in deltas)) return;
  event.preventDefault();
  const order = boardOrder(),
    index = order.indexOf(event.currentTarget.dataset.square),
    row = Math.floor(index / 8),
    column = index % 8,
    [dx, dy] = deltas[event.key],
    nextRow = Math.max(0, Math.min(7, row + dy)),
    nextColumn = Math.max(0, Math.min(7, column + dx)),
    next = order[nextRow * 8 + nextColumn];
  state.focusSquare = next;
  renderBoard();
  squareNodes.get(next).focus();
}
function pieceMatchesPlayer(piece) {
  if (!piece || !state.session) return false;
  const color = piece === piece.toUpperCase() ? "white" : "black";
  if (state.tutorMode)
    return color === (state.turn === "w" ? "white" : "black");
  if (state.session.mode !== "player") return false;
  return (
    state.session.player_color === color &&
    state.session.turn === state.session.player_color
  );
}

async function selectSquare(square) {
  if (state.busy || state.exhibitionRunning || !state.session) return;
  const piece = state.squares[square];
  if (!state.selected) {
    if (!pieceMatchesPlayer(piece)) {
      setError(
        state.session.turn === state.session.player_color
          ? "Select one of your pieces."
          : "Stockfish is to move.",
      );
      return;
    }
    await selectSource(square);
    return;
  }
  if (pieceMatchesPlayer(piece)) {
    await selectSource(square);
    return;
  }
  if (!state.legalTargets.includes(square)) {
    setError(`${square} is not a legal destination for ${state.selected}.`);
    return;
  }
  const candidates = state.legalMoves.filter(
    (move) =>
      move.slice(0, 2) === state.selected && move.slice(2, 4) === square,
  );
  if (candidates.length > 1) {
    openPromotion(candidates);
    return;
  }
  const move = candidates[0];
  if (!move) {
    setError("The legal move list changed. Select the piece again.");
    return;
  }
  clearSelection();
  renderBoard();
  if (state.tutorMode) await submitTutorAnswer(move);
  else await playPlayerTurn(move);
}
function clearSelection() {
  state.selected = null;
  state.legalMoves = [];
  state.legalTargets = [];
  state.legalController?.abort();
  state.legalController = null;
}
async function selectSource(square) {
  setError();
  state.legalController?.abort();
  const controller = new AbortController(),
    legalEpoch = ++state.legalEpoch,
    context = sessionContext();
  state.legalController = controller;
  state.selected = square;
  state.legalMoves = [];
  state.legalTargets = [];
  renderBoard();
  setStatus(
    `${NAMES[state.squares[square]]} selected on ${square}. Loading legal moves…`,
  );
  try {
    const path = state.tutorMode
        ? `/api/v1/sessions/${context.sessionId}/tutor/${state.tutor.interaction_id}/legal-moves?from_square=${square}`
        : `/api/v1/sessions/${context.sessionId}/legal-moves?from_square=${square}`,
      data = await api(path, { signal: controller.signal });
    if (
      !contextIsCurrent(context) ||
      legalEpoch !== state.legalEpoch ||
      state.selected !== square ||
      state.session.revision !== context.revision
    )
      return;
    state.legalMoves = data.moves_uci;
    state.legalTargets = data.destinations;
    setStatus(
      data.destinations.length
        ? `${data.destinations.length} legal destination${data.destinations.length === 1 ? "" : "s"} highlighted.`
        : `${square} has no legal moves.`,
    );
  } catch (error) {
    if (superseded(error)) return;
    if (state.selected === square) clearSelection();
    setError(`Legal moves could not be loaded: ${error.message}`);
  } finally {
    if (state.legalController === controller) state.legalController = null;
  }
  renderBoard();
}

function openPromotion(candidates) {
  state.promotionMoves = candidates;
  state.promotionFocus = state.selected;
  const dialog = $("promotion-dialog");
  dialog.showModal();
  dialog.querySelector('[data-promotion="q"]').focus();
}
function cancelPromotion() {
  state.promotionMoves = [];
  if ($("promotion-dialog").open) $("promotion-dialog").close();
  renderBoard();
  if (state.promotionFocus) squareNodes.get(state.promotionFocus)?.focus();
  state.promotionFocus = null;
}
function choosePromotion(suffix) {
  const move = state.promotionMoves.find((candidate) =>
    candidate.endsWith(suffix),
  );
  if (!move) return;
  state.promotionMoves = [];
  $("promotion-dialog").close();
  clearSelection();
  renderBoard();
  state.promotionFocus = null;
  if (state.tutorMode) void submitTutorAnswer(move);
  else void playPlayerTurn(move);
}

function cancelAnimation() {
  state.animationToken += 1;
  if (!state.activeGhost) return;
  state.activeGhost.sourceNode.querySelector(".piece").style.visibility = "";
  state.activeGhost.ghost.remove();
  state.activeGhost = null;
}
async function animatePly(ply, context) {
  if (!contextIsCurrent(context)) return false;
  const token = ++state.animationToken;
  const board = $("board"),
    boardRect = board.getBoundingClientRect(),
    sourceNode = squareNodes.get(ply.move_uci.slice(0, 2)),
    destinationNode = squareNodes.get(ply.move_uci.slice(2, 4)),
    source = sourceNode.getBoundingClientRect(),
    destination = destinationNode.getBoundingClientRect(),
    glyph = sourceNode.querySelector(".piece").textContent;
  state.lastMove = [ply.move_uci.slice(0, 2), ply.move_uci.slice(2, 4)];
  if (glyph && !matchMedia("(prefers-reduced-motion: reduce)").matches) {
    const ghost = document.createElement("span");
    ghost.className = "piece-ghost";
    ghost.setAttribute("aria-hidden", "true");
    ghost.textContent = glyph;
    Object.assign(ghost.style, {
      left: `${source.left - boardRect.left}px`,
      top: `${source.top - boardRect.top}px`,
      width: `${source.width}px`,
      height: `${source.height}px`,
    });
    board.append(ghost);
    sourceNode.querySelector(".piece").style.visibility = "hidden";
    state.activeGhost = { ghost, sourceNode, token };
    const animation = ghost.animate(
      [
        { transform: "translate(0,0)" },
        {
          transform: `translate(${destination.left - source.left}px,${destination.top - source.top}px)`,
        },
      ],
      { duration: 180, easing: "cubic-bezier(.2,.8,.2,1)" },
    );
    try {
      await animation.finished;
    } catch {
      /* A newer render superseded the move. */
    } finally {
      if (state.activeGhost?.token === token) {
        ghost.remove();
        sourceNode.querySelector(".piece").style.visibility = "";
        state.activeGhost = null;
      }
    }
  }
  if (token !== state.animationToken || !contextIsCurrent(context))
    return false;
  parseFen(ply.fen_after);
  renderBoard();
  return true;
}
async function sessionCommand(action, moveUci = null) {
  const context = sessionContext();
  const result = await api(`/api/v1/sessions/${context.sessionId}/commands`, {
    method: "POST",
    body: JSON.stringify({
      expected_revision: context.revision,
      action,
      move_uci: moveUci,
    }),
  });
  if (!contextIsCurrent(context)) throw new StaleIntentError();
  return result;
}
async function playPlayerTurn(move) {
  const context = sessionContext(),
    busyToken = beginBusy();
  setError();
  renderBoard();
  setStatus("Move submitted…");
  try {
    const afterPlayer = await sessionCommand("player_move", move),
      playerPly = afterPlayer.plies.at(-1);
    state.session = afterPlayer;
    await animatePly(playerPly, context);
    if (!contextIsCurrent(context)) return;
    renderSession();
    setStatus(`You played ${playerPly.move_san}. Stockfish is thinking…`);
    const afterEngine =
      afterPlayer.status === "complete"
        ? afterPlayer
        : await sessionCommand("engine_move");
    if (afterEngine.revision !== afterPlayer.revision) {
      const enginePly = afterEngine.plies.at(-1);
      state.session = afterEngine;
      await animatePly(enginePly, context);
    }
    if (!contextIsCurrent(context)) return;
    renderSession();
    savePreferences();
    setStatus(
      state.session.outcome
        ? `Game over: ${state.session.outcome}.`
        : `${state.session.turn === state.session.player_color ? "Your" : "Stockfish"} turn.`,
    );
    const reviewedPlayer = afterEngine.plies.find(
      (ply) => ply.ply === playerPly.ply,
    );
    if (reviewedPlayer?.analysis_id)
      void pollAnalysis(
        reviewedPlayer.analysis_id,
        `Review of ${reviewedPlayer.move_san}`,
      );
  } catch (error) {
    if (!superseded(error)) {
      setError(`The move could not be played: ${error.message}`);
      await refreshSession();
    }
  } finally {
    endBusy(busyToken);
  }
}
async function runExhibition(token) {
  const context = sessionContext();
  while (
    state.exhibitionRunning &&
    token === state.exhibitionToken &&
    state.session.status !== "complete"
  ) {
    const busyToken = beginBusy();
    renderBoard();
    setStatus(
      `${state.session.turn === "white" ? "White" : "Black"} engine is thinking…`,
    );
    try {
      const next = await sessionCommand("engine_move"),
        ply = next.plies.at(-1);
      state.session = next;
      if (token !== state.exhibitionToken || !state.exhibitionRunning) {
        renderSession();
        savePreferences();
        break;
      }
      await animatePly(ply, context);
      if (!contextIsCurrent(context)) break;
      renderSession();
      savePreferences();
      endBusy(busyToken);
      if (ply.analysis_id)
        await pollAnalysis(ply.analysis_id, `Review of ${ply.move_san}`);
      if (
        next.status === "complete" ||
        token !== state.exhibitionToken ||
        !state.exhibitionRunning
      )
        break;
      await wait(350);
    } catch (error) {
      if (!superseded(error))
        setError(`Engine exhibition stopped: ${error.message}`);
      break;
    } finally {
      endBusy(busyToken);
    }
  }
  if (token === state.exhibitionToken) {
    state.exhibitionRunning = false;
    renderExhibitionControl();
    renderBoard();
  }
}

async function pollAnalysis(analysisId, label) {
  cancelReview(false);
  const token = ++state.reviewToken,
    context = sessionContext(),
    controller = new AbortController();
  state.reviewController = controller;
  state.analysisId = analysisId;
  setReviewStatus(`${label}…`);
  $("cancel").hidden = false;
  try {
    while (token === state.reviewToken && contextIsCurrent(context)) {
      const data = await api(`/api/v1/analyses/${analysisId}`, {
          signal: controller.signal,
        }),
        labels = {
          queued: "Review queued.",
          validating: "Checking the position.",
          engine_running: "Stockfish is calculating.",
          comparison_running: "Comparing the move.",
          model_running: "Gemma is ordering the lesson.",
          complete: "Review complete.",
          engine_only: "Engine review complete; Gemma unavailable.",
          cancelled: "Review cancelled.",
          failed: "Review failed.",
        };
      if (token !== state.reviewToken || !contextIsCurrent(context))
        return null;
      setReviewStatus(labels[data.state] || data.state);
      if (
        ["complete", "engine_only", "failed", "cancelled"].includes(data.state)
      ) {
        if (data.state === "failed")
          setError(data.error?.message || "Review failed.");
        else if (data.evidence && data.coaching) renderResult(data);
        return data;
      }
      await wait(300);
    }
  } catch (error) {
    if (!superseded(error))
      setReviewStatus(`Review interrupted: ${error.message}`);
  } finally {
    if (token === state.reviewToken) {
      $("cancel").hidden = true;
      state.analysisId = null;
      state.reviewController = null;
    }
  }
  return null;
}
async function explainPosition() {
  const context = sessionContext();
  try {
    const created = await api("/api/v1/analyses", {
      method: "POST",
      headers: { "Idempotency-Key": idempotencyKey("analysis") },
      body: JSON.stringify({
        mode: "position",
        fen: state.session.fen,
        rating_bucket: $("rating").value,
        considered_move_uci: null,
      }),
    });
    if (!contextIsCurrent(context)) {
      void api(`/api/v1/analyses/${created.analysis_id}`, {
        method: "DELETE",
      }).catch(() => {});
      return;
    }
    await pollAnalysis(created.analysis_id, "Position explanation");
  } catch (error) {
    if (!superseded(error))
      setError(`The explanation could not start: ${error.message}`);
  }
}
function cancelReview(send = true) {
  state.reviewToken += 1;
  state.reviewController?.abort();
  state.reviewController = null;
  if (send && state.analysisId)
    void fetch(`/api/v1/analyses/${state.analysisId}`, { method: "DELETE" });
  state.analysisId = null;
  $("cancel").hidden = true;
}

function renderResult(data) {
  state.lastAnalysisId = data.analysis_id;
  $("empty-guide").hidden = true;
  $("tutor-panel").hidden = true;
  $("result").hidden = false;
  $("practice").hidden = false;
  $("summary").textContent = data.coaching.summary;
  $("degraded").hidden = data.state !== "engine_only";
  $("degraded").textContent =
    data.state === "engine_only"
      ? "The verified deterministic lesson is shown because Gemma was unavailable."
      : "";
  const candidates = data.evidence.candidate_set?.candidates || [],
    comparison = data.evidence.move_comparison,
    byId = Object.fromEntries(
      [
        ...candidates,
        ...data.evidence.board_facts,
        ...(data.evidence.concepts || []),
        ...(comparison ? [comparison] : []),
      ].map((item) => [item.evidence_id, item]),
    );
  const lesson = (data.coaching.lesson_plan?.steps || []).map((step) => {
    const wrap = document.createElement("div"),
      text = document.createElement("p"),
      detail = document.createElement("div");
    wrap.className = "claim lesson-step";
    text.textContent = step.text;
    detail.className = "claim-evidence";
    detail.textContent = evidenceText(byId[step.concept_id]);
    wrap.append(text, detail);
    return wrap;
  });
  const claims = data.coaching.claims.map((claim) => {
    const wrap = document.createElement("div"),
      text = document.createElement("p");
    wrap.className = "claim";
    text.textContent = claimText(claim, data.evidence);
    wrap.append(text);
    if (claim.evidence_ids.length) {
      const button = document.createElement("button"),
        detail = document.createElement("div");
      button.type = "button";
      button.textContent = "Show cited evidence";
      button.setAttribute("aria-expanded", "false");
      detail.className = "claim-evidence";
      detail.hidden = true;
      detail.textContent = claim.evidence_ids
        .map((id) => evidenceText(byId[id]))
        .join("\n");
      button.addEventListener("click", () => {
        detail.hidden = !detail.hidden;
        button.setAttribute("aria-expanded", String(!detail.hidden));
      });
      wrap.append(button, detail);
    }
    return wrap;
  });
  $("claims").replaceChildren(...lesson, ...claims);
  $("evidence").replaceChildren(
    ...candidates.map((item) => {
      const div = document.createElement("div"),
        score =
          item.score_cp !== null
            ? `${(item.score_cp / 100).toFixed(2)} pawns`
            : `mate ${item.mate_in}`;
      div.className = "candidate";
      div.textContent = `${item.rank}. ${item.move_san} · ${score} · ${item.pv_uci.slice(0, 8).join(" ")}`;
      return div;
    }),
  );
}
async function tutorCommand(action, payload = {}) {
  const context = sessionContext(),
    interaction = state.tutor;
  if (!interaction) throw new Error("No practice question is active.");
  let result;
  try {
    result = await api(
      `/api/v1/sessions/${context.sessionId}/tutor/${interaction.interaction_id}/commands`,
      {
        method: "POST",
        body: JSON.stringify({
          expected_revision: interaction.revision,
          action,
          ...payload,
        }),
      },
    );
  } catch (error) {
    if (
      error instanceof ApiError &&
      ["TUTOR_REVISION_CONFLICT", "TUTOR_STATE_CONFLICT"].includes(error.code)
    ) {
      const authoritative = await api(
        `/api/v1/sessions/${context.sessionId}/tutor/${interaction.interaction_id}`,
      );
      if (contextIsCurrent(context)) {
        state.tutor = authoritative;
        renderTutor();
      }
    }
    throw error;
  }
  if (
    !contextIsCurrent(context) ||
    state.tutor?.interaction_id !== interaction.interaction_id
  )
    throw new StaleIntentError();
  state.tutor = result;
  return result;
}
function renderTutor() {
  const tutor = state.tutor;
  if (!tutor) return;
  $("tutor-question").textContent = tutor.question.prompt;
  $("tutor-hint").hidden = !tutor.hint;
  $("tutor-hint").textContent = tutor.hint || "";
  $("tutor-hint-button").hidden =
    Boolean(tutor.hint) || tutor.status !== "awaiting_answer";
  const feedback = $("tutor-feedback");
  feedback.hidden = !tutor.feedback;
  feedback.className = tutor.feedback ? "tutor-feedback" : "";
  feedback.textContent = tutor.feedback
    ? `${tutor.feedback.message} Your move: ${tutor.feedback.submitted_move_san}. Preferred: ${tutor.feedback.preferred_move_san}.`
    : "";
  const followUp = $("tutor-follow-up");
  followUp.hidden = tutor.status === "awaiting_answer";
  if (tutor.status === "awaiting_follow_up") {
    const prompt = document.createElement("p"),
      options = document.createElement("div");
    prompt.textContent = tutor.follow_up.prompt;
    options.className = "tutor-options";
    for (const option of tutor.follow_up.options) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "quiet";
      button.textContent = option.label;
      button.addEventListener("click", () => {
        void submitFollowUp(option.option_id);
      });
      options.append(button);
    }
    followUp.replaceChildren(prompt, options);
  } else if (tutor.status === "complete") {
    const chosen = tutor.follow_up.options.find(
        (option) => option.option_id === tutor.follow_up.selected_option_id,
      ),
      message = document.createElement("p"),
      button = document.createElement("button");
    message.textContent = tutor.follow_up.correct
      ? `Correct — ${chosen?.label || "that is the key idea"}.`
      : "Not quite. Recheck the cited hint and principal variation in the review.";
    button.type = "button";
    button.className = "primary";
    button.textContent = "Return to game";
    button.addEventListener("click", () => void leaveTutor());
    followUp.replaceChildren(message, button);
  }
  renderBoard();
}
function setTutorStatus(message = "") {
  $("tutor-status").textContent = message;
  $("tutor-status").hidden = !message;
}
function setTutorError(message = "") {
  $("tutor-error").textContent = message;
  $("tutor-error").hidden = !message;
}
function enterTutor(tutor, restored = false) {
  state.tutor = tutor;
  state.tutorMode = true;
  state.lastAnalysisId = tutor.question.source_analysis_id;
  clearSelection();
  parseFen(tutor.question.fen);
  $("practice-banner").hidden = false;
  $("result").hidden = true;
  $("empty-guide").hidden = true;
  $("tutor-panel").hidden = false;
  setTutorError();
  setTutorStatus(restored ? "Restored your unfinished local practice." : "");
  setStatus(
    restored
      ? "Practice restored. Your live game is unchanged."
      : "Practice mode: play the strongest move. Your live game will not change.",
  );
  renderTutor();
  $("tutor-question").focus?.();
}
function clearTutorView() {
  state.tutorMode = false;
  state.tutor = null;
  clearSelection();
  $("practice-banner").hidden = true;
  $("tutor-panel").hidden = true;
  setTutorStatus();
  setTutorError();
}
async function restoreTutor() {
  if (!state.session) return false;
  const list = await api(`/api/v1/sessions/${state.session.session_id}/tutor?limit=20`);
  const active = list.items.find((item) =>
    ["awaiting_answer", "awaiting_follow_up"].includes(item.status),
  );
  if (!active) return false;
  const source = await api(`/api/v1/analyses/${active.question.source_analysis_id}`);
  if (source.evidence && source.coaching) renderResult(source);
  enterTutor(active, true);
  return true;
}
async function beginTutor() {
  if (!state.session || !state.lastAnalysisId) return;
  const busyToken = beginBusy();
  setError();
  try {
    const context = sessionContext(),
      tutor = await api(`/api/v1/sessions/${context.sessionId}/tutor`, {
        method: "POST",
        headers: { "Idempotency-Key": idempotencyKey("tutor") },
        body: JSON.stringify({ source_analysis_id: state.lastAnalysisId }),
      });
    if (!contextIsCurrent(context)) return;
    enterTutor(tutor);
  } catch (error) {
    if (!superseded(error))
      setTutorError(`Practice could not start: ${error.message}`);
  } finally {
    endBusy(busyToken);
  }
}
async function submitTutorAnswer(move) {
  const busyToken = beginBusy();
  setTutorError();
  setTutorStatus("Grading with Stockfish…");
  try {
    await tutorCommand("answer", { move_uci: move });
    setTutorStatus();
    setStatus("Answer graded. Complete the follow-up question.");
    renderTutor();
  } catch (error) {
    if (!superseded(error))
      setTutorError(`The answer could not be graded: ${error.message}`);
  } finally {
    endBusy(busyToken);
  }
}
async function revealTutorHint() {
  setTutorError();
  setTutorStatus("Loading cited hint…");
  try {
    await tutorCommand("hint");
    setTutorStatus();
    renderTutor();
  } catch (error) {
    if (!superseded(error))
      setTutorError(`The hint could not be loaded: ${error.message}`);
  }
}
async function submitFollowUp(optionId) {
  setTutorError();
  setTutorStatus("Saving your follow-up…");
  try {
    await tutorCommand("follow_up", { option_id: optionId });
    setTutorStatus();
    setStatus("Practice complete. Your live game is still unchanged.");
    renderTutor();
  } catch (error) {
    if (!superseded(error))
      setTutorError(`The follow-up could not be submitted: ${error.message}`);
  }
}
async function leaveTutor() {
  if (!state.tutorMode) return;
  setTutorError();
  if (state.tutor && !["complete", "dismissed"].includes(state.tutor.status)) {
    setTutorStatus("Ending practice safely…");
    try {
      await tutorCommand("dismiss");
    } catch (error) {
      if (!state.tutor || !["complete", "dismissed"].includes(state.tutor.status)) {
        setTutorStatus();
        setTutorError(`Practice is still active: ${error.message}`);
        return;
      }
    }
  }
  clearTutorView();
  if (state.session) {
    parseFen(state.session.fen);
    $("result").hidden = !state.lastAnalysisId;
    $("empty-guide").hidden = Boolean(state.lastAnalysisId);
    setStatus(
      state.session.outcome
        ? `Game over: ${state.session.outcome}.`
        : `${state.session.turn === state.session.player_color ? "Your" : "Stockfish"} turn.`,
    );
    renderSession();
    $("result-title").focus();
  }
}
function evidenceText(item) {
  if (!item) return "Evidence unavailable.";
  if ("move_san" in item)
    return `${item.move_san} (${item.move_uci}), rank ${item.rank}, ${item.nodes.toLocaleString()} nodes, line ${item.pv_uci.slice(0, 8).join(" ")}`;
  if ("outcome" in item)
    return `Matched ${item.node_budget_each.toLocaleString()}-node searches: ${item.engine_move_uci} vs ${item.considered_move_uci}; ${item.outcome.replaceAll("_", " ")}.`;
  if ("concept" in item)
    return `${item.concept.replaceAll("_", " ")}: ${item.value}`;
  return `${item.fact_type.replaceAll("_", " ")}: ${item.value}`;
}
function claimText(claim, evidence) {
  const candidates = Object.fromEntries(
    (evidence.candidate_set?.candidates || []).map((item) => [
      item.evidence_id,
      item,
    ]),
  );
  if (claim.kind === "move")
    return `Recommended move: ${candidates[claim.candidate_id].move_san}.`;
  if (claim.kind === "score") {
    const item = candidates[claim.candidate_id];
    return item.mate_in !== null
      ? `Mate evaluation: ${item.mate_in}.`
      : `Evaluation from the side to move: ${(item.score_cp / 100).toFixed(2)} pawns.`;
  }
  if (claim.kind === "line")
    return `Line to calculate: ${candidates[claim.candidate_id].pv_uci.slice(claim.start_ply, claim.end_ply).join(" ")}.`;
  if (claim.kind === "comparison") {
    const item = evidence.move_comparison;
    if (item.outcome === "equal")
      return `${item.considered_move_uci} is effectively equal to ${item.engine_move_uci} within ${item.tolerance_cp} centipawns.`;
    return `The matched-budget comparison favors ${item.outcome === "engine_better" ? item.engine_move_uci : item.considered_move_uci}.`;
  }
  return claim.template_id === "compare_candidate_moves"
    ? "Compare the forcing replies to both moves."
    : "Calculate checks, captures, and threats first.";
}

function renderSession() {
  if (!state.session) return;
  if (document.activeElement !== $("fen")) $("fen").value = state.session.fen;
  parseFen(state.session.fen);
  $("undo").disabled = state.session.plies.length === 0 || state.busy;
  $("game-outcome").hidden = !state.session.outcome;
  $("game-outcome").textContent = state.session.outcome
    ? `Game over · ${state.session.outcome}`
    : "";
  const rows = [];
  state.session.plies.forEach((ply) => {
    const fields = ply.fen_before.split(/\s+/),
      number = Number(fields[5]),
      color = fields[1] === "w" ? "white" : "black";
    let row = rows.find((item) => item.number === number);
    if (!row) {
      row = { number, white: null, black: null };
      rows.push(row);
    }
    row[color] = ply;
  });
  $("move-list").replaceChildren(
    ...rows.map((row) => {
      const item = document.createElement("li"),
        number = document.createElement("span");
      number.textContent = String(row.number).padStart(2, "0");
      item.append(number);
      for (const color of ["white", "black"]) {
        const cell = document.createElement("span"),
          move = row[color],
          label = document.createElement("small");
        cell.className = "move-cell";
        cell.setAttribute(
          "aria-label",
          `${color}, ${move?.move_san || "no move"}, ${move ? (move.actor === "player" ? "you" : "Stockfish") : "empty"}`,
        );
        cell.append(document.createTextNode(move?.move_san || "…"));
        label.textContent = move
          ? move.actor === "player"
            ? "You"
            : "Stockfish"
          : color;
        cell.append(label);
        item.append(cell);
      }
      return item;
    }),
  );
  $("game-panel").hidden = rows.length === 0;
  renderBoard();
}
function restoreCommittedControls() {
  if (!state.session) return;
  $("session-mode").value = state.session.mode;
  $("difficulty").value = state.session.white_difficulty;
  $("fen").value = state.session.fen;
}
async function createSession(fen = START_FEN) {
  const epoch = ++state.sessionEpoch,
    exhibition = $("session-mode").value === "exhibition";
  state.sessionTransition = true;
  setFenError();
  renderMode();
  renderBoard();
  try {
    const created = await api("/api/v1/sessions", {
      method: "POST",
      headers: { "Idempotency-Key": idempotencyKey("session") },
      body: JSON.stringify({
        mode: exhibition ? "exhibition" : "player",
        fen,
        player_color: exhibition ? null : "white",
        white_difficulty: $("difficulty").value,
        black_difficulty: $("difficulty").value,
        rating_bucket: $("rating").value,
      }),
    });
    if (epoch !== state.sessionEpoch) {
      void api(`/api/v1/sessions/${created.session_id}`, {
        method: "DELETE",
      }).catch(() => {});
      return;
    }
    if (
      state.tutor &&
      !["complete", "dismissed"].includes(state.tutor.status)
    ) {
      try {
        await tutorCommand("dismiss");
      } catch (error) {
        await api(`/api/v1/sessions/${created.session_id}`, {
          method: "DELETE",
        }).catch(() => {});
        throw error;
      }
    }
    clearTutorView();
    stopExhibition();
    cancelReview();
    cancelAnimation();
    clearSelection();
    invalidateBusy();
    setError();
    setReviewStatus();
    state.session = created;
    state.lastMove = [];
    state.lastAnalysisId = null;
    $("practice").hidden = true;
    $("result").hidden = true;
    $("empty-guide").hidden = false;
    setStatus(
      exhibition
        ? "Engine exhibition ready."
        : `${state.session.turn === "white" ? "White" : "Black"} to move. Select a piece.`,
    );
    savePreferences();
  } catch (error) {
    if (epoch === state.sessionEpoch) {
      restoreCommittedControls();
      throw error;
    }
  } finally {
    if (epoch === state.sessionEpoch) {
      state.sessionTransition = false;
      renderMode();
      if (state.session) renderSession();
      else renderBoard();
    }
  }
}
function requestSession(fen = START_FEN) {
  void createSession(fen).catch((error) => {
    if (!superseded(error))
      setFenError(`The position could not be loaded: ${error.message}`);
  });
}
async function refreshSession() {
  if (!state.session) return;
  const context = sessionContext();
  try {
    const refreshed = await api(`/api/v1/sessions/${context.sessionId}`);
    if (!contextIsCurrent(context)) return;
    state.session = refreshed;
    renderSession();
  } catch (error) {
    if (!superseded(error) && contextIsCurrent(context))
      setError(error.message);
  }
}
function renderExhibitionControl() {
  if (!state.session || state.session.mode !== "exhibition") return;
  const button = $("exhibition");
  button.disabled =
    state.sessionTransition ||
    state.exhibitionTransition ||
    state.session.status === "complete";
  if (state.sessionTransition) button.textContent = "Loading session…";
  else if (state.exhibitionTransition) button.textContent = "Updating…";
  else if (state.exhibitionRunning) button.textContent = "Pause exhibition";
  else if (state.session.status === "complete")
    button.textContent = "Game complete";
  else if (state.session.status === "paused")
    button.textContent = "Resume exhibition";
  else
    button.textContent = state.session.plies.length
      ? "Continue exhibition"
      : "Start exhibition";
}
function renderMode() {
  const exhibition =
    !state.sessionTransition && state.session?.mode === "exhibition";
  $("exhibition").hidden = !exhibition;
  $("game-kicker").textContent = exhibition
    ? "Stockfish plays both sides and each ply is reviewed before the next."
    : "You play, Stockfish replies, and your move is reviewed automatically.";
  $("board-help").textContent = exhibition
    ? "Start, pause, or resume the engine exhibition here."
    : "Select one of your pieces, then choose a highlighted destination. Choose Q, R, B, or N when promoting.";
  if (exhibition) renderExhibitionControl();
}
function stopExhibition() {
  state.exhibitionRunning = false;
  state.exhibitionToken += 1;
  renderExhibitionControl();
}
async function toggleExhibition() {
  if (
    state.sessionTransition ||
    state.exhibitionTransition ||
    state.session?.mode !== "exhibition"
  )
    return;
  if (state.exhibitionRunning) {
    state.exhibitionTransition = true;
    state.exhibitionRunning = false;
    state.exhibitionToken += 1;
    const reviewContinues = Boolean(state.analysisId);
    cancelReview(false);
    if (reviewContinues) setReviewStatus("Review continues in the background.");
    const running = state.exhibitionPromise;
    renderExhibitionControl();
    try {
      if (running) await running;
      if (
        state.session?.mode === "exhibition" &&
        state.session.status === "active"
      ) {
        state.session = await sessionCommand("pause");
        renderSession();
        savePreferences();
      }
      setStatus(
        state.session?.status === "complete"
          ? `Game over: ${state.session.outcome}.`
          : "Engine exhibition paused.",
      );
    } finally {
      state.exhibitionTransition = false;
      renderExhibitionControl();
    }
    return;
  }
  if (state.session.status === "paused") {
    state.exhibitionTransition = true;
    renderExhibitionControl();
    try {
      state.session = await sessionCommand("resume");
      renderSession();
      savePreferences();
    } finally {
      state.exhibitionTransition = false;
    }
  }
  state.exhibitionRunning = true;
  const token = ++state.exhibitionToken;
  renderExhibitionControl();
  state.exhibitionPromise = runExhibition(token);
  try {
    await state.exhibitionPromise;
  } finally {
    state.exhibitionPromise = null;
    renderExhibitionControl();
  }
}
function setError(message = "") {
  $("input-error").textContent = message;
  $("input-error").hidden = !message;
}
function setFenError(message = "") {
  $("fen-error").textContent = message;
  $("fen-error").hidden = !message;
}
function setStatus(message) {
  $("status").textContent = message;
}
function setReviewStatus(message = "") {
  $("review-status").textContent = message;
  $("review-status").hidden = !message;
}

$("flip").addEventListener("click", () => {
  state.flipped = !state.flipped;
  renderBoard();
  savePreferences();
});
$("new-game").addEventListener("click", () => {
  requestSession();
});
$("load-example").addEventListener("click", () => {
  $("fen").value = EXAMPLE_FEN;
  requestSession(EXAMPLE_FEN);
});
$("apply-fen").addEventListener("click", () => {
  requestSession($("fen").value.trim());
});
$("analyze").addEventListener("click", () => {
  void explainPosition();
});
$("cancel").addEventListener("click", () => {
  cancelReview();
  setReviewStatus("Review cancelled.");
});
$("practice").addEventListener("click", () => {
  void beginTutor();
});
$("tutor-hint-button").addEventListener("click", () => {
  void revealTutorHint();
});
$("return-game").addEventListener("click", () => void leaveTutor());
$("end-practice").addEventListener("click", () => void leaveTutor());
$("undo").addEventListener("click", async () => {
  stopExhibition();
  cancelReview();
  try {
    state.session = await sessionCommand("undo");
    state.lastMove = [];
    renderSession();
    savePreferences();
  } catch (error) {
    setError(error.message);
  }
});
$("session-mode").addEventListener("change", () => {
  requestSession();
});
$("difficulty").addEventListener("change", () => {
  savePreferences();
  requestSession(state.session?.initial_fen || START_FEN);
});
$("rating").addEventListener("change", savePreferences);
$("exhibition").addEventListener("click", () => {
  void toggleExhibition().catch((error) => {
    if (!superseded(error))
      setError(`Exhibition control failed: ${error.message}`);
  });
});
$("promotion-dialog").addEventListener("cancel", (event) => {
  event.preventDefault();
  cancelPromotion();
});
$("promotion-cancel").addEventListener("click", cancelPromotion);
document
  .querySelectorAll("[data-promotion]")
  .forEach((button) =>
    button.addEventListener("click", () =>
      choosePromotion(button.dataset.promotion),
    ),
  );

async function initialize() {
  initializeBoard();
  parseFen(START_FEN);
  renderBoard();
  const saved = loadPreferences();
  renderMode();
  try {
    if (saved.sessionId)
      state.session = await api(`/api/v1/sessions/${saved.sessionId}`);
  } catch {
    localStorage.removeItem(STORAGE_KEY);
  }
  if (!state.session) await createSession(EXAMPLE_FEN);
  else {
    state.sessionEpoch += 1;
    $("session-mode").value = state.session.mode;
    renderMode();
    renderSession();
    setStatus(
      state.session.status === "paused"
        ? "Paused exhibition restored."
        : "Session restored from the local database.",
    );
    const reviewed = [...state.session.plies]
      .reverse()
      .find((ply) => ply.analysis_id);
    let tutorRestored = false;
    try {
      tutorRestored = await restoreTutor();
    } catch (error) {
      setTutorError(`Saved practice could not be restored: ${error.message}`);
    }
    if (reviewed && !tutorRestored)
      void pollAnalysis(reviewed.analysis_id, `Review of ${reviewed.move_san}`);
  }
  try {
    const capabilities = await api("/api/v1/capabilities");
    $("coach-mode").textContent =
      capabilities.model_status === "disabled"
        ? "Deterministic coach"
        : `Gemma 4 · ${capabilities.model_status}`;
    updateStorageBanner(capabilities.storage_status);
  } catch {
    $("coach-mode").textContent = "Coach unavailable";
  }
}
function updateStorageBanner(status) {
  const banner = $("storage-banner");
  banner.hidden = !["degraded", "corrupt"].includes(status);
  banner.querySelector("span").textContent =
    status === "corrupt"
      ? "Local history failed its integrity check. Stop, back it up, and run doctor."
      : "Local history is temporarily unavailable. The last committed state is unchanged.";
  $("retry-storage").hidden = status === "corrupt";
}
$("retry-storage").addEventListener("click", async () => {
  $("retry-storage").disabled = true;
  try {
    const result = await api("/api/v1/storage/retry", { method: "POST" });
    updateStorageBanner(result.storage_status);
  } catch (error) {
    updateStorageBanner(error.code === "STORAGE_CORRUPT" ? "corrupt" : "degraded");
  } finally {
    $("retry-storage").disabled = false;
  }
});
void initialize().catch((error) =>
  setError(`GemmaFischer could not start: ${error.message}`),
);
