// Global variables
let isLoading = false;
let gameState = null;
let selectedSquare = null;
let selectedSquareIndex = null;
let gameMode = 'analysis'; // 'analysis' or 'play'
let stockfishMatch = null;
let matchActive = false;
let lastHybridAnalysis = null;

const STOCKFISH_DEFAULTS = {
  bestDepth: 14,
  bestTimeMs: 1500,
  topDepth: 6,
  topTimeMs: 500,
  topK: 3
};
let stockfishInsightsRequestId = 0;
let stockfishInsightsPending = false;
let lastStockfishFen = null;

// Sanitize helpers with error boundaries
function sanitizeString(str) {
  try {
    if (window.DOMPurify && typeof window.DOMPurify.sanitize === 'function') {
      return DOMPurify.sanitize(str, { ALLOWED_TAGS: [], ALLOWED_ATTR: [] });
    }
    // Fallback: basic HTML entity encoding
    return str.replace(/[&<>"']/g, function(match) {
      const entityMap = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;'
      };
      return entityMap[match];
    });
  } catch (error) {
    console.warn('DOMPurify sanitizeString failed, using raw string:', error);
    return str || '';
  }
}

// -----------------
// Training Controls
// -----------------
async function startTraining() {
  const expertSel = document.getElementById('train-expert');
  const stepsInput = document.getElementById('train-steps');
  if (!expertSel || !stepsInput) return;

  const expert = expertSel.value;
  const steps = parseInt(stepsInput.value || '1000', 10);
  const useInstr = document.getElementById('train-use-instr')?.checked || false;
  const disableEval = document.getElementById('train-disable-eval')?.checked || false;

  try {
    const res = await fetch('/api/train/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ expert, steps, use_instruction: useInstr, disable_eval: disableEval })
    });
    const data = await res.json();
    const statusEl = document.getElementById('train-status');
    if (statusEl) statusEl.textContent = JSON.stringify(data, null, 2);
    if (res.ok) {
      const startBtn = document.getElementById('btn-train-start');
      if (startBtn) startBtn.disabled = true;
    }
  } catch (error) {
    const statusEl = document.getElementById('train-status');
    if (statusEl) statusEl.textContent = `Error: ${error}`;
  }
}

async function stopTraining() {
  try {
    const res = await fetch('/api/train/stop', { method: 'POST' });
    const data = await res.json();
    const statusEl = document.getElementById('train-status');
    if (statusEl) statusEl.textContent = JSON.stringify(data, null, 2);
    const startBtn = document.getElementById('btn-train-start');
    if (startBtn) startBtn.disabled = false;
  } catch (error) {
    const statusEl = document.getElementById('train-status');
    if (statusEl) statusEl.textContent = `Error: ${error}`;
  }
}

async function refreshTrainingStatus() {
  try {
    const res = await fetch('/api/train/status');
    const data = await res.json();
    const statusEl = document.getElementById('train-status');
    if (statusEl) {
      const log = data.logs_tail || '';
      statusEl.textContent = `${JSON.stringify({ ...data, logs_tail: undefined }, null, 2)}\n\n--- Logs ---\n${log}`;
    }
    const running = !!data.running;
    const meta = [];
    if (data.checkpoint_dir) meta.push(`Checkpoint: ${data.checkpoint_dir}`);
    if (data.log_file) meta.push(`Log: ${data.log_file}`);
    if (data.elapsed_sec) meta.push(`Elapsed: ${Math.round(data.elapsed_sec)}s`);
    const infoEl = document.getElementById('train-meta');
    if (infoEl) infoEl.textContent = meta.join('  |  ');
    const startBtn = document.getElementById('btn-train-start');
    if (startBtn) startBtn.disabled = running;
  } catch (error) {
    const statusEl = document.getElementById('train-status');
    if (statusEl) statusEl.textContent = `Error: ${error}`;
  }
}

// -----------------
// Evaluation Tools
// -----------------
async function startEvalStockfish() {
  const fileEl = document.getElementById('eval-file');
  const limitEl = document.getElementById('eval-limit');
  const depthEl = document.getElementById('eval-depth');
  if (!fileEl || !limitEl || !depthEl) return;

  const payload = {
    file: fileEl.value,
    limit: parseInt(limitEl.value || '100', 10),
    depth: parseInt(depthEl.value || '12', 10),
  };
  try {
    const res = await fetch('/api/eval/stockfish', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    const statusEl = document.getElementById('eval-status');
    if (statusEl) statusEl.textContent = JSON.stringify(data, null, 2);
  } catch (error) {
    const statusEl = document.getElementById('eval-status');
    if (statusEl) statusEl.textContent = `Error: ${error}`;
  }
}

async function startEvalPuzzles() {
  const fileEl = document.getElementById('puzz-file');
  const limitEl = document.getElementById('puzz-limit');
  if (!fileEl || !limitEl) return;

  const payload = {
    file: fileEl.value,
    limit: parseInt(limitEl.value || '200', 10),
  };
  try {
    const res = await fetch('/api/eval/puzzles', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    const statusEl = document.getElementById('eval-status');
    if (statusEl) statusEl.textContent = JSON.stringify(data, null, 2);
  } catch (error) {
    const statusEl = document.getElementById('eval-status');
    if (statusEl) statusEl.textContent = `Error: ${error}`;
  }
}

async function refreshEvalStatus() {
  try {
    const res = await fetch('/api/eval/status');
    const data = await res.json();
    const statusEl = document.getElementById('eval-status');
    if (statusEl) {
      const log = data.logs_tail || '';
      statusEl.textContent = `${JSON.stringify({ ...data, logs_tail: undefined }, null, 2)}\n\n--- Logs ---\n${log}`;
    }
  } catch (error) {
    const statusEl = document.getElementById('eval-status');
    if (statusEl) statusEl.textContent = `Error: ${error}`;
  }
}

async function loadEvalHistory() {
  try {
    const res = await fetch('/api/eval/history');
    const data = await res.json();
    const historyEl = document.getElementById('eval-history');
    if (historyEl) historyEl.textContent = JSON.stringify(data, null, 2);
  } catch (error) {
    const historyEl = document.getElementById('eval-history');
    if (historyEl) historyEl.textContent = `Error: ${error}`;
  }
}

// -----------------
// Dataset Utilities
// -----------------
async function cleanDataset(kind) {
  const inEl = document.getElementById(kind === 'uci' ? 'data-uci-in' : 'data-tutor-in');
  const outEl = document.getElementById(kind === 'uci' ? 'data-uci-out' : 'data-tutor-out');
  if (!inEl || !outEl) return;

  try {
    const res = await fetch('/api/data/clean', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ kind, input_path: inEl.value, output_path: outEl.value })
    });
    const data = await res.json();
    const statusEl = document.getElementById('data-status');
    if (statusEl) statusEl.textContent = JSON.stringify(data, null, 2);
  } catch (error) {
    const statusEl = document.getElementById('data-status');
    if (statusEl) statusEl.textContent = `Error: ${error}`;
  }
}

async function refreshDataStatus() {
  try {
    const res = await fetch('/api/data/status');
    const data = await res.json();
    const statusEl = document.getElementById('data-status');
    if (statusEl) {
      const log = data.logs_tail || '';
      statusEl.textContent = `${JSON.stringify({ ...data, logs_tail: undefined }, null, 2)}\n\n--- Logs ---\n${log}`;
    }
  } catch (error) {
    const statusEl = document.getElementById('data-status');
    if (statusEl) statusEl.textContent = `Error: ${error}`;
  }
}

// -----------------
// Router Diagnostics
// -----------------
async function refreshRouterDiagnostics() {
  const container = document.getElementById('router-diagnostics');
  if (!container) return;

  try {
    const [data, health] = await Promise.all([
      safeFetch('/api/router/diagnostics'),
      safeFetch('/api/engine/health').catch(() => null)
    ]);

    if (!data || !data.moe_enabled) {
      container.innerHTML = `
        <p class="text-muted small mb-0">
          Mixture-of-Experts routing is currently disabled.
        </p>
      `;
      return;
    }

    const cache = data.cache || {};
    const perf = (data.performance && data.performance.expert_metrics) || {};

    const featureHit = typeof cache.feature_cache_hit_rate === 'number'
      ? (cache.feature_cache_hit_rate * 100).toFixed(1)
      : '—';
    const routingHit = typeof cache.routing_cache_hit_rate === 'number'
      ? (cache.routing_cache_hit_rate * 100).toFixed(1)
      : '—';

    container.innerHTML = '';

    const summary = document.createElement('div');
    summary.className = 'small text-muted mb-2';
    summary.textContent = `Cache hit (features/routing): ${featureHit}% / ${routingHit}%`;
    container.appendChild(summary);

    const expertList = document.createElement('ul');
    expertList.className = 'list-unstyled small mb-0';

    Object.entries(perf).forEach(([expert, metrics]) => {
      const item = document.createElement('li');
      const accuracy = typeof metrics.accuracy === 'number'
        ? `${(metrics.accuracy * 100).toFixed(1)}%`
        : '—';
      const responseTime = typeof metrics.response_time === 'number'
        ? `${metrics.response_time.toFixed(2)}s`
        : '—';
      item.innerHTML = `
        <strong>${sanitizeString(expert)}</strong>
        <span class="text-muted ms-1">• accuracy ${accuracy} • p50 latency ${responseTime}</span>
      `;
      expertList.appendChild(item);
    });

    if (!expertList.children.length) {
      const empty = document.createElement('li');
      empty.className = 'text-muted';
      empty.textContent = 'No performance samples collected yet.';
      expertList.appendChild(empty);
    }

    container.appendChild(expertList);

    if (data.decision_log_path) {
      const logHint = document.createElement('div');
      logHint.className = 'text-muted small mt-2';
      logHint.textContent = `Decision log: ${data.decision_log_path}`;
      container.appendChild(logHint);
    }

    if (lastHybridAnalysis && lastHybridAnalysis.best_move) {
      const engineSummary = document.createElement('div');
      engineSummary.className = 'text-muted small mt-2';
      engineSummary.textContent = `LC0 last recommendation: ${lastHybridAnalysis.best_move} (${lastHybridAnalysis.engine || 'LC0'})`;
      container.appendChild(engineSummary);
    }

    if (health && health.available) {
      const healthDiv = document.createElement('div');
      healthDiv.className = 'text-muted small mt-1';
      const primary = health.primary;
      const fallback = health.fallback;
      healthDiv.textContent = `Primary engine: ${primary?.name || 'unknown'} (${primary?.engine_path || 'n/a'})`;
      if (fallback) {
        healthDiv.textContent += ` • Fallback: ${fallback.name}`;
      }
      container.appendChild(healthDiv);
    }
  } catch (error) {
    console.warn('Unable to load router diagnostics', error);
    container.innerHTML = `
      <p class="text-muted small mb-0">
        Unable to load router metrics. Check server logs for details.
      </p>
    `;
  }
}

// -----------------
// Adapter & Settings Management
// -----------------
async function refreshAdapters() {
  try {
    const res = await fetch('/api/adapters/list');
    const data = await res.json();
    const statusEl = document.getElementById('adapters-status');
    if (statusEl) statusEl.textContent = JSON.stringify(data, null, 2);

    const metaEl = document.getElementById('adapters-meta');
    if (metaEl && data.available) {
      metaEl.textContent = `Available: ${Object.keys(data.available).join(', ')}`;
    }

    const moeStatusEl = document.getElementById('moe-info');
    if (moeStatusEl) {
      try {
        const modelRes = await fetch('/api/model_info');
        const modelData = await modelRes.json();
        if (modelData.moe_enabled && modelData.moe_available) {
          moeStatusEl.style.display = 'block';
          const info = [];
          info.push(`Enabled: ${modelData.moe_enabled}`);
          if (modelData.moe_experts) info.push(`Experts: ${modelData.moe_experts.join(', ')}`);
          moeStatusEl.innerHTML = info.join('<br>');
        } else {
          moeStatusEl.style.display = 'none';
        }
      } catch (err) {
        console.warn('Unable to fetch MoE info', err);
      }
    }
  } catch (error) {
    const statusEl = document.getElementById('adapters-status');
    if (statusEl) statusEl.textContent = `Error: ${error}`;
  }
}

async function listAdapters() {
  try {
    const res = await fetch('/api/adapters/list');
    const data = await res.json();
    const statusEl = document.getElementById('adapters-status');
    if (statusEl) statusEl.textContent = JSON.stringify(data, null, 2);
  } catch (error) {
    const statusEl = document.getElementById('adapters-status');
    if (statusEl) statusEl.textContent = `Error: ${error}`;
  }
}

async function activateAdapter() {
  const adapterName = prompt('Enter adapter name to activate:');
  if (!adapterName) return;
  try {
    const res = await fetch('/api/adapters/activate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: adapterName })
    });
    const data = await res.json();
    const statusEl = document.getElementById('adapters-status');
    if (statusEl) statusEl.textContent = JSON.stringify(data, null, 2);
  } catch (error) {
    const statusEl = document.getElementById('adapters-status');
    if (statusEl) statusEl.textContent = `Error: ${error}`;
  }
}

async function loadSettings() {
  try {
    const res = await fetch('/api/settings/get');
    const data = await res.json();
    const policyEl = document.getElementById('engine-policy');
    if (policyEl && data.engine_policy) policyEl.value = data.engine_policy;
    const rerankEl = document.getElementById('engine-rerank');
    if (rerankEl && typeof data.engine_rerank !== 'undefined') rerankEl.checked = !!data.engine_rerank;
    const constrainEl = document.getElementById('engine-constrain');
    if (constrainEl && typeof data.engine_constrain !== 'undefined') constrainEl.checked = !!data.engine_constrain;
    const moeEl = document.getElementById('moe-enabled');
    if (moeEl && typeof data.moe_enabled !== 'undefined') moeEl.checked = !!data.moe_enabled;
  } catch (error) {
    console.warn('Failed to load adapter settings', error);
  }
}

async function saveSettings() {
  const policyEl = document.getElementById('engine-policy');
  const rerankEl = document.getElementById('engine-rerank');
  const constrainEl = document.getElementById('engine-constrain');
  const moeEl = document.getElementById('moe-enabled');

  const payload = {
    engine_policy: policyEl ? policyEl.value : undefined,
    engine_rerank: rerankEl ? rerankEl.checked : undefined,
    engine_constrain: constrainEl ? constrainEl.checked : undefined,
    moe_enabled: moeEl ? moeEl.checked : undefined,
  };

  try {
    const res = await fetch('/api/settings/set', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    const statusEl = document.getElementById('adapters-status');
    if (statusEl) statusEl.textContent = JSON.stringify(data, null, 2);
  } catch (error) {
    const statusEl = document.getElementById('adapters-status');
    if (statusEl) statusEl.textContent = `Error: ${error}`;
  }
}

function sanitizeHTML(html) {
  try {
    if (window.DOMPurify && typeof window.DOMPurify.sanitize === 'function') {
      return DOMPurify.sanitize(html, { RETURN_DOM_FRAGMENT: true });
    }
    // Fallback: create text node only
    const fragment = document.createDocumentFragment();
    fragment.appendChild(document.createTextNode(html || ''));
    return fragment;
  } catch (error) {
    console.warn('DOMPurify sanitizeHTML failed, using text fragment:', error);
    const fragment = document.createDocumentFragment();
    fragment.appendChild(document.createTextNode(html || ''));
    return fragment;
  }
}

function getStockfishInsightsBody() {
  return document.getElementById('stockfish-insights-body');
}

// Loading states and feedback functions
function showLoadingSpinner(elementId, message = 'Loading...') {
  const element = document.getElementById(elementId);
  if (!element) return;

  element.innerHTML = `
    <div class="d-flex align-items-center justify-content-center p-3">
      <div class="spinner-border spinner-border-sm me-2" role="status">
        <span class="visually-hidden">Loading...</span>
      </div>
      <span>${sanitizeString(message)}</span>
    </div>
  `;
}

function hideLoadingSpinner(elementId) {
  const element = document.getElementById(elementId);
  if (!element) return;

  // Clear loading state but don't remove content
  const spinner = element.querySelector('.spinner-border');
  if (spinner && spinner.parentElement) {
    spinner.parentElement.remove();
  }
}

function showErrorMessage(elementId, message, type = 'danger') {
  const element = document.getElementById(elementId);
  if (!element) return;

  const alertClass = type === 'warning' ? 'alert-warning' : 'alert-danger';
  element.innerHTML = `
    <div class="alert ${alertClass} alert-dismissible fade show" role="alert">
      <i class="fas fa-exclamation-triangle me-2"></i>
      ${sanitizeString(message)}
      <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    </div>
  `;
}

function showSuccessMessage(elementId, message) {
  const element = document.getElementById(elementId);
  if (!element) return;

  element.innerHTML = `
    <div class="alert alert-success alert-dismissible fade show" role="alert">
      <i class="fas fa-check-circle me-2"></i>
      ${sanitizeString(message)}
      <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    </div>
  `;

  // Auto-dismiss after 3 seconds
  setTimeout(() => {
    const alert = element.querySelector('.alert');
    if (alert) {
      alert.classList.remove('show');
      setTimeout(() => alert.remove(), 150);
    }
  }, 3000);
}

function updateStatusIndicator(status, message) {
  const indicator = document.getElementById('modelStatusBanner');
  const textElement = document.getElementById('modelStatusText');

  if (!indicator || !textElement) return;

  // Remove existing classes
  indicator.classList.remove('alert-info', 'alert-success', 'alert-warning', 'alert-danger', 'd-none');

  if (status === 'loading') {
    indicator.classList.add('alert-info');
    textElement.textContent = message || 'Loading...';
  } else if (status === 'success') {
    indicator.classList.add('alert-success');
    textElement.textContent = message || 'Operation completed successfully';
  } else if (status === 'warning') {
    indicator.classList.add('alert-warning');
    textElement.textContent = message || 'Warning';
  } else if (status === 'error') {
    indicator.classList.add('alert-danger');
    textElement.textContent = message || 'An error occurred';
  }

  // Show the banner
  indicator.classList.remove('d-none');
}

// API response validation and safe fetch wrapper
async function safeFetch(url, options = {}) {
  try {
    // Set default headers for JSON requests
    const defaultHeaders = {
      'Content-Type': 'application/json',
      ...options.headers
    };

    const response = await fetch(url, {
      ...options,
      headers: defaultHeaders
    });

    if (!response.ok) {
      let errorMessage = `HTTP ${response.status}: ${response.statusText}`;

      try {
        const errorData = await response.json();
        if (errorData.message) {
          errorMessage = errorData.message;
        }
      } catch (parseError) {
        // Ignore JSON parse errors for error responses
      }

      throw new Error(errorMessage);
    }

    // Try to parse JSON response
    const contentType = response.headers.get('content-type');
    if (contentType && contentType.includes('application/json')) {
      const data = await response.json();
      return data;
    } else {
      // Return text for non-JSON responses
      return await response.text();
    }

  } catch (error) {
    console.error(`API call failed: ${url}`, error);

    // Re-throw with more context
    if (error.name === 'TypeError' && error.message.includes('fetch')) {
      throw new Error('Network error: Unable to connect to server');
    }

    throw error;
  }
}

// Helper function to handle API errors consistently
function handleApiError(error, context = 'operation') {
  console.error(`Error during ${context}:`, error);

  let userMessage = 'An unexpected error occurred. Please try again.';

  if (error.message) {
    if (error.message.includes('Network error')) {
      userMessage = 'Network error: Please check your connection and try again.';
    } else if (error.message.includes('404')) {
      userMessage = 'The requested resource was not found.';
    } else if (error.message.includes('500')) {
      userMessage = 'Server error: Please try again later.';
    } else if (error.message.includes('timeout')) {
      userMessage = 'Request timed out. Please try again.';
    } else {
      userMessage = error.message;
    }
  }

  // Show error in status indicator
  updateStatusIndicator('error', userMessage);

  return userMessage;
}

function setStockfishInsightsPlaceholder(message = 'Include a FEN in your question or analyze the board to see engine suggestions.') {
  const body = getStockfishInsightsBody();
  if (!body) return;
  stockfishInsightsPending = false;
  stockfishInsightsRequestId += 1; // invalidate in-flight requests
  body.innerHTML = '';
  const placeholder = document.createElement('p');
  placeholder.className = 'stockfish-empty mb-0';
  placeholder.textContent = message;
  body.appendChild(placeholder);
  lastStockfishFen = null;
}

function setStockfishInsightsLoading(contextText = null) {
  const body = getStockfishInsightsBody();
  if (!body) return;
  body.innerHTML = '';
  const wrapper = document.createElement('div');
  wrapper.className = 'stockfish-loading';
  const spinner = document.createElement('div');
  spinner.className = 'loading';
  wrapper.appendChild(spinner);
  const text = document.createElement('span');
  text.textContent = contextText
    ? `Analyzing ${contextText} with Stockfish…`
    : 'Analyzing with Stockfish…';
  wrapper.appendChild(text);
  body.appendChild(wrapper);
}

function setStockfishInsightsError(message) {
  const body = getStockfishInsightsBody();
  if (!body) return;
  body.innerHTML = '';
  const errorDiv = document.createElement('div');
  errorDiv.className = 'stockfish-empty text-danger';
  errorDiv.textContent = message;
  body.appendChild(errorDiv);
}

function updateEngineAnalysis(payload) {
  const body = document.getElementById('lc0-analysis-body');
  if (!body) return;

  if (!payload || !payload.best_move) {
    body.innerHTML = '<p class="text-muted small mb-0">No LC0 analysis available.</p>';
    lastHybridAnalysis = null;
    return;
  }

  lastHybridAnalysis = payload;

  const evalText = payload.mate_in !== null && payload.mate_in !== undefined
    ? `Mate in ${payload.mate_in}`
    : (payload.evaluation_pawns !== null && payload.evaluation_pawns !== undefined
      ? `${payload.evaluation_pawns} pawns`
      : (payload.evaluation_cp !== null && payload.evaluation_cp !== undefined
        ? `${(payload.evaluation_cp / 100).toFixed(2)} pawns`
        : 'N/A'));

  const pvText = (payload.principal_variation && payload.principal_variation.length)
    ? payload.principal_variation.join(' ')
    : '—';

  const keyPoints = (payload.key_points || [])
    .map(point => `<li>${sanitizeString(point)}</li>`)
    .join('');

  const explanation = payload.explanation ? sanitizeString(payload.explanation) : '';

  body.innerHTML = `
    <div class="lc0-meta mb-2">
      <div>Engine: <strong>${sanitizeString(payload.engine || 'LC0')}</strong>${payload.fallback_used ? ' (fallback)' : ''}</div>
      <div>Computation time: ${payload.engine_time ? payload.engine_time.toFixed(2) : '—'} s</div>
    </div>
    <div class="lc0-move-highlight mb-2">Best move: ${sanitizeString(payload.best_move || '—')}</div>
    <div class="lc0-meta mb-2">Evaluation: ${sanitizeString(evalText)}</div>
    <div class="lc0-meta mb-2">Principal variation: ${sanitizeString(pvText)}</div>
    ${explanation ? `<div class="lc0-explanation mb-2">${explanation}</div>` : ''}
    ${keyPoints ? `<ul class="small lc0-key-points">${keyPoints}</ul>` : ''}
  `;
}

async function requestHybridAnalysis() {
  const fen = getCurrentBoardFEN();
  const intentSelect = document.getElementById('strategic-intent');
  const intent = intentSelect ? intentSelect.value : null;

  try {
    const response = await fetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ fen, intent })
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || 'LC0 analysis failed');
    }
    updateEngineAnalysis(data);
    updateStatusIndicator('success', 'LC0 analysis complete.');
  } catch (error) {
    console.error('Hybrid analysis failed:', error);
    updateEngineAnalysis(null);
    updateStatusIndicator('error', `Hybrid analysis failed: ${error.message || error}`);
  }
}

function extractFenCandidate(text) {
  if (!text) return null;
  const fenRegex = /(?:FEN[:\s]*)?(([prnbqkPRNBQK1-8]+\/){7}[prnbqkPRNBQK1-8]+\s+[wb]\s+[KQkq-]{1,4}\s+[a-h1-8-]+\s+\d+\s+\d+)/i;
  const match = text.match(fenRegex);
  return match ? match[1].trim() : null;
}

function questionLikelyHasFen(text) {
  return !!extractFenCandidate(text);
}

function formatStockfishScore(entry) {
  if (!entry) return '—';
  if (typeof entry.mate === 'number' && entry.mate !== 0) {
    const mateAbs = Math.abs(entry.mate);
    return entry.mate > 0 ? `M${mateAbs}` : `M-${mateAbs}`;
  }
  if (typeof entry.score_cp === 'number') {
    const value = (entry.score_cp / 100).toFixed(2);
    return entry.score_cp >= 0 ? `+${value}` : value;
  }
  return '—';
}

function renderStockfishInsights(payload) {
  const body = getStockfishInsightsBody();
  if (!body) return;
  stockfishInsightsPending = false;
  body.innerHTML = '';

  const bestMove = payload?.best;
  if (!bestMove) {
    setStockfishInsightsError('No engine analysis available.');
    return;
  }

  const header = document.createElement('div');
  header.className = 'd-flex flex-column flex-sm-row justify-content-between align-items-sm-center';

  const bestContainer = document.createElement('div');
  bestContainer.className = 'stockfish-best-move';
  const bestLabel = document.createElement('span');
  bestLabel.textContent = 'Best move: ';
  const bestBadge = document.createElement('span');
  bestBadge.className = 'badge bg-primary';
  bestBadge.textContent = sanitizeString(bestMove.san || bestMove.uci || '—');
  bestContainer.appendChild(bestLabel);
  bestContainer.appendChild(bestBadge);

  const bestMeta = document.createElement('div');
  bestMeta.className = 'stockfish-meta mt-2 mt-sm-0';
  const duration = payload.analysis_duration_ms != null ? (payload.analysis_duration_ms / 1000).toFixed(2) : '—';
  let generatedAtText = '';
  if (payload.generated_at) {
    const generatedDate = new Date(payload.generated_at);
    if (!Number.isNaN(generatedDate.valueOf())) {
      generatedAtText = ` • ${generatedDate.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
    }
  }
  bestMeta.textContent = `Depth ${payload.best_depth} (top ${payload.top_depth}) • ${duration}s${generatedAtText}`;

  header.appendChild(bestContainer);
  header.appendChild(bestMeta);
  body.appendChild(header);

  const table = document.createElement('table');
  table.className = 'table table-sm table-striped mt-3 stockfish-move-table';
  const thead = document.createElement('thead');
  const headerRow = document.createElement('tr');
  ['#', 'Move', 'Score', 'PV'].forEach(text => {
    const th = document.createElement('th');
    th.scope = 'col';
    th.textContent = text;
    headerRow.appendChild(th);
  });
  thead.appendChild(headerRow);
  table.appendChild(thead);

  const tbody = document.createElement('tbody');
  const topMoves = Array.isArray(payload.top_moves) ? payload.top_moves : [];
  if (topMoves.length === 0) {
    const emptyRow = document.createElement('tr');
    const emptyCell = document.createElement('td');
    emptyCell.colSpan = 4;
    emptyCell.className = 'text-muted';
    emptyCell.textContent = 'No alternative moves returned.';
    emptyRow.appendChild(emptyCell);
    tbody.appendChild(emptyRow);
  } else {
    topMoves.slice(0, STOCKFISH_DEFAULTS.topK).forEach((entry, idx) => {
      const row = document.createElement('tr');
      if (entry.uci && bestMove.uci && entry.uci === bestMove.uci) {
        row.classList.add('table-success');
      }
      const rankCell = document.createElement('td');
      rankCell.textContent = `${idx + 1}`;
      row.appendChild(rankCell);

      const moveCell = document.createElement('td');
      const moveStrong = document.createElement('strong');
      moveStrong.textContent = sanitizeString(entry.san || entry.uci || '—');
      moveCell.appendChild(moveStrong);
      const moveBadge = document.createElement('span');
      moveBadge.className = 'badge bg-light text-muted border ms-2';
      moveBadge.textContent = `depth ${entry.depth ?? payload.top_depth}`;
      moveCell.appendChild(moveBadge);
      if (entry.uci && entry.san && entry.san !== entry.uci) {
        const moveSmall = document.createElement('div');
        moveSmall.className = 'text-muted small';
        moveSmall.textContent = entry.uci;
        moveCell.appendChild(moveSmall);
      }
      row.appendChild(moveCell);

      const scoreCell = document.createElement('td');
      const scoreSpan = document.createElement('span');
      const scoreText = formatStockfishScore(entry);
      if (scoreText.startsWith('+') || scoreText.startsWith('M')) {
        scoreSpan.className = 'stockfish-score-positive';
      } else if (scoreText.startsWith('-')) {
        scoreSpan.className = 'stockfish-score-negative';
      } else {
        scoreSpan.className = 'text-muted';
      }
      scoreSpan.textContent = scoreText;
      scoreCell.appendChild(scoreSpan);
      row.appendChild(scoreCell);

      const pvCell = document.createElement('td');
      const pvMoves = (entry.pv_san && entry.pv_san.length ? entry.pv_san : (entry.pv || [])).slice(0, 6);
      const pvText = pvMoves.length ? pvMoves.join(' ') : '—';
      const pvSpan = document.createElement('span');
      pvSpan.className = 'text-muted';
      pvSpan.textContent = pvText;
      pvCell.appendChild(pvSpan);
      row.appendChild(pvCell);

      tbody.appendChild(row);
    });
  }

  table.appendChild(tbody);
  body.appendChild(table);

  if (payload.fen) {
    const fenBlock = document.createElement('div');
    fenBlock.className = 'stockfish-meta mt-2';
    fenBlock.textContent = `FEN: ${payload.fen}`;
    body.appendChild(fenBlock);
    lastStockfishFen = payload.fen;
  }
}

function requestStockfishInsights({ question = null, fen = null } = {}) {
  const body = {
    question,
    fen,
    best_depth: STOCKFISH_DEFAULTS.bestDepth,
    best_time_limit_ms: STOCKFISH_DEFAULTS.bestTimeMs,
    top_depth: STOCKFISH_DEFAULTS.topDepth,
    top_time_limit_ms: STOCKFISH_DEFAULTS.topTimeMs,
    top_k: STOCKFISH_DEFAULTS.topK
  };

  stockfishInsightsPending = true;
  const requestId = ++stockfishInsightsRequestId;

  fetch('/api/analysis/top_moves', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  })
    .then(async response => {
      const data = await response.json().catch(() => ({}));
      if (requestId !== stockfishInsightsRequestId) {
        return;
      }
      if (!response.ok) {
        if (response.status === 400) {
          setStockfishInsightsPlaceholder(data.error || 'No FEN detected for analysis.');
        } else {
          setStockfishInsightsError(data.error || 'Stockfish analysis failed.');
        }
        return;
      }
      renderStockfishInsights(data);
    })
    .catch(error => {
      console.error('Stockfish insights error:', error);
      if (requestId === stockfishInsightsRequestId) {
        setStockfishInsightsError('Unable to fetch Stockfish analysis.');
      }
    })
    .finally(() => {
      if (requestId === stockfishInsightsRequestId) {
        stockfishInsightsPending = false;
      }
    });
}

function triggerStockfishInsightsFromQuestion(question) {
  if (!questionLikelyHasFen(question)) {
    setStockfishInsightsPlaceholder();
    return;
  }
  const fenCandidate = extractFenCandidate(question);
  const fenPreview = fenCandidate ? `${fenCandidate.split(' ').slice(0, 3).join(' ')} …` : null;
  setStockfishInsightsLoading(fenPreview);
  requestStockfishInsights({ question });
}

function analyzeCurrentPositionWithStockfish() {
  const fen = getCurrentBoardFEN();
  if (!fen) {
    setStockfishInsightsError('Unable to determine current board position.');
    return;
  }
  if (fen === lastStockfishFen && !stockfishInsightsPending) {
    setStockfishInsightsPlaceholder('Analysis already reflects the current board.');
    return;
  }
  const fenPreview = `${fen.split(' ').slice(0, 3).join(' ')} …`;
  setStockfishInsightsLoading(fenPreview);
  requestStockfishInsights({ fen });
}

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
  // Progressive enhancement and feature detection
  const features = {
    fetch: typeof fetch !== 'undefined',
    promises: typeof Promise !== 'undefined',
    domPurify: typeof DOMPurify !== 'undefined',
    localStorage: typeof localStorage !== 'undefined',
    serviceWorker: 'serviceWorker' in navigator
  };

  // Check for required features
  let missingFeatures = [];
  Object.entries(features).forEach(([feature, available]) => {
    if (!available) {
      missingFeatures.push(feature);
      console.warn(`Feature not available: ${feature}`);
    }
  });

  // Warn about critical missing features
  if (missingFeatures.length > 0) {
    const criticalFeatures = ['fetch', 'promises'];
    const missingCritical = missingFeatures.filter(f => criticalFeatures.includes(f));

    if (missingCritical.length > 0) {
      updateStatusIndicator('error',
        `Critical features not supported: ${missingCritical.join(', ')}. Please use a modern browser.`);
      return; // Don't initialize if critical features are missing
    }

    // Non-critical warnings
    if (missingFeatures.includes('domPurify')) {
      console.warn('DOMPurify not loaded - using fallback sanitization');
    }
  }

  // Initialize application
  try {
    initializeChessBoard();
    loadExamples();
    setupEventListeners();
    loadGameState();

    // Welcome
    showMessage('🎮 **ChessGemma Ready!**\n\nClick squares to analyze positions or toggle Play Mode to start a game!', 'success');

    setStockfishInsightsPlaceholder();
    const analyzeBtn = document.getElementById('stockfish-analyze-btn');
    if (analyzeBtn) {
      analyzeBtn.addEventListener('click', () => analyzeCurrentPositionWithStockfish());
    }

    // Initialize real-time status updates
    startStatusUpdates();

  } catch (error) {
    console.error('Failed to initialize application:', error);
    updateStatusIndicator('error', 'Failed to initialize application. Please refresh the page.');
  }
});

// Initialize chess board grid
function initializeChessBoard() {
  const board = document.getElementById('chessBoard');
  if (!board) return;
  board.innerHTML = '';
  for (let i = 0; i < 64; i++) {
    const square = document.createElement('div');
    square.className = `chess-square ${(i + Math.floor(i / 8)) % 2 === 0 ? 'light' : 'dark'}`;
    square.onclick = () => handleSquareClick(i);
    board.appendChild(square);
  }
}

// Load example questions
async function loadExamples() {
  try {
    const response = await fetch('/api/examples');
    const data = await response.json();
    const container = document.getElementById('examplesContainer');
    if (!container) return;
    data.examples.forEach(example => {
      const button = document.createElement('button');
      button.className = 'example-btn';
      button.textContent = example;
      button.onclick = () => useExample(example);
      container.appendChild(button);
    });
  } catch (error) {
    console.error('Failed to load examples:', error);
  }
}

function useExample(question) {
  const input = document.getElementById('questionInput');
  if (!input) return;
  input.value = question;
  askQuestion();
}

function handleKeyPress(event) {
  if (event.key === 'Enter' && !isLoading) {
    askQuestion();
  }
}

// Keyboard navigation support
function handleQuestionKeydown(event) {
  // Allow Enter to submit (but not Shift+Enter for multiline)
  if (event.key === 'Enter' && !event.shiftKey && !isLoading) {
    event.preventDefault();
    askQuestion();
    return;
  }

  // Ctrl+Enter or Cmd+Enter to submit
  if (event.key === 'Enter' && (event.ctrlKey || event.metaKey) && !isLoading) {
    event.preventDefault();
    askQuestion();
    return;
  }

  // Escape to clear input
  if (event.key === 'Escape') {
    const input = event.target;
    if (input && input.value) {
      input.value = '';
      updateStatusIndicator('info', 'Input cleared');
    }
    return;
  }

  // Arrow keys for history navigation (future enhancement)
  // For now, just prevent default behavior on some keys
}

function setupEventListeners() {
  const startBtn = document.getElementById('btn-train-start');
  const stopBtn = document.getElementById('btn-train-stop');
  const refreshBtn = document.getElementById('btn-train-refresh');
  const questionInput = document.getElementById('questionInput');

  if (startBtn && typeof startTraining === 'function') {
    startBtn.addEventListener('click', startTraining);
  }
  if (stopBtn && typeof stopTraining === 'function') {
    stopBtn.addEventListener('click', stopTraining);
  }
  if (refreshBtn && typeof refreshTrainingStatus === 'function') {
    refreshBtn.addEventListener('click', refreshTrainingStatus);
  }

  // Keyboard navigation support
  if (questionInput) {
    questionInput.addEventListener('keydown', handleQuestionKeydown);
    questionInput.setAttribute('tabindex', '1');
    questionInput.setAttribute('aria-label', 'Enter your chess question');
  }

  if (typeof refreshTrainingStatus === 'function') {
    setInterval(() => {
      if (!document.hidden) {
        refreshTrainingStatus();
      }
    }, 15000);
  }

  if (typeof refreshRouterDiagnostics === 'function') {
    refreshRouterDiagnostics();
    setInterval(() => {
      if (!document.hidden) {
        refreshRouterDiagnostics();
      }
    }, 20000);
  }

  if (typeof refreshAdapters === 'function') {
    refreshAdapters();
  }
  if (typeof loadSettings === 'function') {
    loadSettings();
  }
}

// Ask a question
async function askQuestion() {
  if (isLoading) return;
  const input = document.getElementById('questionInput');
  if (!input) return;
  const question = input.value.trim();
  if (!question) {
    showMessage('Please enter a question.', 'error');
    return;
  }
  addMessage(question, 'user');
  input.value = '';
  isLoading = true;

  const loadingDiv = document.createElement('div');
  loadingDiv.className = 'message assistant';
  loadingDiv.innerHTML = `
    <div class="d-flex align-items-center">
      <i class="fas fa-robot me-2"></i>
      <strong>ChessGemma</strong>
      <div class="loading ms-2"></div>
    </div>
    <p>Thinking about your question...</p>
  `;
  document.getElementById('chatMessages').appendChild(loadingDiv);
  scrollToBottom();

  triggerStockfishInsightsFromQuestion(question);

  try {
    const expert = 'auto'; // Always use auto mode for intelligent routing
    const response = await fetch('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, context: '', expert })
    });
    const data = await response.json();
    loadingDiv.remove();
    let messageClass = 'assistant';

    if (data.analysis) {
      updateEngineAnalysis(data.analysis);
    } else if (data.best_move || data.principal_variation) {
      updateEngineAnalysis(data);
    }

    const container = document.createElement('div');

    const headerDiv = document.createElement('div');
    headerDiv.className = 'd-flex align-items-center mb-2';
    const botIcon = document.createElement('i');
    botIcon.className = 'fas fa-robot me-2';
    headerDiv.appendChild(botIcon);
    const strongEl = document.createElement('strong');
    strongEl.textContent = 'ChessGemma';
    headerDiv.appendChild(strongEl);
    if (data.expert && data.expert !== 'auto') {
      const expertBadge = document.createElement('span');
      expertBadge.className = 'badge bg-secondary ms-2';
      expertBadge.textContent = sanitizeString(data.expert);
      headerDiv.appendChild(expertBadge);
    }
    container.appendChild(headerDiv);

    const responsePara = document.createElement('p');
    const safeResponse = sanitizeHTML(data.response || data.error || 'No response received');
    responsePara.appendChild(safeResponse);
    container.appendChild(responsePara);

    if (data.confidence) {
      let confidenceClass = 'confidence-low';
      if (data.confidence > 0.7) confidenceClass = 'confidence-high';
      else if (data.confidence > 0.4) confidenceClass = 'confidence-medium';
      const confidenceDiv = document.createElement('div');
      confidenceDiv.className = `confidence-badge ${confidenceClass}`;
      confidenceDiv.textContent = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;
      container.appendChild(confidenceDiv);
    }

    if (data.moe_used) {
      const moeDiv = document.createElement('div');
      moeDiv.className = 'moe-info small text-muted mt-2 p-2 bg-light rounded';
      const moeIcon = document.createElement('i');
      moeIcon.className = 'fas fa-network-wired me-1';
      moeDiv.appendChild(moeIcon);
      const moeStrong = document.createElement('strong');
      moeStrong.textContent = 'MoE Routing:';
      moeDiv.appendChild(moeStrong);
      moeDiv.appendChild(document.createTextNode(` ${sanitizeString(data.primary_expert || 'auto')}`));
      if (data.ensemble_mode) {
        moeDiv.appendChild(document.createTextNode(' (ensemble)'));
      }
      if (data.routing_reasoning) {
        moeDiv.appendChild(document.createElement('br'));
        const em = document.createElement('em');
        em.textContent = sanitizeString(data.routing_reasoning);
        moeDiv.appendChild(em);
      }
      container.appendChild(moeDiv);
      updateExpertStatus(data);
    }

    addMessage(container, messageClass);
  } catch (error) {
    console.error('Error:', error);
    loadingDiv.remove();
    addMessage('Sorry, I encountered an error while processing your question. Please try again.', 'error');
    updateEngineAnalysis(null);
  } finally {
    isLoading = false;
  }
}

function addMessage(content, type = 'assistant') {
  const messagesDiv = document.getElementById('chatMessages');
  const messageDiv = document.createElement('div');
  messageDiv.className = `message ${type}`;
  if (typeof content === 'string') {
    messageDiv.textContent = sanitizeString(content);
  } else if (content instanceof Node || content instanceof DocumentFragment) {
    messageDiv.appendChild(content);
  }
  messagesDiv.appendChild(messageDiv);
  scrollToBottom();
}

function scrollToBottom() {
  const messagesDiv = document.getElementById('chatMessages');
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function handleSquareClick(squareIndex) {
  const file = String.fromCharCode(97 + (squareIndex % 8));
  const rank = 8 - Math.floor(squareIndex / 8);
  const square = `${file}${rank}`;
  if (gameMode === 'play') {
    handlePlayModeClick(square, squareIndex);
  } else {
    handleAnalysisModeClick(square, squareIndex);
  }
}

function handleAnalysisModeClick(square, squareIndex) {
  const squares = document.querySelectorAll('.chess-square');
  squares.forEach(sq => sq.classList.remove('selected'));
  squares[squareIndex].classList.add('selected');
  const piece = squares[squareIndex].textContent;
  const pieceName = getPieceName(piece);
  const message = piece ? `Selected square ${square} with ${pieceName}` : `Selected empty square ${square}`;
  showMessage(message, 'info', 3000);
  setTimeout(() => {
    const currentFEN = getCurrentBoardFEN();
    const question = piece ? `FEN: ${currentFEN}\nQuestion: What can the ${pieceName} on ${square} do?` : `FEN: ${currentFEN}\nQuestion: What pieces can move to ${square}?`;
    const input = document.getElementById('questionInput');
    if (input) input.value = question;
    askQuestion();
  }, 1000);
}

function handlePlayModeClick(square, squareIndex) {
  const squares = document.querySelectorAll('.chess-square');
  if (selectedSquare === null) {
    selectedSquare = square;
    selectedSquareIndex = squareIndex;
    squares.forEach(sq => sq.classList.remove('selected'));
    squares[squareIndex].classList.add('selected');
    getLegalMoves(square);
  } else {
    if (square === selectedSquare) {
      // Deselect if clicked again
      squares.forEach(sq => sq.classList.remove('selected', 'legal-move'));
      selectedSquare = null;
      selectedSquareIndex = null;
      return;
    }
    let move = `${selectedSquare}${square}`;
    // Promotion handling
    const fromPiece = squares[selectedSquareIndex]?.textContent;
    const destRank = parseInt(square[1]);
    if ((fromPiece === '♙' && destRank === 8) || (fromPiece === '♟' && destRank === 1)) {
      let promo = (window.prompt('Promote to (q,r,b,n)?', 'q') || 'q').toLowerCase();
      if (!['q','r','b','n'].includes(promo)) promo = 'q';
      move = `${move}${promo}`;
    }
    // Clear highlights before sending
    const lm = document.querySelectorAll('.legal-move');
    lm.forEach(el => el.classList.remove('legal-move'));
    makeMove(move);
    selectedSquare = null;
    selectedSquareIndex = null;
  }
}

function getCurrentBoardFEN() {
  if (gameState && gameState.fen) return gameState.fen;
  return 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
}

function getPieceName(piece) {
  const pieceNames = {
    '♜': 'Black Rook', '♞': 'Black Knight', '♝': 'Black Bishop',
    '♛': 'Black Queen', '♚': 'Black King', '♟': 'Black Pawn',
    '♖': 'White Rook', '♘': 'White Knight', '♗': 'White Bishop',
    '♕': 'White Queen', '♔': 'White King', '♙': 'White Pawn'
  };
  return pieceNames[piece] || 'Unknown Piece';
}

function showMessage(message, type = 'info', duration = 0) {
  addMessage(message, type);
  if (duration > 0) {
    setTimeout(() => {
      const messages = document.querySelectorAll('.message');
      if (messages.length > 0) {
        messages[messages.length - 1].remove();
      }
    }, duration);
  }
}

async function loadGameState() {
  try {
    const response = await fetch('/api/game/state');
    gameState = await response.json();
    // Update model loaded banner
    try {
      const infoResp = await fetch('/api/model_info');
      const info = await infoResp.json();
      const banner = document.querySelector('#modelStatusBanner');
      const loadedText = info.loaded ? '✅ Model loaded' : '⚠️ Model not loaded';
      if (banner) {
        banner.textContent = loadedText;
      } else {
        const msg = document.createElement('div');
        msg.className = 'message info';
        msg.id = 'modelStatusBanner';
        msg.innerText = loadedText;
        const chat = document.getElementById('chatMessages');
        if (chat) chat.prepend(msg);
      }
    } catch (e) { /* ignore */ }
    if (gameState && gameState.fen) {
      initializeChessBoard();
      updateBoardFromFEN(gameState.fen);
    }
  } catch (error) {
    console.error('Failed to load game state:', error);
  }
}

async function makeMove(moveUCI) {
  try {
    const response = await fetch('/api/game/move', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ move: moveUCI })
    });
    const result = await response.json();
    if (result.success) {
      const moveText = result.san || result.move;
      const playerText = result.current_player === 'white' ? 'White' : 'Black';
      showMessage(`✅ ${playerText} played: ${moveText}`, 'success');
      updateBoardFromFEN(result.fen);
      gameState = result;
      selectedSquare = null;
      selectedSquareIndex = null;
      const squares = document.querySelectorAll('.chess-square');
      squares.forEach(sq => sq.classList.remove('selected', 'legal-move'));
      if (result.current_player === 'black' && gameMode === 'play') {
        showMessage('🤖 AI is thinking...', 'info', 3000);
        setTimeout(() => getAIMove(), 2000);
      }
    } else {
      showMessage(`❌ Invalid move: ${result.error}`, 'danger');
    }
  } catch (error) {
    console.error('Move error:', error);
    showMessage('❌ Error making move', 'danger');
  }
}

async function getAIMove() {
  try {
    showMessage('AI is thinking...', 'info', 2000);
    const response = await fetch('/api/game/ai_move', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ expert: 'auto' })
    });
    const result = await response.json();
    if (result.success) {
      const moveText = result.san || result.move;
      let aiMessage = `🤖 **AI played: ${moveText}**`;
      if (result.ai_response) {
        aiMessage += `\n\n💭 **AI Reasoning:**\n${result.ai_response}`;
      }
      showMessage(aiMessage, 'info');
      updateBoardFromFEN(result.fen);
      gameState = result;
    } else {
      showMessage(`❌ AI error: ${result.error}`, 'danger');
    }
  } catch (error) {
    console.error('AI move error:', error);
    showMessage('❌ Error getting AI move', 'danger');
  }
}

async function makeHybridAIMove() {
  if (gameMode !== 'play') {
    showMessage('🤖 Hybrid AI moves only available in Play Mode', 'warning');
    return;
  }

  if (isLoading) {
    showMessage('⏳ AI is already thinking...', 'warning');
    return;
  }

  try {
    isLoading = true;
    showMessage('🤖 Hybrid AI is analyzing position...', 'info');

    // Get selected strategic intent
    const intentSelect = document.getElementById('strategic-intent');
    const strategicIntent = intentSelect ? intentSelect.value : 'positional';

    showMessage(`🎯 AI analyzing with ${strategicIntent} strategy...`, 'info', 3000);

    const response = await fetch('/api/game/ai_move', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        expert: 'hybrid',
        strategic_intent: strategicIntent
      })
    });

    const result = await response.json();

    if (result.success) {
      const moveText = result.san || result.move;

      // Enhanced hybrid response display
      let aiMessage = `🤖 **AI played: ${moveText}**\n\n`;

      if (result.hybrid_analysis) {
        const analysis = result.hybrid_analysis;
        aiMessage += `🎯 **Strategy**: ${analysis.strategic_guidance?.intent || strategicIntent}\n`;
        aiMessage += `🎚️ **Confidence**: ${((analysis.confidence || 0) * 100).toFixed(0)}%\n`;
        aiMessage += `⏱️ **Analysis**: ${(analysis.total_time || 0).toFixed(1)}s `;

        if (analysis.llm_time && analysis.lc0_time) {
          aiMessage += `(LLM: ${(analysis.llm_time).toFixed(1)}s, LC0: ${(analysis.lc0_time).toFixed(1)}s)`;
        }
        aiMessage += '\n\n';
      }

      if (result.ai_response) {
        aiMessage += `💭 **Analysis**:\n${result.ai_response}`;
      }

      showMessage(aiMessage, 'success');
      updateBoardFromFEN(result.fen);
      gameState = result;
    } else {
      showMessage(`❌ Hybrid AI error: ${result.error}`, 'danger');
    }

  } catch (error) {
    showMessage(`❌ Hybrid AI error: ${error}`, 'danger');
  } finally {
    isLoading = false;
  }
}

async function getLegalMoves(square) {
  try {
    const response = await fetch('/api/game/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ square })
    });
    const analysis = await response.json();
    highlightLegalMoves(analysis.legal_moves);
    let message = `🔍 **${analysis.piece_name} on ${square}**\n`;
    message += `**Legal moves:** ${analysis.legal_moves.join(', ')}`;
    if (analysis.rag_advice && analysis.rag_advice.length > 0) {
      message += `\n\n📚 **Chess Knowledge:**\n${analysis.rag_advice.join('\n')}`;
    }
    showMessage(message, 'info');
  } catch (error) {
    console.error('Analysis error:', error);
  }
}

function highlightLegalMoves(legalMoves) {
  const squares = document.querySelectorAll('.chess-square');
  squares.forEach(sq => sq.classList.remove('legal-move'));
  legalMoves.forEach(move => {
    const toSquare = move.slice(2, 4);
    const file = toSquare.charCodeAt(0) - 97;
    const rank = 8 - parseInt(toSquare[1]);
    const squareIndex = rank * 8 + file;
    if (squareIndex >= 0 && squareIndex < 64) {
      squares[squareIndex].classList.add('legal-move');
    }
  });
}

function updateBoardFromFEN(fen) {
  const fenParts = fen.split(' ');
  const boardState = fenParts[0];
  const currentPlayer = fenParts[1];
  const squares = document.querySelectorAll('.chess-square');
  squares.forEach(sq => {
    sq.textContent = '';
    sq.classList.remove('selected', 'legal-move', 'engine-highlight');
  });
  // Validate FEN piece placement has 8 ranks
  const ranks = boardState.split('/');
  if (ranks.length !== 8) return;
  let rank = 0;
  let file = 0;
  for (let i = 0; i < boardState.length; i++) {
    const char = boardState[i];
    if (char === '/') {
      rank++;
      file = 0;
    } else if (char >= '1' && char <= '8') {
      file += parseInt(char);
    } else {
      const squareIndex = rank * 8 + file;
      if (squareIndex < 64) {
        squares[squareIndex].textContent = getPieceSymbol(char);
      }
      file++;
    }
  }
  updateGameStateDisplay(currentPlayer);
}

function getPieceSymbol(fenChar) {
  const pieceMap = {
    'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',
    'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟'
  };
  return pieceMap[fenChar] || '';
}

function updateGameStateDisplay(currentPlayer) {
  const playerText = currentPlayer === 'w' ? 'White' : 'Black';
  showMessage(`${playerText} to move`, 'info', 2000);
}

function toggleGameMode() {
  gameMode = gameMode === 'analysis' ? 'play' : 'analysis';
  const modeText = gameMode === 'play' ? 'Play Mode' : 'Analysis Mode';
  const modeIcon = gameMode === 'play' ? '🎮' : '🔍';
  showMessage(`${modeIcon} Switched to ${modeText}`, 'info', 3000);
  const toggleButton = document.querySelector('button[onclick="toggleGameMode()"]');
  if (toggleButton) {
    toggleButton.classList.remove('play-mode-active', 'analysis-mode-active');
    if (gameMode === 'play') {
      toggleButton.classList.add('play-mode-active');
      toggleButton.innerHTML = '<i class="fas fa-gamepad me-1"></i>Exit Play Mode';
      // Show hybrid controls in play mode
      const hybridControls = document.getElementById('hybrid-controls');
      if (hybridControls) hybridControls.style.display = 'block';
    } else {
      toggleButton.classList.add('analysis-mode-active');
      toggleButton.innerHTML = '<i class="fas fa-search me-1"></i>Enter Play Mode';
      // Hide hybrid controls in analysis mode
      const hybridControls = document.getElementById('hybrid-controls');
      if (hybridControls) hybridControls.style.display = 'none';
    }
  }
  selectedSquare = null;
  selectedSquareIndex = null;
  const squares = document.querySelectorAll('.chess-square');
  squares.forEach(sq => sq.classList.remove('selected', 'legal-move'));
}

async function resetGame() {
  try {
    const response = await fetch('/api/game/reset', { method: 'POST', headers: { 'Content-Type': 'application/json' } });
    const result = await response.json();
    if (result.success) {
      showMessage('🔄 Game reset to starting position', 'success');
      selectedSquare = null;
      selectedSquareIndex = null;
      const squares = document.querySelectorAll('.chess-square');
      squares.forEach(sq => sq.classList.remove('selected', 'legal-move'));
      await loadGameState();
      if (gameState && gameState.fen) updateBoardFromFEN(gameState.fen);
      else initializeChessBoard();
    }
  } catch (error) {
    console.error('Reset error:', error);
    showMessage('Error resetting game', 'error', 3000);
  }
}

// Stockfish match helpers
async function testStockfish() {
  try {
    showMessage('🔍 Testing Stockfish availability...', 'info');
    const response = await fetch('/api/match/test');
    const result = await response.json();
    if (result.success) {
      showMessage(`✅ ${result.message}\n📍 Path: ${result.path}\n🎯 Test move: ${result.test_move}`, 'success');
    } else {
      showMessage(`❌ Stockfish test failed: ${result.error}`, 'danger');
    }
  } catch (error) {
    console.error('Stockfish test error:', error);
    showMessage('❌ Error testing Stockfish', 'danger');
  }
}

async function toggleStockfishMatch() {
  if (matchActive) await stopStockfishMatch(); else await startStockfishMatch();
}

async function startStockfishMatch() {
  try {
    showMessage('🎮 Starting Stockfish vs Model match...', 'info');
    const response = await fetch('/api/match/start', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_plays_white: true, time_control: '10+0.1' })
    });
    const result = await response.json();
    if (result.success) {
      matchActive = true;
      stockfishMatch = result;
      showMessage(`🏆 ${result.message}`, 'success');
      const button = document.querySelector('button[onclick="toggleStockfishMatch()"]');
      if (button) {
        button.innerHTML = '<i class="fas fa-stop me-1"></i>Stop Match';
        button.className = 'btn btn-sm btn-danger';
      }
      playMatchMoves();
    } else {
      showMessage(`❌ Failed to start match: ${result.error}`, 'danger');
    }
  } catch (error) {
    console.error('Match start error:', error);
    showMessage('❌ Error starting match', 'danger');
  }
}

async function stopStockfishMatch() {
  try {
    const response = await fetch('/api/match/stop', { method: 'POST', headers: { 'Content-Type': 'application/json' } });
    const result = await response.json();
    if (result.success) {
      matchActive = false;
      stockfishMatch = null;
      showMessage('🛑 Match stopped', 'info');
      const button = document.querySelector('button[onclick="toggleStockfishMatch()"]');
      if (button) {
        button.innerHTML = '<i class="fas fa-chess me-1"></i>Stockfish Match';
        button.className = 'btn btn-sm btn-warning';
      }
    }
  } catch (error) {
    console.error('Match stop error:', error);
  }
}

async function playMatchMoves() {
  if (!matchActive) return;
  try {
    const response = await fetch('/api/match/play', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_plays_white: true })
    });
    const result = await response.json();
    if (result.success) {
      const player = result.player;
      const move = result.san;
      const time = result.time_taken.toFixed(2);
      showMessage(`🏆 ${player} played: ${move} (${time}s)`, 'info');
      updateBoardFromFEN(result.fen);
      if (result.is_game_over) {
        const gameResult = result.game_result;
        showMessage(`🏁 Game Over! Winner: ${gameResult[0].toUpperCase()}, Reason: ${gameResult[1]}`, 'success');
        matchActive = false;
        const button = document.querySelector('button[onclick="toggleStockfishMatch()"]');
        if (button) {
          button.innerHTML = '<i class="fas fa-chess me-1"></i>Stockfish Match';
          button.className = 'btn btn-sm btn-warning';
        }
      } else {
        setTimeout(() => playMatchMoves(), 1000);
      }
    } else {
      showMessage(`❌ Match error: ${result.error}`, 'danger');
      matchActive = false;
    }
  } catch (error) {
    console.error('Match play error:', error);
    showMessage('❌ Error playing match', 'danger');
    matchActive = false;
  }
}

// Update the expert status display based on MoE routing
function updateExpertStatus(data) {
  try {
    // Update MoE status badge
    const moeStatusEl = document.getElementById('moe-status');
    if (moeStatusEl) {
      moeStatusEl.textContent = data.moe_used ? 'MoE Active' : 'Single Expert';
      moeStatusEl.className = data.moe_used ? 'badge bg-success ms-2' : 'badge bg-secondary ms-2';
    }

    // Update routing indicator
    const routingEl = document.getElementById('routing-indicator');
    if (routingEl) {
      if (data.primary_expert) {
        routingEl.textContent = `Using: ${data.primary_expert} expert`;
      } else {
        routingEl.textContent = 'Auto-routing active';
      }
    }

    // Highlight the active expert badge
    const badges = document.querySelectorAll('#expert-badges .badge');
    badges.forEach(badge => {
      badge.classList.remove('bg-success');
      badge.classList.add('bg-secondary');
    });

    // Highlight the expert that was used
    if (data.primary_expert) {
      const expertBadge = document.querySelector(`#expert-badges .badge[data-expert="${data.primary_expert}"]`);
      if (expertBadge) {
        expertBadge.classList.remove('bg-secondary');
        expertBadge.classList.add('bg-success');
      }
    }

    console.log('Expert status updated:', data.primary_expert, data.moe_used);
  } catch (error) {
    console.error('Error updating expert status:', error);
  }
}

// Real-time status updates
let statusUpdateInterval = null;

function startStatusUpdates() {
  // Initial status check
  updateSystemStatus();

  // Set up periodic status updates (every 30 seconds)
  statusUpdateInterval = setInterval(updateSystemStatus, 30000);

  // Update status when page becomes visible
  document.addEventListener('visibilitychange', function() {
    if (!document.hidden) {
      updateSystemStatus();
    }
  });
}

function stopStatusUpdates() {
  if (statusUpdateInterval) {
    clearInterval(statusUpdateInterval);
    statusUpdateInterval = null;
  }
}

async function updateSystemStatus() {
  try {
    const healthData = await safeFetch('/api/health');
    const statsData = await safeFetch('/api/stats');

    // Update connection status
    if (healthData && healthData.status === 'ok') {
      updateStatusIndicator('success', 'System online');
    }

    // Update performance metrics if available
    if (statsData && statsData.performance) {
      const perf = statsData.performance;
      console.log(`📊 System Status: ${perf.total_requests} requests, ${perf.avg_response_time?.toFixed(2)}s avg response time`);
    }

  } catch (error) {
    console.warn('Status update failed:', error);
    updateStatusIndicator('warning', 'Connection issues detected');
  }
}

// Initialize expert status on page load
document.addEventListener('DOMContentLoaded', function() {
  // Set data attributes for expert badges
  const badges = document.querySelectorAll('#expert-badges .badge');
  badges[0].setAttribute('data-expert', 'uci');
  badges[1].setAttribute('data-expert', 'tutor');
  badges[2].setAttribute('data-expert', 'director');

  console.log('Expert status display initialized');
});
