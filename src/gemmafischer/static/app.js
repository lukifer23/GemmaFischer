const pieces = {p:'♟',r:'♜',n:'♞',b:'♝',q:'♛',k:'♚',P:'♙',R:'♖',N:'♘',B:'♗',Q:'♕',K:'♔'};
const state = { flipped:false, selected:null, squares:{}, analysisId:null, generation:0, poll:null };
const $ = (id) => document.getElementById(id);

function parseFen(fen) {
  const fields = fen.trim().split(/\s+/);
  if (fields.length !== 6) throw new Error(`FEN requires six fields; received ${fields.length}.`);
  const ranks = fields[0].split('/');
  if (ranks.length !== 8) throw new Error('FEN board requires eight ranks.');
  const squares = {};
  ranks.forEach((rank, r) => {
    let file = 0;
    for (const char of rank) {
      if (/\d/.test(char)) file += Number(char);
      else { if (!pieces[char] || file > 7) throw new Error('FEN contains an invalid piece placement.'); squares['abcdefgh'[file] + (8-r)] = char; file++; }
    }
    if (file !== 8) throw new Error('Every FEN rank must describe eight squares.');
  });
  state.squares = squares;
  return fields;
}

function renderBoard() {
  const board = $('board'); board.replaceChildren();
  const files = state.flipped ? [...'hgfedcba'] : [...'abcdefgh'];
  const ranks = state.flipped ? [1,2,3,4,5,6,7,8] : [8,7,6,5,4,3,2,1];
  ranks.forEach((rank, row) => files.forEach((file, col) => {
    const square = file + rank;
    const button = document.createElement('button');
    button.type = 'button'; button.className = `square ${(row+col)%2 ? 'dark':'light'}`;
    button.setAttribute('role','gridcell'); button.dataset.square = square;
    const piece = state.squares[square];
    button.setAttribute('aria-label', `${piece ? pieces[piece] + ' on ' : 'empty '}${square}`);
    button.innerHTML = `${piece ? `<span aria-hidden="true">${pieces[piece]}</span>` : ''}<span class="coordinate" aria-hidden="true">${square}</span>`;
    if (state.selected === square) button.classList.add('selected');
    button.addEventListener('click', () => selectSquare(square));
    button.addEventListener('keydown', boardKeydown);
    board.append(button);
  }));
}

function selectSquare(square) {
  if (document.querySelector('input[name="mode"]:checked').value !== 'compare') return;
  if (!state.selected) { state.selected = square; renderBoard(); return; }
  $('consider').value = state.selected + square;
  state.selected = null; renderBoard(); $('consider').focus();
}

function boardKeydown(event) {
  const keys = {ArrowLeft:-1,ArrowRight:1,ArrowUp:-8,ArrowDown:8};
  if (!(event.key in keys)) return;
  event.preventDefault();
  const cells = [...document.querySelectorAll('.square')];
  const next = Math.max(0, Math.min(63, cells.indexOf(event.currentTarget)+keys[event.key]));
  cells[next].focus();
}

function setError(message='') { $('input-error').textContent=message; $('input-error').hidden=!message; }
function activeMode() { return document.querySelector('input[name="mode"]:checked').value; }

document.querySelectorAll('input[name="mode"]').forEach(input => input.addEventListener('change', () => {
  $('consider-wrap').hidden = activeMode() !== 'compare';
  $('analyze').textContent = activeMode() === 'compare' ? 'Compare my move' : 'Analyze position';
  state.selected = null; renderBoard();
}));
$('fen').addEventListener('change', () => { try { parseFen($('fen').value); setError(); renderBoard(); } catch(e) { setError(e.message); } });
$('flip').addEventListener('click', () => { state.flipped=!state.flipped; renderBoard(); });
$('load-example').addEventListener('click', () => { $('fen').value='r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3'; parseFen($('fen').value); renderBoard(); setError(); });
$('cancel').addEventListener('click', async () => { if(state.analysisId) await fetch(`/api/v1/analyses/${state.analysisId}`,{method:'DELETE'}); stopPolling(); setStatus('Analysis cancelled. Your position is preserved.'); });
$('analyze').addEventListener('click', analyze);

async function analyze() {
  setError();
  try { parseFen($('fen').value); renderBoard(); } catch(e) { setError(e.message); $('fen').focus(); return; }
  const mode=activeMode(), considered=$('consider').value.trim().toLowerCase();
  if(mode==='compare' && !/^[a-h][1-8][a-h][1-8][qrbn]?$/.test(considered)) { setError('Enter a considered move in UCI notation, such as f1b5.'); $('consider').focus(); return; }
  const response=await fetch('/api/v1/analyses',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode,fen:$('fen').value.trim(),rating_bucket:$('rating').value,considered_move_uci:mode==='compare'?considered:null})});
  const payload=await response.json();
  if(!response.ok){ setError(payload.error?.message || 'The analysis could not start.'); return; }
  state.analysisId=payload.analysis_id; state.generation=payload.generation;
  $('result').hidden=true; $('cancel').hidden=false; $('analyze').disabled=true;
  setStatus('Queued for local analysis.');
  state.poll=setInterval(poll,500); await poll();
}

async function poll() {
  const response=await fetch(`/api/v1/analyses/${state.analysisId}`); const data=await response.json();
  if(data.generation!==state.generation) return;
  const labels={queued:'Queued.',validating:'Checking the position.',engine_running:'Analyzing candidate moves.',comparison_running:'Comparing your move.',model_running:'Preparing the explanation.',complete:'Analysis complete.',engine_only:'Engine analysis complete; model coaching was unavailable.',cancelled:'Analysis cancelled.',failed:'Analysis failed.'};
  setStatus(labels[data.state]||data.state);
  if(['complete','engine_only','failed','cancelled'].includes(data.state)) {
    stopPolling();
    if(data.state==='failed'){ setError(data.error?.message||'Analysis failed.'); return; }
    if(data.evidence&&data.coaching) renderResult(data);
  }
}
function stopPolling(){ if(state.poll) clearInterval(state.poll); state.poll=null; $('cancel').hidden=true; $('analyze').disabled=false; }
function setStatus(message){ $('status').textContent=message; }

function renderResult(data) {
  $('result').hidden=false; $('summary').textContent=data.coaching.summary;
  $('degraded').hidden=data.state!=='engine_only'; $('degraded').textContent=data.state==='engine_only'?'Verified engine evidence is available. Gemma coaching was unavailable, so this result uses the deterministic coach.':'';
  const byId=Object.fromEntries([...data.evidence.candidates,...data.evidence.board_facts].map(item=>[item.evidence_id,item]));
  $('claims').replaceChildren(...data.coaching.claims.map((claim,index)=>{
    const wrap=document.createElement('div'); wrap.className='claim';
    const text=document.createElement('p'); text.textContent=claimText(claim,data.evidence); wrap.append(text);
    if(claim.evidence_ids.length){ const button=document.createElement('button'); button.type='button'; button.textContent='Show cited evidence'; button.setAttribute('aria-expanded','false'); const detail=document.createElement('div'); detail.className='claim-evidence'; detail.hidden=true; detail.textContent=claim.evidence_ids.map(id=>JSON.stringify(byId[id]||{missing:id},null,2)).join('\n'); button.addEventListener('click',()=>{detail.hidden=!detail.hidden;button.setAttribute('aria-expanded',String(!detail.hidden));}); wrap.append(button,detail); }
    return wrap;
  }));
  $('evidence').replaceChildren(...data.evidence.candidates.map(item=>{const div=document.createElement('div');div.className='candidate';div.textContent=`${item.rank}. ${item.move_san} · ${item.score_cp!==null?(item.score_cp/100).toFixed(2):'mate '+item.mate_in} · ${item.pv_uci.slice(0,8).join(' ')}`;return div;}));
  $('result-title').textContent=activeMode()==='compare'?'Your move comparison':'Your position lesson';
  $('result-title').focus?.();
}
function claimText(claim,evidence){ const c=Object.fromEntries(evidence.candidates.map(x=>[x.evidence_id,x])); if(claim.kind==='move')return`Recommended move: ${c[claim.candidate_id].move_san}.`; if(claim.kind==='score'){const x=c[claim.candidate_id];return x.mate_in!==null?`Mate evaluation: ${x.mate_in}.`:`Evaluation from the side to move: ${(x.score_cp/100).toFixed(2)} pawns.`;} if(claim.kind==='line')return`Line to calculate: ${c[claim.candidate_id].pv_uci.slice(claim.start_ply,claim.end_ply).join(' ')}.`; if(claim.kind==='comparison'){const best=c[claim.better_candidate_id],considered=c[claim.considered_candidate_id];return best.evidence_id===considered.evidence_id?`Your move ${considered.move_san} matches the engine's first choice.`:`Compare ${best.move_san} with ${considered.move_san}.`;} return claim.template_id==='compare_candidate_moves'?'Compare the forcing replies to both moves.':'Calculate checks, captures, and threats first.'; }

parseFen($('fen').value); renderBoard();
