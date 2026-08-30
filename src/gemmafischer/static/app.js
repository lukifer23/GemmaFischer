const pieces={p:'♟',r:'♜',n:'♞',b:'♝',q:'♛',k:'♚',P:'♙',R:'♖',N:'♘',B:'♗',Q:'♕',K:'♔'};
const START_FEN='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
const EXAMPLE_FEN='r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3';
const SESSION_KEY='gemmafischer.session.v1';
const state={
  flipped:false,selected:null,legalTargets:[],squares:{},turn:'w',busy:false,lastMove:[],
  history:[],moveRows:[],analysisId:null,reviewToken:0,exhibitionRunning:false,exhibitionToken:0,
};
const squareNodes=new Map();
const $=id=>document.getElementById(id);
const wait=milliseconds=>new Promise(resolve=>setTimeout(resolve,milliseconds));

function persistSession(){
  try{localStorage.setItem(SESSION_KEY,JSON.stringify({schema:1,fen:$('fen').value,history:state.history,moveRows:state.moveRows,flipped:state.flipped,mode:$('session-mode').value,difficulty:$('difficulty').value,rating:$('rating').value}));}catch{/* A private browser may deny local storage. */}
}
function restoreSession(){
  try{
    const saved=JSON.parse(localStorage.getItem(SESSION_KEY)||'null');if(!saved||saved.schema!==1)return false;
    parseFen(saved.fen);$('fen').value=saved.fen;state.history=Array.isArray(saved.history)?saved.history:[];state.moveRows=Array.isArray(saved.moveRows)?saved.moveRows:[];state.flipped=Boolean(saved.flipped);
    if(['play','exhibition'].includes(saved.mode))$('session-mode').value=saved.mode;if(['casual','club','strong'].includes(saved.difficulty))$('difficulty').value=saved.difficulty;if(['1000-1199','1200-1399','1400-1599','1600-1800'].includes(saved.rating))$('rating').value=saved.rating;
    const exhibition=$('session-mode').value==='exhibition';$('exhibition').hidden=!exhibition;$('game-kicker').textContent=exhibition?'Stockfish plays both sides. Every move is reviewed before the next.':'You move, Stockfish replies, and your move is reviewed automatically.';$('board-help').textContent=exhibition?'Start the exhibition to advance one reviewed engine move at a time.':'Select one of your pieces, then its destination. Promotions become queens.';
    $('undo').disabled=state.history.length===0;return true;
  }catch{localStorage.removeItem(SESSION_KEY);return false;}
}

function parseFen(fen){
  const fields=fen.trim().split(/\s+/);
  if(fields.length!==6)throw new Error(`FEN requires six fields; received ${fields.length}.`);
  const ranks=fields[0].split('/');if(ranks.length!==8)throw new Error('FEN board requires eight ranks.');
  const squares={};
  ranks.forEach((rank,row)=>{let file=0;for(const char of rank){
    if(/\d/.test(char))file+=Number(char);
    else{if(!pieces[char]||file>7)throw new Error('FEN contains invalid piece placement.');squares['abcdefgh'[file]+(8-row)]=char;file++;}
  }if(file!==8)throw new Error('Every FEN rank must describe eight squares.');});
  if(!['w','b'].includes(fields[1]))throw new Error('FEN side to move must be w or b.');
  state.squares=squares;state.turn=fields[1];return fields;
}

function initializeBoard(){
  for(const rank of [8,7,6,5,4,3,2,1])for(const file of 'abcdefgh'){
    const square=file+rank,button=document.createElement('button'),piece=document.createElement('span'),coordinate=document.createElement('span');
    button.type='button';button.dataset.square=square;button.setAttribute('role','gridcell');
    piece.className='piece';piece.setAttribute('aria-hidden','true');coordinate.className='coordinate';coordinate.setAttribute('aria-hidden','true');coordinate.textContent=square;
    button.append(piece,coordinate);button.addEventListener('click',()=>selectSquare(square));button.addEventListener('keydown',boardKeydown);squareNodes.set(square,button);
  }
}

function renderBoard(){
  const board=$('board'),files=state.flipped?[...'hgfedcba']:[...'abcdefgh'],ranks=state.flipped?[1,2,3,4,5,6,7,8]:[8,7,6,5,4,3,2,1];
  board.classList.toggle('busy',state.busy);board.classList.toggle('automated',state.exhibitionRunning);board.setAttribute('aria-busy',String(state.busy));
  const ordered=[];
  ranks.forEach((rank,row)=>files.forEach((file,col)=>{
    const square=file+rank,button=squareNodes.get(square),piece=state.squares[square],legal=state.legalTargets.includes(square);
    button.className=`square ${(row+col)%2?'dark':'light'}`;
    if(state.selected===square)button.classList.add('selected');if(state.lastMove.includes(square))button.classList.add('last-move');if(legal)button.classList.add(piece?'legal-capture':'legal-target');
    button.querySelector('.piece').textContent=piece?pieces[piece]:'';
    button.setAttribute('aria-label',`${piece?pieces[piece]+' on ':'empty '}${square}${legal?', legal destination':''}`);ordered.push(button);
  }));
  const current=[...board.children];
  if(current.length!==ordered.length||ordered.some((node,index)=>current[index]!==node))board.replaceChildren(...ordered);
}

async function showMove(fen,moveUci){
  const source=squareNodes.get(moveUci.slice(0,2)).getBoundingClientRect(),destinationNode=squareNodes.get(moveUci.slice(2,4)),destination=destinationNode.getBoundingClientRect();
  $('fen').value=fen;parseFen(fen);state.lastMove=[moveUci.slice(0,2),moveUci.slice(2,4)];renderBoard();
  if(matchMedia('(prefers-reduced-motion: reduce)').matches)return;
  const movingPiece=destinationNode.querySelector('.piece');
  const animation=movingPiece.animate([{transform:`translate(${source.left-destination.left}px,${source.top-destination.top}px) scale(.94)`,zIndex:4},{transform:'translate(0,0) scale(1)',zIndex:4}],{duration:220,easing:'cubic-bezier(.2,.8,.2,1)'});
  persistSession();try{await animation.finished;}catch{/* A newer move superseded the animation. */}
}

function pieceMatchesTurn(piece){return Boolean(piece)&&(state.turn==='w'?piece===piece.toUpperCase():piece===piece.toLowerCase());}
async function selectSquare(square){
  if(state.busy||state.exhibitionRunning)return;
  const piece=state.squares[square];
  if(!state.selected){if(!pieceMatchesTurn(piece)){setError(`Select a ${state.turn==='w'?'white':'black'} piece to move.`);return;}await selectSource(square);return;}
  if(pieceMatchesTurn(piece)){await selectSource(square);return;}
  if(!state.legalTargets.includes(square)){setError(`${square} is not a legal destination for ${state.selected}.`);return;}
  const move=state.selected+square;state.selected=null;state.legalTargets=[];renderBoard();await playPlayerTurn(move);
}

async function selectSource(square){
  setError();state.selected=square;state.legalTargets=[];renderBoard();
  try{
    const response=await fetch('/api/v1/board/legal-moves',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({fen:$('fen').value.trim(),from_square:square})});
    const data=await response.json();if(state.selected!==square)return;
    if(!response.ok){state.selected=null;setError(data.error?.message||'Legal moves could not be loaded.');}
    else{state.legalTargets=data.destinations;if(!data.destinations.length)setError(`${square} has no legal moves.`);}
  }catch(error){if(state.selected===square){state.selected=null;setError(`Legal moves could not be loaded: ${error.message}`);}}
  renderBoard();
}

function boardKeydown(event){const keys={ArrowLeft:-1,ArrowRight:1,ArrowUp:-8,ArrowDown:8};if(!(event.key in keys))return;event.preventDefault();const cells=[...document.querySelectorAll('.square')],next=Math.max(0,Math.min(63,cells.indexOf(event.currentTarget)+keys[event.key]));cells[next].focus();}
function setError(message=''){$('input-error').textContent=message;$('input-error').hidden=!message;}
function setStatus(message){$('status').textContent=message;}
function setReviewStatus(message=''){$('review-status').textContent=message;$('review-status').hidden=!message;}

function saveHistory(fen){state.history.push({fen,rows:structuredClone(state.moveRows)});$('undo').disabled=false;}
function recordPly(fenBefore,san){
  const fields=fenBefore.split(/\s+/),color=fields[1]==='w'?'white':'black',number=Number(fields[5]);let row=state.moveRows.find(item=>item.number===number);
  if(!row){row={number,white:'',black:''};state.moveRows.push(row);}row[color]=san;renderMoveList();
}
function renderMoveList(){
  $('move-list').replaceChildren(...state.moveRows.map(row=>{const item=document.createElement('li');[String(row.number).padStart(2,'0'),row.white||'…',row.black||'…'].forEach(value=>{const span=document.createElement('span');span.textContent=value;item.append(span);});return item;}));
  $('game-panel').hidden=state.moveRows.length===0;
}

async function playPlayerTurn(move){
  setError();state.busy=true;renderBoard();setStatus('Move submitted. Stockfish is thinking…');
  try{
    const response=await fetch('/api/v1/board/moves',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({fen:$('fen').value.trim(),move_uci:move,engine_reply:true,difficulty:$('difficulty').value})});
    const data=await response.json();if(!response.ok){setError(data.error?.message||'The move could not be played.');setStatus(`${state.turn==='w'?'White':'Black'} to move.`);return;}
    saveHistory(data.fen_before);recordPly(data.fen_before,data.human_move_san);await showMove(data.fen_after_human,data.human_move_uci);
    if(data.engine_move_uci){await wait(70);recordPly(data.fen_after_human,data.engine_move_san);await showMove(data.fen,data.engine_move_uci);}
    if(data.game_over){showOutcome(data.outcome);setStatus(`Game over: ${data.outcome}.`);}
    else setStatus(`You played ${data.human_move_san}. ${data.engine_name} replied ${data.engine_move_san}. ${data.turn==='white'?'White':'Black'} to move.`);
    void reviewMove(data.fen_before,data.human_move_uci,'Reviewing your move');
  }catch(error){setError(`The local service could not play the turn: ${error.message}`);}
  finally{state.busy=false;renderBoard();}
}

async function runExhibition(token){
  while(state.exhibitionRunning&&token===state.exhibitionToken){
    state.busy=true;renderBoard();setStatus(`${state.turn==='w'?'White':'Black'} engine is thinking…`);
    try{
      const response=await fetch('/api/v1/board/engine-turn',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({fen:$('fen').value.trim(),difficulty:$('difficulty').value})});
      const data=await response.json();if(!response.ok){setError(data.error?.message||'The engine turn failed.');break;}if(!state.exhibitionRunning||token!==state.exhibitionToken)break;
      saveHistory(data.fen_before);recordPly(data.fen_before,data.move_san);await showMove(data.fen,data.move_uci);setStatus(`${data.engine_name} played ${data.move_san}.`);
      state.busy=false;renderBoard();await reviewMove(data.fen_before,data.move_uci,`Reviewing ${data.move_san}`);
      if(data.game_over){showOutcome(data.outcome);break;}await wait(450);
    }catch(error){setError(`Engine exhibition stopped: ${error.message}`);break;}
    finally{state.busy=false;renderBoard();}
  }
  if(token===state.exhibitionToken){state.exhibitionRunning=false;$('exhibition').textContent='Start exhibition';renderBoard();}
}

function showOutcome(outcome){$('game-outcome').textContent=`Game over · ${outcome}`;$('game-outcome').hidden=false;}
function resetSession(fen=START_FEN){
  state.exhibitionRunning=false;state.exhibitionToken++;$('exhibition').textContent='Start exhibition';cancelReview();$('fen').value=fen;parseFen(fen);state.selected=null;state.legalTargets=[];state.lastMove=[];state.history=[];state.moveRows=[];
  $('undo').disabled=true;$('game-outcome').hidden=true;$('result').hidden=true;$('empty-guide').hidden=false;renderMoveList();setError();setReviewStatus();renderBoard();setStatus(`${state.turn==='w'?'White':'Black'} to move. Select a piece.`);
  persistSession();
}

async function reviewMove(fen,move,label){return submitAnalysis({mode:'compare',fen,rating_bucket:$('rating').value,considered_move_uci:move},label);}
async function explainPosition(){
  try{parseFen($('fen').value);renderBoard();}catch(error){setError(error.message);$('fen').focus();return;}
  await submitAnalysis({mode:'position',fen:$('fen').value.trim(),rating_bucket:$('rating').value,considered_move_uci:null},'Explaining this position');
}
async function submitAnalysis(payload,label){
  cancelReview();const token=++state.reviewToken;setReviewStatus(`${label}…`);$('cancel').hidden=false;$('analyze').disabled=true;
  try{
    const response=await fetch('/api/v1/analyses',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}),created=await response.json();
    if(!response.ok){setReviewStatus(created.error?.message||'The review could not start.');return null;}state.analysisId=created.analysis_id;
    while(token===state.reviewToken){
      const pollResponse=await fetch(`/api/v1/analyses/${created.analysis_id}`),data=await pollResponse.json();
      const labels={queued:'Review queued.',validating:'Checking the position.',engine_running:'Stockfish is calculating.',comparison_running:'Comparing the move.',model_running:'Gemma is preparing the lesson.',complete:'Review complete.',engine_only:'Engine review complete. Gemma was unavailable.',cancelled:'Review cancelled.',failed:'Review failed.'};setReviewStatus(labels[data.state]||data.state);
      if(['complete','engine_only','failed','cancelled'].includes(data.state)){if(data.state==='failed')setError(data.error?.message||'Review failed.');else if(data.evidence&&data.coaching)renderResult(data);return data;}await wait(750);
    }
  }catch(error){setReviewStatus(`Review interrupted: ${error.message}`);}finally{if(token===state.reviewToken){$('cancel').hidden=true;$('analyze').disabled=false;state.analysisId=null;}}
  return null;
}

function cancelReview(send=true){state.reviewToken++;if(send&&state.analysisId)void fetch(`/api/v1/analyses/${state.analysisId}`,{method:'DELETE'});state.analysisId=null;$('cancel').hidden=true;$('analyze').disabled=false;}
function renderResult(data){
  $('empty-guide').hidden=true;$('result').hidden=false;$('summary').textContent=data.coaching.summary;
  $('degraded').hidden=data.state!=='engine_only';$('degraded').textContent=data.state==='engine_only'?'Verified engine evidence is available. Gemma coaching was unavailable, so this review uses the deterministic coach.':'';
  const byId=Object.fromEntries([...data.evidence.candidates,...data.evidence.board_facts].map(item=>[item.evidence_id,item]));
  $('claims').replaceChildren(...data.coaching.claims.map(claim=>{const wrap=document.createElement('div'),text=document.createElement('p');wrap.className='claim';text.textContent=claimText(claim,data.evidence);wrap.append(text);if(claim.evidence_ids.length){const button=document.createElement('button'),detail=document.createElement('div');button.type='button';button.textContent='Show cited evidence';button.setAttribute('aria-expanded','false');detail.className='claim-evidence';detail.hidden=true;detail.textContent=claim.evidence_ids.map(id=>JSON.stringify(byId[id]||{missing:id},null,2)).join('\n');button.addEventListener('click',()=>{detail.hidden=!detail.hidden;button.setAttribute('aria-expanded',String(!detail.hidden));});wrap.append(button,detail);}return wrap;}));
  $('evidence').replaceChildren(...data.evidence.candidates.map(item=>{const div=document.createElement('div');div.className='candidate';div.textContent=`${item.rank}. ${item.move_san} · ${item.score_cp!==null?(item.score_cp/100).toFixed(2):'mate '+item.mate_in} · ${item.pv_uci.slice(0,8).join(' ')}`;return div;}));
}
function claimText(claim,evidence){const candidates=Object.fromEntries(evidence.candidates.map(item=>[item.evidence_id,item]));if(claim.kind==='move')return`Recommended move: ${candidates[claim.candidate_id].move_san}.`;if(claim.kind==='score'){const item=candidates[claim.candidate_id];return item.mate_in!==null?`Mate evaluation: ${item.mate_in}.`:`Evaluation from the side to move: ${(item.score_cp/100).toFixed(2)} pawns.`;}if(claim.kind==='line')return`Line to calculate: ${candidates[claim.candidate_id].pv_uci.slice(claim.start_ply,claim.end_ply).join(' ')}.`;if(claim.kind==='comparison'){const best=candidates[claim.better_candidate_id],considered=candidates[claim.considered_candidate_id];if(best.evidence_id===considered.evidence_id)return`${considered.move_san} matches the engine's first choice.`;if(best.score_cp!==null&&considered.score_cp!==null)return`${best.move_san} evaluates ${((best.score_cp-considered.score_cp)/100).toFixed(2)} pawns better than ${considered.move_san}.`;return`The engine prefers ${best.move_san} to ${considered.move_san} by mate outcome.`;}return claim.template_id==='compare_candidate_moves'?'Compare the forcing replies to both moves.':'Calculate checks, captures, and threats first.';}

$('flip').addEventListener('click',()=>{state.flipped=!state.flipped;renderBoard();persistSession();});
$('load-example').addEventListener('click',()=>resetSession(EXAMPLE_FEN));
$('new-game').addEventListener('click',()=>resetSession());
$('undo').addEventListener('click',()=>{const previous=state.history.pop();if(!previous)return;state.exhibitionRunning=false;state.exhibitionToken++;cancelReview();$('exhibition').textContent='Start exhibition';$('fen').value=previous.fen;parseFen(previous.fen);state.moveRows=previous.rows;state.lastMove=[];state.legalTargets=[];$('undo').disabled=state.history.length===0;$('game-outcome').hidden=true;renderMoveList();renderBoard();setStatus(`${state.turn==='w'?'White':'Black'} to move.`);persistSession();});
$('fen').addEventListener('change',()=>{try{resetSession($('fen').value.trim());}catch(error){setError(error.message);}});
$('analyze').addEventListener('click',explainPosition);
$('cancel').addEventListener('click',()=>{cancelReview();setReviewStatus('Review cancelled.');});
$('session-mode').addEventListener('change',()=>{const exhibition=$('session-mode').value==='exhibition';state.exhibitionRunning=false;state.exhibitionToken++;cancelReview();$('exhibition').hidden=!exhibition;$('exhibition').textContent='Start exhibition';$('game-kicker').textContent=exhibition?'Stockfish plays both sides. Every move is reviewed before the next.':'You move, Stockfish replies, and your move is reviewed automatically.';$('board-help').textContent=exhibition?'Start the exhibition to advance one reviewed engine move at a time.':'Select one of your pieces, then its destination. Promotions become queens.';renderBoard();setStatus(exhibition?'Engine exhibition ready.':'Select a piece to move.');persistSession();});
$('exhibition').addEventListener('click',()=>{if(state.exhibitionRunning){state.exhibitionRunning=false;state.exhibitionToken++;cancelReview();$('exhibition').textContent='Start exhibition';setStatus('Engine exhibition paused.');renderBoard();return;}state.exhibitionRunning=true;const token=++state.exhibitionToken;$('exhibition').textContent='Pause exhibition';renderBoard();void runExhibition(token);});
$('difficulty').addEventListener('change',persistSession);$('rating').addEventListener('change',persistSession);

async function loadHealth(){try{const response=await fetch('/api/v1/health'),health=await response.json();$('coach-mode').textContent=health.model_profile==='full'?'Gemma 4 coaching':'Deterministic coaching';}catch{$('coach-mode').textContent='Coach unavailable';}}
initializeBoard();parseFen($('fen').value);const restored=restoreSession();renderMoveList();renderBoard();if(restored)setStatus('Session restored. Continue from the current position.');void loadHealth();
