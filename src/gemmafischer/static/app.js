const PIECES={p:'♟',r:'♜',n:'♞',b:'♝',q:'♛',k:'♚',P:'♙',R:'♖',N:'♘',B:'♗',Q:'♕',K:'♔'};
const NAMES={p:'black pawn',r:'black rook',n:'black knight',b:'black bishop',q:'black queen',k:'black king',P:'white pawn',R:'white rook',N:'white knight',B:'white bishop',Q:'white queen',K:'white king'};
const START_FEN='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
const EXAMPLE_FEN='r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3';
const STORAGE_KEY='gemmafischer.session.v2';
const $=id=>document.getElementById(id);
const wait=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const state={session:null,squares:{},turn:'w',flipped:false,selected:null,legalTargets:[],lastMove:[],focusSquare:'a8',busy:false,reviewToken:0,analysisId:null,exhibitionToken:0,exhibitionRunning:false};
const squareNodes=new Map();

async function api(path,options={}){
  const response=await fetch(path,{...options,headers:{'Content-Type':'application/json',...(options.headers||{})}});
  const data=response.status===204?null:await response.json();
  if(!response.ok)throw new Error(data?.error?.message||`Request failed (${response.status})`);
  return data;
}
function savePreferences(){try{localStorage.setItem(STORAGE_KEY,JSON.stringify({schema:2,sessionId:state.session?.session_id,flipped:state.flipped,mode:$('session-mode').value,difficulty:$('difficulty').value,rating:$('rating').value}));}catch{/* Storage can be unavailable. */}}
function loadPreferences(){try{const saved=JSON.parse(localStorage.getItem(STORAGE_KEY)||'null');if(!saved||saved.schema!==2)return{};state.flipped=Boolean(saved.flipped);if(['player','exhibition'].includes(saved.mode))$('session-mode').value=saved.mode;if(['casual','club','strong'].includes(saved.difficulty))$('difficulty').value=saved.difficulty;if(['1000-1199','1200-1399','1400-1599','1600-1800'].includes(saved.rating))$('rating').value=saved.rating;return saved;}catch{localStorage.removeItem(STORAGE_KEY);return{};}}

function parseFen(fen){
  const fields=fen.trim().split(/\s+/);if(fields.length!==6)throw new Error(`FEN requires six fields; received ${fields.length}.`);
  const ranks=fields[0].split('/');if(ranks.length!==8)throw new Error('FEN board requires eight ranks.');const squares={};
  ranks.forEach((rank,row)=>{let file=0;for(const char of rank){if(/\d/.test(char))file+=Number(char);else{if(!PIECES[char]||file>7)throw new Error('FEN contains invalid piece placement.');squares['abcdefgh'[file]+(8-row)]=char;file+=1;}}if(file!==8)throw new Error('Every FEN rank must describe eight squares.');});
  if(!['w','b'].includes(fields[1]))throw new Error('FEN side to move must be w or b.');state.squares=squares;state.turn=fields[1];
}
function initializeBoard(){for(const rank of [8,7,6,5,4,3,2,1])for(const file of 'abcdefgh'){const square=file+rank,button=document.createElement('button'),piece=document.createElement('span'),coordinate=document.createElement('span');button.type='button';button.dataset.square=square;button.setAttribute('role','gridcell');piece.className='piece';piece.setAttribute('aria-hidden','true');coordinate.className='coordinate';coordinate.setAttribute('aria-hidden','true');coordinate.textContent=square;button.append(piece,coordinate);button.addEventListener('click',()=>selectSquare(square));button.addEventListener('focus',()=>{state.focusSquare=square;});button.addEventListener('keydown',boardKeydown);squareNodes.set(square,button);}}
function boardOrder(){const files=state.flipped?[...'hgfedcba']:[...'abcdefgh'],ranks=state.flipped?[1,2,3,4,5,6,7,8]:[8,7,6,5,4,3,2,1];return ranks.flatMap(rank=>files.map(file=>file+rank));}
function renderBoard(){
  const board=$('board'),order=boardOrder();board.classList.toggle('busy',state.busy);board.classList.toggle('automated',state.exhibitionRunning);board.setAttribute('aria-busy',String(state.busy));
  order.forEach((square,index)=>{const button=squareNodes.get(square),piece=state.squares[square],legal=state.legalTargets.includes(square),row=Math.floor(index/8),column=index%8;button.className=`square ${(row+column)%2?'dark':'light'}`;if(state.selected===square)button.classList.add('selected');if(state.lastMove.includes(square))button.classList.add('last-move');if(legal)button.classList.add(piece?'legal-capture':'legal-target');button.querySelector('.piece').textContent=piece?PIECES[piece]:'';button.querySelector('.coordinate').textContent=row===7?square[0]:(column===0?square[1]:'');button.setAttribute('aria-label',`${piece?NAMES[piece]+' on ':'Empty '}${square}${legal?', legal destination':''}`);button.tabIndex=square===state.focusSquare?0:-1;});
  if(board.children.length!==64||[...board.children].some((node,index)=>node!==squareNodes.get(order[index])))board.replaceChildren(...order.map(square=>squareNodes.get(square)));
}
function boardKeydown(event){const deltas={ArrowLeft:[-1,0],ArrowRight:[1,0],ArrowUp:[0,-1],ArrowDown:[0,1]};if(!(event.key in deltas))return;event.preventDefault();const order=boardOrder(),index=order.indexOf(event.currentTarget.dataset.square),row=Math.floor(index/8),column=index%8,[dx,dy]=deltas[event.key],nextRow=Math.max(0,Math.min(7,row+dy)),nextColumn=Math.max(0,Math.min(7,column+dx)),next=order[nextRow*8+nextColumn];state.focusSquare=next;renderBoard();squareNodes.get(next).focus();}
function pieceMatchesPlayer(piece){if(!piece||!state.session||state.session.mode!=='player')return false;const white=piece===piece.toUpperCase();return state.session.player_color===(white?'white':'black')&&state.session.turn===state.session.player_color;}

async function selectSquare(square){
  if(state.busy||state.exhibitionRunning||!state.session)return;const piece=state.squares[square];
  if(!state.selected){if(!pieceMatchesPlayer(piece)){setError(state.session.turn===state.session.player_color?'Select one of your pieces.':'Stockfish is to move.');return;}await selectSource(square);return;}
  if(pieceMatchesPlayer(piece)){await selectSource(square);return;}if(!state.legalTargets.includes(square)){setError(`${square} is not a legal destination for ${state.selected}.`);return;}
  const move=state.selected+square;state.selected=null;state.legalTargets=[];renderBoard();await playPlayerTurn(move);
}
async function selectSource(square){setError();state.selected=square;state.legalTargets=[];renderBoard();setStatus(`${NAMES[state.squares[square]]} selected on ${square}. Loading legal moves…`);try{const data=await api(`/api/v1/sessions/${state.session.session_id}/legal-moves?from_square=${square}`);if(state.selected!==square)return;state.legalTargets=data.destinations;setStatus(data.destinations.length?`${data.destinations.length} legal destination${data.destinations.length===1?'':'s'} highlighted.`:`${square} has no legal moves.`);}catch(error){if(state.selected===square)state.selected=null;setError(`Legal moves could not be loaded: ${error.message}`);}renderBoard();}

async function animatePly(ply){
  const sourceNode=squareNodes.get(ply.move_uci.slice(0,2)),destinationNode=squareNodes.get(ply.move_uci.slice(2,4)),source=sourceNode.getBoundingClientRect(),destination=destinationNode.getBoundingClientRect(),glyph=sourceNode.querySelector('.piece').textContent;state.lastMove=[ply.move_uci.slice(0,2),ply.move_uci.slice(2,4)];
  if(glyph&&!matchMedia('(prefers-reduced-motion: reduce)').matches){const ghost=document.createElement('span');ghost.className='piece-ghost';ghost.textContent=glyph;Object.assign(ghost.style,{left:`${source.left}px`,top:`${source.top}px`,width:`${source.width}px`,height:`${source.height}px`});document.body.append(ghost);sourceNode.querySelector('.piece').style.visibility='hidden';const animation=ghost.animate([{transform:'translate(0,0)'},{transform:`translate(${destination.left-source.left}px,${destination.top-source.top}px)`}],{duration:180,easing:'cubic-bezier(.2,.8,.2,1)'});try{await animation.finished;}catch{/* A newer render superseded the move. */}ghost.remove();sourceNode.querySelector('.piece').style.visibility='';}
  parseFen(ply.fen_after);renderBoard();
}
async function sessionCommand(action,moveUci=null){return api(`/api/v1/sessions/${state.session.session_id}/commands`,{method:'POST',body:JSON.stringify({expected_revision:state.session.revision,action,move_uci:moveUci})});}
async function playPlayerTurn(move){
  setError();state.busy=true;renderBoard();setStatus('Move submitted…');
  try{const afterPlayer=await sessionCommand('player_move',move),playerPly=afterPlayer.plies.at(-1);state.session=afterPlayer;await animatePly(playerPly);renderSession();setStatus(`You played ${playerPly.move_san}. Stockfish is thinking…`);const afterEngine=afterPlayer.status==='complete'?afterPlayer:await sessionCommand('engine_move');if(afterEngine.revision!==afterPlayer.revision){const enginePly=afterEngine.plies.at(-1);state.session=afterEngine;await animatePly(enginePly);}renderSession();savePreferences();setStatus(state.session.outcome?`Game over: ${state.session.outcome}.`:`${state.session.turn===state.session.player_color?'Your':'Stockfish'} turn.`);const reviewedPlayer=afterEngine.plies.find(ply=>ply.ply===playerPly.ply);if(reviewedPlayer?.analysis_id)void pollAnalysis(reviewedPlayer.analysis_id,`Review of ${reviewedPlayer.move_san}`);
  }catch(error){setError(`The move could not be played: ${error.message}`);await refreshSession();}finally{state.busy=false;renderBoard();}
}
async function runExhibition(token){
  while(state.exhibitionRunning&&token===state.exhibitionToken&&state.session.status!=='complete'){state.busy=true;renderBoard();setStatus(`${state.session.turn==='white'?'White':'Black'} engine is thinking…`);try{const next=await sessionCommand('engine_move'),ply=next.plies.at(-1);state.session=next;await animatePly(ply);renderSession();savePreferences();state.busy=false;renderBoard();if(ply.analysis_id)await pollAnalysis(ply.analysis_id,`Review of ${ply.move_san}`);if(next.status==='complete')break;await wait(350);}catch(error){setError(`Engine exhibition stopped: ${error.message}`);break;}finally{state.busy=false;renderBoard();}}
  if(token===state.exhibitionToken){state.exhibitionRunning=false;$('exhibition').textContent=state.session.status==='complete'?'Game complete':'Resume exhibition';renderBoard();}
}

async function pollAnalysis(analysisId,label){
  cancelReview();const token=++state.reviewToken;state.analysisId=analysisId;setReviewStatus(`${label}…`);$('cancel').hidden=false;
  try{while(token===state.reviewToken){const data=await api(`/api/v1/analyses/${analysisId}`),labels={queued:'Review queued.',validating:'Checking the position.',engine_running:'Stockfish is calculating.',comparison_running:'Comparing the move.',model_running:'Gemma is ordering the lesson.',complete:'Review complete.',engine_only:'Engine review complete; Gemma unavailable.',cancelled:'Review cancelled.',failed:'Review failed.'};setReviewStatus(labels[data.state]||data.state);if(['complete','engine_only','failed','cancelled'].includes(data.state)){if(data.state==='failed')setError(data.error?.message||'Review failed.');else if(data.evidence&&data.coaching)renderResult(data);return data;}await wait(300);}}
  catch(error){setReviewStatus(`Review interrupted: ${error.message}`);}finally{if(token===state.reviewToken){$('cancel').hidden=true;state.analysisId=null;}}return null;
}
async function explainPosition(){try{const created=await api('/api/v1/analyses',{method:'POST',body:JSON.stringify({mode:'position',fen:state.session.fen,rating_bucket:$('rating').value,considered_move_uci:null})});await pollAnalysis(created.analysis_id,'Position explanation');}catch(error){setError(`The explanation could not start: ${error.message}`);}}
function cancelReview(send=true){state.reviewToken+=1;if(send&&state.analysisId)void fetch(`/api/v1/analyses/${state.analysisId}`,{method:'DELETE'});state.analysisId=null;$('cancel').hidden=true;}

function renderResult(data){
  $('empty-guide').hidden=true;$('result').hidden=false;$('summary').textContent=data.coaching.summary;$('degraded').hidden=data.state!=='engine_only';$('degraded').textContent=data.state==='engine_only'?'The verified deterministic lesson is shown because Gemma was unavailable.':'';
  const candidates=data.evidence.candidate_set?.candidates||[],comparison=data.evidence.move_comparison,byId=Object.fromEntries([...candidates,...data.evidence.board_facts,...(data.evidence.concepts||[]),...(comparison?[comparison]:[])].map(item=>[item.evidence_id,item]));
  const lesson=(data.coaching.lesson_plan?.steps||[]).map(step=>{const wrap=document.createElement('div'),text=document.createElement('p'),detail=document.createElement('div');wrap.className='claim lesson-step';text.textContent=step.text;detail.className='claim-evidence';detail.textContent=evidenceText(byId[step.concept_id]);wrap.append(text,detail);return wrap;});
  const claims=data.coaching.claims.map(claim=>{const wrap=document.createElement('div'),text=document.createElement('p');wrap.className='claim';text.textContent=claimText(claim,data.evidence);wrap.append(text);if(claim.evidence_ids.length){const button=document.createElement('button'),detail=document.createElement('div');button.type='button';button.textContent='Show cited evidence';button.setAttribute('aria-expanded','false');detail.className='claim-evidence';detail.hidden=true;detail.textContent=claim.evidence_ids.map(id=>evidenceText(byId[id])).join('\n');button.addEventListener('click',()=>{detail.hidden=!detail.hidden;button.setAttribute('aria-expanded',String(!detail.hidden));});wrap.append(button,detail);}return wrap;});
  $('claims').replaceChildren(...lesson,...claims);
  $('evidence').replaceChildren(...candidates.map(item=>{const div=document.createElement('div'),score=item.score_cp!==null?`${(item.score_cp/100).toFixed(2)} pawns`:`mate ${item.mate_in}`;div.className='candidate';div.textContent=`${item.rank}. ${item.move_san} · ${score} · ${item.pv_uci.slice(0,8).join(' ')}`;return div;}));
}
function evidenceText(item){if(!item)return'Evidence unavailable.';if('move_san'in item)return`${item.move_san} (${item.move_uci}), rank ${item.rank}, ${item.nodes.toLocaleString()} nodes, line ${item.pv_uci.slice(0,8).join(' ')}`;if('outcome'in item)return`Matched ${item.node_budget_each.toLocaleString()}-node searches: ${item.engine_move_uci} vs ${item.considered_move_uci}; ${item.outcome.replaceAll('_',' ')}.`;if('concept'in item)return`${item.concept.replaceAll('_',' ')}: ${item.value}`;return`${item.fact_type.replaceAll('_',' ')}: ${item.value}`;}
function claimText(claim,evidence){const candidates=Object.fromEntries((evidence.candidate_set?.candidates||[]).map(item=>[item.evidence_id,item]));if(claim.kind==='move')return`Recommended move: ${candidates[claim.candidate_id].move_san}.`;if(claim.kind==='score'){const item=candidates[claim.candidate_id];return item.mate_in!==null?`Mate evaluation: ${item.mate_in}.`:`Evaluation from the side to move: ${(item.score_cp/100).toFixed(2)} pawns.`;}if(claim.kind==='line')return`Line to calculate: ${candidates[claim.candidate_id].pv_uci.slice(claim.start_ply,claim.end_ply).join(' ')}.`;if(claim.kind==='comparison'){const item=evidence.move_comparison;if(item.outcome==='equal')return`${item.considered_move_uci} is effectively equal to ${item.engine_move_uci} within ${item.tolerance_cp} centipawns.`;return`The matched-budget comparison favors ${item.outcome==='engine_better'?item.engine_move_uci:item.considered_move_uci}.`;}return claim.template_id==='compare_candidate_moves'?'Compare the forcing replies to both moves.':'Calculate checks, captures, and threats first.';}

function renderSession(){
  if(!state.session)return;$('fen').value=state.session.fen;parseFen(state.session.fen);$('undo').disabled=state.session.plies.length===0||state.busy;$('game-outcome').hidden=!state.session.outcome;$('game-outcome').textContent=state.session.outcome?`Game over · ${state.session.outcome}`:'';
  const rows=[];state.session.plies.forEach(ply=>{const fields=ply.fen_before.split(/\s+/),number=Number(fields[5]),color=fields[1]==='w'?'white':'black';let row=rows.find(item=>item.number===number);if(!row){row={number,white:'',black:''};rows.push(row);}row[color]=ply.move_san;});
  $('move-list').replaceChildren(...rows.map(row=>{const item=document.createElement('li');[String(row.number).padStart(2,'0'),row.white||'…',row.black||'…'].forEach(value=>{const span=document.createElement('span');span.textContent=value;item.append(span);});return item;}));$('game-panel').hidden=rows.length===0;renderBoard();
}
async function createSession(fen=START_FEN){stopExhibition();cancelReview();setError();const exhibition=$('session-mode').value==='exhibition';state.session=await api('/api/v1/sessions',{method:'POST',body:JSON.stringify({mode:exhibition?'exhibition':'player',fen,player_color:exhibition?null:'white',white_difficulty:$('difficulty').value,black_difficulty:$('difficulty').value,rating_bucket:$('rating').value})});state.selected=null;state.legalTargets=[];state.lastMove=[];$('result').hidden=true;$('empty-guide').hidden=false;renderMode();renderSession();setStatus(exhibition?'Engine exhibition ready.':`${state.session.turn==='white'?'White':'Black'} to move. Select a piece.`);savePreferences();}
async function refreshSession(){if(!state.session)return;try{state.session=await api(`/api/v1/sessions/${state.session.session_id}`);renderSession();}catch(error){setError(error.message);}}
function renderMode(){const exhibition=$('session-mode').value==='exhibition';$('exhibition').hidden=!exhibition;$('game-kicker').textContent=exhibition?'Stockfish plays both sides and each ply is reviewed before the next.':'You play, Stockfish replies, and your move is reviewed automatically.';$('board-help').textContent=exhibition?'Start, pause, or resume the engine exhibition here.':'Select one of your pieces, then choose a highlighted destination.';}
function stopExhibition(){state.exhibitionRunning=false;state.exhibitionToken+=1;$('exhibition').textContent='Start exhibition';}
function setError(message=''){$('input-error').textContent=message;$('input-error').hidden=!message;}
function setStatus(message){$('status').textContent=message;}
function setReviewStatus(message=''){$('review-status').textContent=message;$('review-status').hidden=!message;}

$('flip').addEventListener('click',()=>{state.flipped=!state.flipped;renderBoard();savePreferences();});
$('new-game').addEventListener('click',()=>{void createSession();});
$('load-example').addEventListener('click',()=>{void createSession(EXAMPLE_FEN);});
$('fen').addEventListener('change',()=>{void createSession($('fen').value.trim()).catch(error=>setError(error.message));});
$('analyze').addEventListener('click',()=>{void explainPosition();});
$('cancel').addEventListener('click',()=>{cancelReview();setReviewStatus('Review cancelled.');});
$('undo').addEventListener('click',async()=>{stopExhibition();cancelReview();try{state.session=await sessionCommand('undo');state.lastMove=[];renderSession();savePreferences();}catch(error){setError(error.message);}});
$('session-mode').addEventListener('change',()=>{renderMode();void createSession();});
$('difficulty').addEventListener('change',()=>{savePreferences();void createSession(state.session?.initial_fen||START_FEN);});
$('rating').addEventListener('change',savePreferences);
$('exhibition').addEventListener('click',()=>{if(state.exhibitionRunning){stopExhibition();cancelReview();setStatus('Engine exhibition paused.');renderBoard();return;}state.exhibitionRunning=true;const token=++state.exhibitionToken;$('exhibition').textContent='Pause exhibition';renderBoard();void runExhibition(token);});

async function initialize(){initializeBoard();parseFen(START_FEN);renderBoard();const saved=loadPreferences();renderMode();try{if(saved.sessionId)state.session=await api(`/api/v1/sessions/${saved.sessionId}`);}catch{localStorage.removeItem(STORAGE_KEY);}if(!state.session)await createSession(EXAMPLE_FEN);else{$('session-mode').value=state.session.mode;renderMode();renderSession();setStatus('Session restored from the local database.');const reviewed=[...state.session.plies].reverse().find(ply=>ply.analysis_id);if(reviewed)void pollAnalysis(reviewed.analysis_id,`Review of ${reviewed.move_san}`);}try{const capabilities=await api('/api/v1/capabilities');$('coach-mode').textContent=capabilities.model_status==='disabled'?'Deterministic coach':`Gemma 4 · ${capabilities.model_status}`;}catch{$('coach-mode').textContent='Coach unavailable';}}
void initialize().catch(error=>setError(`GemmaFischer could not start: ${error.message}`));
