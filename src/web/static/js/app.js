// Global variables
let isLoading = false;
let gameState = null;
let selectedSquare = null;
let selectedSquareIndex = null;
let gameMode = 'analysis'; // 'analysis' or 'play'
let stockfishMatch = null;
let matchActive = false;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
  initializeChessBoard();
  loadExamples();
  setupEventListeners();
  loadGameState();

  // Welcome
  showMessage('🎮 **ChessGemma Ready!**\n\nClick squares to analyze positions or toggle Play Mode to start a game!', 'success');
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

function setupEventListeners() {
  // Reserved for future interactions
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

  try {
    const expertEl = document.getElementById('expertSelect');
    const expert = expertEl ? expertEl.value : 'auto';
    const response = await fetch('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, context: '', expert })
    });
    const data = await response.json();
    loadingDiv.remove();
    let messageClass = 'assistant';
    let confidenceClass = 'confidence-low';
    if (data.confidence > 0.7) confidenceClass = 'confidence-high';
    else if (data.confidence > 0.4) confidenceClass = 'confidence-medium';
    const confidenceText = data.confidence ? `<div class="confidence-badge ${confidenceClass}">Confidence: ${(data.confidence * 100).toFixed(1)}%</div>` : '';
    addMessage(`
      <div class="d-flex align-items-center mb-2">
        <i class="fas fa-robot me-2"></i>
        <strong>ChessGemma</strong>
      </div>
      <p>${data.response || data.error || 'No response received'}</p>
      ${confidenceText}
    `, messageClass);
  } catch (error) {
    console.error('Error:', error);
    loadingDiv.remove();
    addMessage('Sorry, I encountered an error while processing your question. Please try again.', 'error');
  } finally {
    isLoading = false;
  }
}

function addMessage(content, type = 'assistant') {
  const messagesDiv = document.getElementById('chatMessages');
  const messageDiv = document.createElement('div');
  messageDiv.className = `message ${type}`;
  messageDiv.innerHTML = content;
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
    // Update model loaded badge
    try {
      const infoResp = await fetch('/api/model_info');
      const info = await infoResp.json();
      const header = document.querySelector('#chatPanel .message');
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
    const expertEl = document.getElementById('expertSelect');
    const expert = expertEl ? expertEl.value : 'tutor';
    const response = await fetch('/api/game/ai_move', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    , body: JSON.stringify({ expert })
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
    sq.classList.remove('selected', 'legal-move');
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
    } else {
      toggleButton.classList.add('analysis-mode-active');
      toggleButton.innerHTML = '<i class="fas fa-search me-1"></i>Enter Play Mode';
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


