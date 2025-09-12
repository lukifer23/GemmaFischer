# ChessGemma API Reference

## Overview

ChessGemma provides a REST API for chess analysis, Q&A, and model training. The system features Mixture of Experts (MoE) routing and supports UCI-compatible chess engine functionality.

**Base URL**: `http://localhost:5000`

## Web Interface API

### Chess Q&A Endpoints

#### POST `/api/ask`
Main chess question answering endpoint with MoE routing.

**Parameters:**
- `question` (string): Chess-related question
- `context` (string, optional): Additional context (FEN, position description)
- `expert` (string): Expert selection - `"auto"`, `"uci"`, `"tutor"`, `"director"`

**Response:**
```json
{
  "response": "Analysis and answer...",
  "confidence": 0.85,
  "model_loaded": true,
  "mode": "tutor",
  "moe_used": true,
  "primary_expert": "tutor",
  "ensemble_mode": false,
  "routing_reasoning": "Query contains educational elements"
}
```

**Examples:**
```bash
# Auto-routing (recommended)
curl -X POST http://localhost:5000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What's the best move here?", "context": "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "expert": "auto"}'

# Manual expert selection
curl -X POST http://localhost:5000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "e2e4", "expert": "uci"}'
```

#### GET `/api/model_info`
Get system status and MoE information.


### POST `/api/ask`
Main chess question answering endpoint with MoE routing.

**Parameters:**
- `question` (string): Chess-related question
- `expert` (string): Expert selection - `"auto"\`, `"uci"\`, `"tutor"\`, `"director"\`

**Response:**
```json
{
  "response": "Analysis and answer...\",
  "moe_used": true,
  "primary_expert": "tutor\",
  "confidence": 0.85
}
```

### POST `/api/game/ai_move`
Get AI's recommended chess move.

**Parameters:**
- `expert` (string): Expert selection (default: auto)

**Response:**
```json
{
  "success": true,
  "move": "e2e4\",
  "fen": "updated_fen\",
  "ai_response": "Move explanation...\"
}
```

### GET `/api/model_info`
Get system status and available experts.

**Response:**
```json
{
  "moe_enabled": true,
  "experts_available": ["uci", "tutor", "director"],
  "model_loaded": true
}
```

## Training API

### Expert Training
The system supports training specialized expert models:

```bash
# UCI Expert training
python -m src.training.train_lora_poc --expert uci --config auto --max_steps_override 1600 --disable_eval

# Tutor Expert training  
python -m src.training.train_lora_poc --expert tutor --config auto --max_steps_override 1000 --disable_eval

# Director Expert training
python -m src.training.train_lora_poc --expert director --config auto --max_steps_override 1000 --disable_eval
```

## Web Interface

Launch the web interface for interactive chess analysis:

```bash
python -m src.web.run_web_app
# Visit: http://localhost:5000
```

### Features
- **Interactive Chess Board**: Click-to-move interface
- **Real-time Q&A**: Ask questions about chess positions
- **Expert Selection**: Auto/UCI/Tutor/Director modes
- **Live MoE Feedback**: Shows which expert is being used
- **Training Controls**: GUI for model training
