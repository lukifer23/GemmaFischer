# GemmaFischer API Reference (Historical v2)

> **Historical and unverified.** This describes the quarantined Flask/MoE application, not the supported vNext API. See `docs/evidence-contract.md` and generated `docs/openapi.json`.

## Overview

GemmaFischer provides a REST API for chess analysis, Q&A, and model training. The system features a hybrid architecture combining LeelaChess Zero (LC0) for precise move generation with Gemma-3 LLM for strategic guidance and educational explanations, with intelligent MoE routing.

**Base URL**: `http://localhost:5000`

## Web Interface API

### Chess Q&A Endpoints

#### POST `/api/ask`
Main chess question answering endpoint with hybrid MoE routing.

**Parameters:**
- `question` (string): Chess-related question
- `context` (string, optional): Additional context (FEN, position description)
- `expert` (string): Expert selection - `"auto"` (intelligent routing), `"uci"` (LC0 engine), `"tutor"` (educational analysis), `"director"` (strategic guidance)

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

#### POST `/api/ask_parallel`
Get responses from all three experts (UCI, Tutor, Director) simultaneously for comprehensive analysis.

**Parameters:**
- `question` (string): Chess-related question
- `context` (string, optional): Additional context (FEN, position description)
- `experts` (array, optional): List of experts to query (default: `["uci", "tutor", "director"]`)

**Response:**
```json
{
  "question": "What's the best move for white?",
  "context": "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
  "experts": ["uci", "tutor", "director"],
  "total_time": 3.45,
  "results": {
    "uci": {
      "response": "e4d5",
      "confidence": 0.92,
      "generation_time": 1.2,
      "mode": "engine",
      "cached": false
    },
    "tutor": {
      "response": "The best move is exd5, capturing the knight on c6. This develops your bishop while attacking the opponent's knight...",
      "confidence": 0.88,
      "generation_time": 2.1,
      "mode": "tutor",
      "cached": false
    },
    "director": {
      "response": "This is an open game where White has a strong center. The key tactical opportunity is the discovered attack after exd5...",
      "confidence": 0.85,
      "generation_time": 1.8,
      "mode": "director",
      "cached": false
    }
  }
}
```

**Examples:**
```bash
# Get all expert responses simultaneously
curl -X POST http://localhost:5000/api/ask_parallel \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the best move for white?",
    "context": "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
  }'

# Query specific experts only
curl -X POST http://localhost:5000/api/ask_parallel \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Explain castling",
    "experts": ["tutor", "director"]
  }'
```

**Benefits:**
- **Comprehensive Analysis**: See tactical, educational, and strategic perspectives simultaneously
- **Cross-Validation**: Compare expert responses for consistency
- **Educational Value**: Learn from multiple teaching approaches
- **Debugging**: Test all experts on the same query for development

#### GET `/api/model_info`
Get system status and MoE information.

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
./run_hybrid_webapp.sh
# Visit: http://localhost:5000
```

### Features
- **Interactive Chess Board**: Click-to-move interface
- **Real-time Q&A**: Ask questions about chess positions
- **Expert Selection**: Auto/UCI/Tutor/Director modes
- **Live MoE Feedback**: Shows which expert is being used
- **Training Controls**: GUI for model training
