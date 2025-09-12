# ChessGemma Dataset Guide

## Overview

ChessGemma uses standardized training datasets for fine-tuning specialized expert models. All datasets are validated for quality and formatted consistently.

**Current Status**: 105K+ validated training samples across three expert domains.

## Available Datasets

### Primary Training Datasets

| Dataset | Location | Size | Purpose |
|---------|----------|------|---------|
| UCI Expert | `data/standardized/standardized_uci_expert.jsonl` | 50,000 samples | Chess move generation |
| Tutor Expert | `data/standardized/standardized_tutor_expert.jsonl` | 50,000 samples | Chess explanations |
| Director Expert | `data/standardized/standardized_director_expert.jsonl` | 5,133 samples | Q&A reasoning |

### Dataset Quality
- **Validation**: 100% move legality verification with Stockfish
- **Format**: Standardized JSONL schema
- **Quality**: All samples validated for correctness
- **Metadata**: Includes FEN positions, ratings, and quality scores


## Data Format

### UCI Expert Format
```json
{
  "task": "engine_uci",
  "prompt": "FEN: [position]
Generate the best move in UCI format only:",
  "response": "e2e4",
  "meta": {
    "fen": "[position]",
    "rating": 1500,
    "quality_score": 0.8
  }
}
```

### Tutor Expert Format
```json
{
  "task": "tutor_explain",
  "prompt": "Analyze this position step by step...",
  "response": "Tactical analysis... Best move: e2e4",
  "meta": {
    "fen": "[position]",
    "source": "lichess_puzzles"
  }
}
```

### Director Expert Format
```json
{
  "task": "director_qa",
  "prompt": "Strategic analysis question...",
  "response": "Detailed strategic reasoning...",
  "meta": {
    "topic": "tactics",
    "complexity": "intermediate"
  }
}
```
