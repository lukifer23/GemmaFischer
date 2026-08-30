# GemmaFischer Dataset Guide (Historical v2)

> **Historical and unverified.** All referenced datasets are quarantined pending provenance, semantic, and leakage review.

## Overview

GemmaFischer uses standardized training datasets for fine-tuning specialized expert models. All datasets are validated for quality and formatted consistently.

**Current Status**: 150K validated, placeholder-free training samples across three expert domains.

## Available Datasets

### Primary Training Datasets

| Dataset | Location | Size | Purpose |
|---------|----------|------|---------|
| UCI Expert | `data/standardized/standardized_uci_expert_v2.jsonl` | 50,000 samples | Chess move generation (depth-14 Stockfish labels) |
| Tutor Expert | `data/standardized/standardized_tutor_expert_v2.jsonl` | 49,999 samples | Chess explanations |
| Director Expert | `data/standardized/standardized_director_expert_v3.jsonl` | 49,999 samples | Strategic Q&A reasoning |

### Dataset Quality
- **Validation**: 100% move legality verification with Stockfish
- **Stockfish**: Depth-14 best move plus depth-6 "top-3" alternatives captured in metadata
- **Format**: Standardized JSONL schema
- **Quality**: All samples validated for correctness
- **Metadata**: Includes FEN positions, ratings, and quality scores

### Validation & Evaluation Datasets
| Dataset | Location | Purpose |
|---------|----------|---------|
| Router Eval Suite | `data/validation/eval_suite.jsonl` | Mixed intent queries used for MoE regression checks |
| UCI Eval Positions | `data/validation/eval_mixed_positions_200.jsonl` | 200 Stockfish-verified move targets |
| Tutor Eval Puzzles | `data/validation/tutor_comprehensive_validation.json` | Step-by-step analysis prompts with reference answers |
| Director Eval QA | `data/validation/director_comprehensive_validation.json` | Strategic/rules queries with curated expectations |

Regenerate the evaluation artifacts after dataset updates:

```bash
# Generate and validate datasets (consolidated)
python scripts/validate_and_repair_datasets.py --generate --datasets-dir data/standardized --repair
```


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
    "quality_score": 0.8,
    "stockfish_depth": 14,
    "stockfish_time_limit_ms": 1500,
    "top_moves_depth6": ["e2e4", "d2d4", "g1f3"],
    "stockfish_analysis": {
      "best_move": {
        "uci": "e2e4",
        "score_cp": 34,
        "depth": 14,
        "seldepth": 22,
        "pv": ["e2e4", "c7c5", "g1f3"]
      },
      "top_moves": {
        "depth": 6,
        "time_limit_ms": 500,
        "entries": [
          {"uci": "e2e4", "score_cp": 34},
          {"uci": "d2d4", "score_cp": 28},
          {"uci": "g1f3", "score_cp": 21}
        ]
      }
    }
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
