# Project Plan: GemmaFischer — Chess LLM Engine + Tutor

## Summary

**GemmaFischer** is a lightweight language model (based on Gemma 3B or 270M) designed to function as both:

- A **chess engine**, capable of interfacing with UCI backends, evaluating positions, and suggesting moves
- A **chess tutor/analyst**, able to explain concepts, annotate games, and provide historical insight through chain-of-thought (CoT) reasoning

---

## Project Checklist

### Core LLM Development

- [ ] Start from *ChessGemma* (fine-tuned on PGNs) → evolve as `GemmaFischer`
- [ ] Add **Chain-of-Thought (CoT)** reasoning capability
  - Format: `[Position] → Think step-by-step → Evaluate threats → Suggest best move`
- [ ] Train dual-purpose outputs:
  - `Tutor:` Mode with rich explanations
  - `Engine:` Mode with minimal fast UCI-style outputs
- [ ] Expand training data:
  - Annotated PGNs (Lichess Elite, King's Database)
  - Historical games: Fischer, Kasparov, AlphaZero
  - Theory books (openings, endgames, tactics)
- [ ] Format prompts to support style, reasoning, and role conditioning

---

### Training & Evaluation Objectives

- [ ] **Move Prediction Accuracy**
  - Compare LLM move predictions to Stockfish at depth 12
- [ ] **Tactical Awareness**
  - Train on Lichess tactics DB + custom puzzles
- [ ] **Position Evaluation**
  - Fine-tune on FEN → Eval scores (from SF or Leela)
- [ ] **Style Conditioning (optional)**
  - Prompt tokens: `Style[Fischer]`, `Style[Aggressive]`, etc.

---

### UCI Backend Interfacing

- [ ] Implement adapter for UCI protocol compatibility
  - Input: UCI-style moves → Output: bestmove
- [ ] Format board state for prompt injection
  - SAN/UCI → FEN → parsed prompt
- [ ] Optional: fallback mode with Stockfish API or `python-chess`
- [ ] Build bridge module:
  - `LLM_UCI.py` → handles I/O to command line UCI loop

---

### Optional: Embedding Search Functions

- [ ] Generate embeddings for:
  - Openings (ECO code → vector)
  - Endgame patterns
  - Historical game similarity
- [ ] Use vector search (e.g., FAISS, qdrant) for:
  - Retrieval-Augmented Move Suggestions
  - Similar position explanations

---

### Visual Understanding Layer (Optional)

- [ ] Train vision model:
  - Board image → FEN parsing
  - Use CNN or ViT encoder + FEN tokenizer
- [ ] Combine with LLM:
  ```plaintext
  <image:board.png>
  [FEN decoded internally]
  LLM: Analyzing current position...
  ```
- [ ] Use synthetic datasets:
  - Render images from PGNs to create paired training data

---

## Integrated Chess Knowledge Goals

- [ ] Encode **chess theory**
  - Openings, endgames, tactics, positional themes
- [ ] Encode **chess history**
  - Famous matches, historical strategies, key players
- [ ] Style-based advice
  - Match historical player playstyles
- [ ] Didactic tone in Tutor mode:
  - "Let's think this through: your knight is controlling d5 and f6..."

---

## Techniques & Tools Needed

| Task                     | Tools / Techniques                                  |
|--------------------------|-----------------------------------------------------|
| LLM Fine-tuning          | QLoRA / LoRA (270M or 2B) in HuggingFace or GGUF    |
| Dataset Creation         | PGN parsers, `python-chess`, SF/Lc0 for evals       |
| CoT Prompting            | Instruction tuning / FLAN-style prompts             |
| UCI Interfacing          | `python-chess`, custom subprocess or socket bridge  |
| Embeddings               | SentenceTransformers, or LLM internal pooler        |
| Vision Module            | CNN → encoder → decoder (ViT or CLIP-style)         |
| Evaluation               | Elo tests, SF comparisons, tutor accuracy scoring   |

---

## Evaluation Plan

- [ ] **UCI Output Validity**
  - Move legality, SAN/UCI parsing, legal follow-ups
- [ ] **Prompt→Move Benchmarking**
  - Compare to Stockfish or LeelaZero on test sets
- [ ] **Explanation Accuracy**
  - Chain-of-thought scoring vs annotated data
- [ ] **Style Emulation Tests**
  - "Play like [Fischer]" → style match against real Fischer game

---

## Future Ideas

- Multi-turn memory for full-game analysis
- Voice assistant mode ("Coach Fischer, what's next?")
- Lichess or Chess.com API interface
- Export annotated PGNs or printable lessons
- Detect and critique user style: "Your midgame is often rushed…"

---

## Suggested File Structure

```plaintext
GemmaFischer/
├── src/
│   ├── training/           # Training scripts and configs
│   │   ├── train.py
│   │   ├── train_lora.py
│   │   └── configs/
│   ├── inference/          # Inference and UCI bridge
│   │   ├── inference.py
│   │   ├── chess_engine.py
│   │   └── uci_bridge.py
│   ├── evaluation/         # Evaluation and testing
│   │   ├── chess_evaluation.py
│   │   ├── position_accuracy_tests.py
│   │   └── style_emulation_eval.py
│   └── web/               # Web interface
│       └── app.py
├── data/
│   ├── raw/               # Raw datasets
│   ├── processed/         # Processed data
│   ├── datasets/          # Training datasets
│   ├── annotated_games.pgn
│   ├── fen_evals.csv
│   └── lichess_tactics.json
├── models/                # Model storage
│   └── gemmafischer-270M-lora/
├── checkpoints/           # Training checkpoints
├── docs/                  # Documentation
│   ├── PROJECT_PLAN.md    # This file
│   ├── ARCHITECTURE.md
│   └── TRAINING_GUIDE.md
├── prompts/               # Prompt templates
│   ├── tutor_mode.txt
│   └── engine_mode.txt
├── tests/                 # Unit tests
└── README.md
```

---

## End Goal

A dual-purpose, lightweight chess LLM that can:
- Play chess via UCI
- Explain its reasoning and evaluate positions
- Adapt its style and tutor users
- Eventually understand visual board input and retrieval-enhanced memory
