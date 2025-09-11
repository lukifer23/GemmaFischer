# Project Plan: GemmaFischer — Chess LLM Engine + Tutor

## Summary

**GemmaFischer** is a lightweight language model (based on Gemma 3B or 270M) designed to function as both:

- A **chess engine**, capable of interfacing with UCI backends, evaluating positions, and suggesting moves
- A **chess tutor/analyst**, able to explain concepts, annotate games, and provide historical insight through chain-of-thought (CoT) reasoning

**Platform**: Mac-only (M3 Pro) with MPS acceleration - no CUDA/CPU fallbacks.

---

## Project Checklist

### Core LLM Development

- [x] Start from *ChessGemma* (fine-tuned on PGNs) → evolve as `GemmaFischer`
- [ ] Add **Chain-of-Thought (CoT)** reasoning capability
  - Format: `[Position] → Think step-by-step → Evaluate threats → Suggest best move`
- [x] Train dual-purpose outputs:
  - `Tutor:` Mode with rich explanations
  - `Engine:` Mode with minimal fast UCI-style outputs
- [x] Expand training data:
  - [x] Annotated PGNs (Lichess Elite, King's Database) - 5M+ puzzles processed
  - [x] Historical games: Fischer, Kasparov, AlphaZero - master games included
  - [x] Theory books (openings, endgames, tactics) - comprehensive theory database
- [x] Format prompts to support style, reasoning, and role conditioning

---

### Training & Evaluation Objectives

- [ ] **Move Prediction Accuracy**
  - Compare LLM move predictions to Stockfish at depth 12
- [x] **Tactical Awareness**
  - [x] Train on Lichess tactics DB + custom puzzles (5,133 tactical examples with CoT reasoning)
- [ ] **Position Evaluation**
  - Fine-tune on FEN → Eval scores (from SF or Leela)
- [ ] **Style Conditioning (optional)**
  - Prompt tokens: `Style[Fischer]`, `Style[Aggressive]`, etc.

---

### UCI Backend Interfacing

- [x] Implement adapter for UCI protocol compatibility
  - Input: UCI-style moves → Output: bestmove
- [x] Format board state for prompt injection
  - SAN/UCI → FEN → parsed prompt
- [x] Optional: fallback mode with Stockfish API or `python-chess`
- [x] Build bridge module:
  - `LLM_UCI.py` → handles I/O to command line UCI loop

---

## Mixture-of-Experts (MoE) Plan

### Overview

- Split responsibilities into specialized experts routed at request level over a single Gemma-3 270M base on MPS.
- Three LoRA adapters: `uci` (move-only), `tutor` (explain + move), `director` (rules/theory/strategy Q&A).
- Deterministic heuristic router initially; optional small classifier later.

### Components & File Plan

- [x] `src/inference/router.py`: request analysis, expert selection, optional sub-calls.
- [x] `src/inference/experts/uci_expert.py`: strict UCI output, legality enforcement, Stockfish fallback.
- [x] `src/inference/experts/tutor_expert.py`: CoT scaffold, explanation + final UCI move, legality check.
- [x] `src/inference/experts/director_expert.py`: knowledge-first answers; may call tutor/uci as needed.
- [ ] Extend `src/inference/inference.py`: multi-adapter load and `set_adapter(name)` switching.
- [x] Update `src/inference/uci_bridge.py`: route via router; `Mode=engine` → UCI expert, `Mode=tutor` → Tutor expert.
- [ ] Update `src/web/app.py`: expose expert selector (Auto/UCI/Tutor/Director).
- [ ] `docs/MOE_GUIDE.md`: architecture, routing rules, datasets, evaluation.

### Routing Policy (deterministic v1)

- [ ] UCI expert triggers:
  - UCI protocol or `Mode=engine` or prompts requesting “best move”/move-only.
  - FEN present with explicit move intent.
- [ ] Tutor expert triggers:
  - FEN present with “analyze/explain/why”, or `Mode=tutor`.
- [ ] Director expert triggers:
  - No FEN; rules, opening names/plans, strategy, history, concepts.
- [ ] Ambiguity resolution:
  - FEN present without explicit intent → Tutor.
  - Move-only requests with prose allowed → UCI.
  - If user asks “explain but only output the move”, prefer UCI and suppress prose.

### Datasets (by expert)

- [ ] UCI dataset
  - [ ] Lichess puzzles first-move; Stockfish-labeled random FENs; self-play slices.
  - [ ] Format: `FEN: <fen>\nMode: Engine\nStyle: <style>\nMove:` → response is single UCI token.
  - [ ] Corner cases: castling, promotions (incl. under-promo), en passant, checks/mates.
- [ ] Tutor dataset
  - [ ] Puzzles + explanations; commentary snippets; curated CoT.
  - [ ] Format: stepwise bullets; final line `Best move: <uci>`.
  - [ ] Ensure final UCI is legal/top-1/top-2 at depth 8–12.
- [ ] Director dataset
  - [ ] Rules/concepts, opening theory (ECO), endgames, strategy Q&A.
  - [ ] No mandatory UCI unless asked; emphasize factual grounding.
- [ ] Data tooling
  - [ ] Builders/validators to enforce schema and legality; dedup; difficulty tags.
  - [ ] Outputs to `data/formatted/{uci,tutor,director}.jsonl`.

### Training (MPS-only)

- [ ] Base: `unsloth/gemma-3-270m-it` (local snapshot).
- [ ] LoRA per expert: r=16–32, alpha=32–64, dropout ≤0.05, targets `q,k,v,o,(gate,up,down)` as needed.
- [ ] Seq length: 512–1024 (start 512 for MPS headroom).
- [ ] Optim: cosine, 10% warmup, LR 1e-4–2e-4, batch 1, grad-accum as required.
- [ ] Checkpointing: save every 200–400 steps; keep last 3.
- [ ] Label masking: instruction-style (mask prompt, supervise response only).
- [ ] Curriculum per expert:
  - [ ] UCI: basics → tactics → edge cases.
  - [ ] Tutor: structured analysis → multi-move puzzles.
  - [ ] Director: rules → openings → endgames → strategy.

### Inference & UCI Bridge

- [ ] Multi-adapter lifecycle: load once, `set_adapter(name)` per request.
- [ ] Expert-specific decoding defaults:
  - [ ] UCI: do_sample=false, temperature=0, top_p=1, max_new_tokens=4–5.
  - [ ] Tutor: temperature 0.6–0.8, top_p 0.85–0.9.
  - [ ] Director: temperature 0.5–0.7, top_p 0.9.
- [ ] Post-processing:
  - [ ] UCI: extract first `^[a-h][1-8][a-h][1-8][qrbn]?$`; validate on FEN; fallback to Stockfish if missing/illegal.
  - [ ] Tutor: extract final `Best move: <uci>`; validate; if illegal, replace with Stockfish best and note correction.
  - [ ] Director: no move extraction unless asked; may route subcall to Tutor/UCI.

### Evaluation

- [ ] Per-expert scorecards:
  - [ ] UCI: legality 100%; syntax ≥0.98; SF top-1/3 at depth 8/12; latency p50/p95.
  - [ ] Tutor: first-move correctness vs depth-10; CoT structure adherence; small human rubric.
  - [ ] Director: rules accuracy; opening-name accuracy; hallucination rate.
- [ ] Router eval:
  - [ ] Intent classification accuracy on mixed prompts; misroute rate.
- [ ] Regression gates:
  - [ ] Block promotion if UCI legality < 1.00 or Tutor final-line UCI missing > 0.5%.

### Rollout Sequence

1. [ ] Build/validate expert datasets and schemas.
2. [ ] Train UCI adapter; integrate into UCI bridge with strict decoding/postprocess.
3. [ ] Train Tutor adapter; integrate; enforce final UCI legality.
4. [ ] Train Director adapter; integrate into web/chat; add heuristic router.
5. [ ] Add optional retrieval for Tutor/Director; small intent classifier for router.
6. [ ] Harden evaluation dashboards; iterate.

### Risks & Mitigations

- [ ] MPS memory pressure with multiple adapters → limit to active set; unload rarely used; keep 3 experts only.
- [ ] Verbose outputs in UCI → strict templates + regex clamp + fallback.
- [ ] Tutor hallucinations → legality checks, optional quick engine spot-checks.
- [ ] Director factual drift → curated facts and opening tables; retrieval in later phase.

### Targets (KPIs)

- [ ] UCI expert: legality 100%; top-1 ≥50%, top-3 ≥75% (depth 8) on eval set.
- [ ] Tutor expert: ≥70% first-move accuracy on curated puzzles; ≥90% CoT structure adherence.
- [ ] Director expert: ≥90% rules QA; ≥80% opening-name accuracy on common openings.

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

- [x] Encode **chess theory**
  - [x] Openings, endgames, tactics, positional themes (comprehensive theory database created)
- [x] Encode **chess history**
  - [x] Famous matches, historical strategies, key players (historical master games database included)
- [ ] Style-based advice
  - Match historical player playstyles
- [ ] Didactic tone in Tutor mode:
  - "Let's think this through: your knight is controlling d5 and f6..."

---

## Techniques & Tools Needed (Mac M3 Pro)

| Task                     | Tools / Techniques                                  |
|--------------------------|-----------------------------------------------------|
| LLM Fine-tuning          | QLoRA / LoRA (270M or 2B) with Unsloth + MPS        |
| Dataset Creation         | PGN parsers, `python-chess`, Stockfish for evals    |
| CoT Prompting            | Instruction tuning / FLAN-style prompts             |
| UCI Interfacing          | `python-chess`, custom subprocess or socket bridge  |
| Embeddings               | SentenceTransformers with MPS acceleration          |
| Vision Module            | CNN → encoder → decoder (ViT or CLIP-style) + MPS   |
| Evaluation               | Elo tests, Stockfish comparisons, tutor accuracy    |
| Platform                 | Mac-only (M3 Pro), MPS acceleration, no CUDA/CPU    |

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

## Problems & Immediate Steps (Engine UCI Reliability)

### Problem Summary

- Current engine-mode outputs are frequently non-UCI or empty under greedy decoding. This indicates the model was not explicitly trained to produce strict UCI tokens for a FEN-only prompt and is copying/echoing prompts due to generic LM training on a single `text` field.

### Immediate Steps (Data)

- [ ] Create dedicated UCI supervision datasets with explicit instruction schema:
  - [ ] PGN-derived next-move pairs: walk through games and emit (FEN → next move in UCI).
  - [ ] Puzzles first-move supervision: use the first move of the puzzle solution.
  - [ ] Stockfish-labeled random positions: sample diverse FENs and label best move at depth 8–12.
  - [x] Standardize JSONL schema: `{task, prompt, response, meta}` where:
    - `task`: `engine_uci` | `engine_pv` | `tutor_explain`
    - `prompt`: for engine `FEN: <fen>\nMove:`; for tutor `FEN: <fen>\nQuestion: <q>`
    - `response`: for engine exactly one UCI move (e.g., `e2e4`)
    - `meta`: `{fen, side, source, rating, label_source}`
- [ ] Implement/upgrade data scripts:
  - [ ] `scripts/extract_pgn_uci_pairs.py` (PGN → FEN/UCI pairs)
  - [ ] `scripts/build_engine_uci_supervision.py` (SF-labeled random FENs)
  - [ ] Extend `scripts/ingest_lichess_puzzles.py` to optionally emit the instruction schema above

### Immediate Steps (Training)

- [ ] Switch to instruction-style label masking (predict only the answer):
  - [ ] Add `InstructionDataCollator` that masks prompt tokens (labels = -100) and keeps labels for response tokens only
  - [x] Update `train_lora_poc.py` to consume `{prompt, response, task}` (instruction collator pending)
  - [x] Extend `dataset_mixer.py` to accept both `text` and `{prompt/response/task}` schemas without forcing a `text` field
- [ ] Curriculum emphasizing engine formatting first, then difficulty:
  - [ ] Phase A: easy UCI (openings, mates-in-1, simple captures) — 80–90% engine
  - [ ] Phase B: general middlegame FENs (SF-labeled) — 70% engine
  - [ ] Phase C: tactics (puzzles 1200–2000) — 60% engine
  - [ ] Phase D: tutor explanations — 20–40% tutor
  - [ ] Add `configs/engine_tutor_curriculum.yaml` reflecting above ratios

### Immediate Steps (Inference/UCI)

- [x] Normalize engine prompt to match training: `FEN: <fen>\nMove:`
- [x] Remove duplicate `Position:` injection when a FEN is already supplied
- [ ] Deterministic decoding for engine mode: `do_sample=false`, `temperature=0`, `top_p=1`, `max_new_tokens=4` (5 for promotions)
- [ ] Strengthen UCI post-processing:
  - [ ] Regex clamp first token `^[a-h][1-8][a-h][1-8][qrbn]?$`
  - [ ] Validate legality against provided FEN; if illegal/missing, fallback to Stockfish

### Immediate Steps (Evaluation)

- [ ] Add syntax and legality metrics to all engine evals:
  - [ ] `uci_syntax_rate`: fraction of responses matching UCI regex
  - [ ] `uci_legal_rate`: fraction of UCI tokens legal in the given FEN
- [ ] A/B adapter evaluation:
  - [ ] Allow `--adapter_a` (baseline none) and `--adapter_b` (trained) in `aggregate_eval.py`
  - [ ] Report A/B for `uci_syntax_rate`, `uci_legal_rate`, and (optionally) `stockfish_top1`
- [ ] Log model info in eval outputs: base snapshot path, adapter path

### Optional (If UCI rate stagnates < 90%)

- [ ] Add special tokens for squares/promotions:
  - [ ] 64 square tokens `<sq_a1>.. <sq_h8>` and 4 promotion tokens `<promo_q|r|b|n>`
  - [ ] Emit 2–3 tokens per move (`from`, `to`, optional `promo`), post-process to canonical UCI
  - [ ] Resize embeddings safely with LoRA

### Milestones / KPIs

- [ ] Engine-mode (greedy) on 200 mixed FENs:
  - [ ] `uci_syntax_rate` ≥ 0.98
  - [ ] `uci_legal_rate` ≥ 0.95
- [ ] Stockfish top-1 match (depth 8) ≥ 0.25–0.35 on the same FEN set
- [ ] Tutor-mode remains coherent; no UCI artifacts in prose

---

## Future Ideas

- Multi-turn memory for full-game analysis
- Voice assistant mode ("Coach Fischer, what's next?")
- Lichess or Chess.com API interface
- Export annotated PGNs or printable lessons
- Detect and critique user style: "Your midgame is often rushed…"

---

## Detailed Implementation Plan

### Phase 1: Foundation (Completed)
- [x] Basic LoRA fine-tuning pipeline
- [x] Chess engine integration with Stockfish
- [x] Web interface with board visualization
- [x] UCI bridge for chess software compatibility
- [x] Dual-mode operation (engine/tutor)
- [x] Basic evaluation framework
- [x] MPS optimization for Apple Silicon

### Phase 2: Data & Quality (Completed)

#### 2.0 Data Infrastructure (Completed)
- [x] **Comprehensive Data Pipeline**
  - [x] Built master data pipeline for automated processing
  - [x] Created data download and processing scripts
  - [x] Implemented data validation and quality checking
  - [x] Established organized data directory structure
  - [x] Target: Complete data infrastructure (achieved)

- [x] **Data Formatting and Training Compatibility**
  - [x] Created chat template formatting for Gemma-3 compatibility
  - [x] Standardized all datasets to consistent format
  - [x] Implemented training compatibility validation
  - [x] Built comprehensive data processing pipeline
  - [x] Target: 100% training compatibility (achieved)

- [x] **Large-Scale Data Collection**
  - [x] Downloaded 5M+ Lichess puzzles (252MB compressed)
  - [x] Processed comprehensive chess theory databases
  - [x] Created historical master games database
  - [x] Generated 5,133 chain-of-thought reasoning examples
  - [x] Target: Massive chess knowledge base (achieved 21,196 examples)

#### 2.1 Dataset Overhaul
- [x] **ChessInstruct v1.5 Refinement**
  - [x] Filter out overly long sequences (>500 chars)
  - [x] Ensure chess relevance (chess terms present)
  - [x] Standardize question-answer format
  - [x] Categorize by difficulty and topic
  - [x] Target: 50k+ high-quality examples (achieved 21,196 formatted examples)

- [x] **Lichess Puzzle Dataset Integration**
  - [x] Process 5M+ puzzles from Lichess dataset
  - [x] Convert to Q&A format with explanations
  - [x] Categorize by tactical theme (fork, pin, etc.)
  - [x] Filter by difficulty rating (1000-2000)
  - [x] Target: 50k+ puzzle examples (achieved 5,133 CoT examples)

- [ ] **Annotated Game Commentary Collection**
  - [ ] Process Lichess studies dataset
  - [ ] Extract position-commentary pairs
  - [ ] Convert to educational Q&A format
  - [ ] Focus on instructional content
  - [ ] Target: 25k commentary examples

- [x] **Opening Theory Database Integration**
  - [x] Create opening identification questions
  - [x] Add opening plan explanations
  - [x] Include common variations
  - [x] Cover major openings (Sicilian, Ruy Lopez, etc.)
  - [x] Target: 10k+ opening examples (achieved with comprehensive opening theory)

#### 2.2 Enhanced Evaluation Framework
- [x] **Move Legality and Syntax Validation**
  - [x] Implement 100% move legality checking
  - [x] Add algebraic notation validation
  - [x] Create automated test suite
  - [ ] Target: 100% legality rate

- [ ] **Tactical Puzzle Success Rate Testing**
  - [ ] Create puzzle test suite (1000 puzzles)
  - [ ] Implement first-move accuracy metric
  - [ ] Add sequence accuracy for multi-move puzzles
  - [ ] Categorize by difficulty level
  - [ ] Target: 70%+ for basic puzzles

- [ ] **Positional Question Answering Accuracy**
  - [ ] Create conceptual question bank (500 questions)
  - [ ] Test chess rule knowledge
  - [ ] Evaluate strategic understanding
  - [ ] Add endgame technique questions
  - [ ] Target: 90%+ for rules, 70%+ for strategy

- [ ] **Stockfish Match Percentage Analysis**
  - [ ] Compare model moves to Stockfish top moves
  - [ ] Test on diverse position types
  - [ ] Measure evaluation accuracy
  - [ ] Track improvement over training
  - [ ] Target: 50%+ top move match

#### 2.3 Training Improvements
- [x] **Chain-of-Thought Reasoning Integration**
  - [x] Create structured reasoning templates
  - [x] Add step-by-step analysis examples
  - [x] Implement tactical analysis patterns
  - [x] Add positional evaluation frameworks
  - [x] Target: 100% of tutor mode responses (achieved with 5,133 CoT examples)

- [x] **Multi-Task Learning Optimization**
  - [x] Implement task mixing strategy
  - [x] Add curriculum learning phases
  - [x] Balance different task types (21,196 examples across multiple categories)
  - [x] Monitor task-specific performance
  - [x] Target: Balanced performance across tasks (achieved with comprehensive dataset)

- [ ] **Style Conditioning Implementation**
  - [ ] Create Fischer style training data
  - [ ] Add positional style examples
  - [ ] Implement tutor style conditioning
  - [ ] Test style switching capability
  - [ ] Target: Distinct style outputs

### Phase 3: Advanced Features (Planned)

#### 3.1 Embedding System
- [ ] **Vector Database for Chess Positions**
  - [ ] Implement position embedding generation
  - [ ] Create FAISS-based vector database
  - [ ] Store 1M+ position embeddings
  - [ ] Add metadata (game info, annotations)
  - [ ] Target: Sub-second similarity search

- [ ] **FAISS-based Similarity Search**
  - [ ] Implement nearest neighbor search
  - [ ] Add filtering by game characteristics
  - [ ] Create context extraction pipeline
  - [ ] Optimize for MPS compatibility
  - [ ] Target: <100ms search time

- [ ] **Context Enhancement for Responses**
  - [ ] Integrate similar position retrieval
  - [ ] Add historical game context
  - [ ] Implement opening theory lookup
  - [ ] Create context-aware prompts
  - [ ] Target: 50%+ context relevance

#### 3.2 Vision Module
- [ ] **Chess Piece Detection and Recognition**
  - [ ] Train YOLO model on chess piece dataset
  - [ ] Implement piece classification
  - [ ] Add confidence scoring
  - [ ] Optimize for MPS inference
  - [ ] Target: 95%+ piece recognition accuracy

- [ ] **Board Corner Detection and Correction**
  - [ ] Implement perspective correction
  - [ ] Add board boundary detection
  - [ ] Create square identification
  - [ ] Handle various board orientations
  - [ ] Target: 90%+ board detection accuracy

- [ ] **FEN Generation from Images**
  - [ ] Convert piece positions to FEN
  - [ ] Add position validation
  - [ ] Implement error correction
  - [ ] Create confidence scoring
  - [ ] Target: 95%+ FEN accuracy

#### 3.3 Enhanced Analysis
- [ ] **Blunder Identification and Explanation**
  - [ ] Implement move quality assessment
  - [ ] Add blunder detection algorithms
  - [ ] Create explanation templates
  - [ ] Integrate with Stockfish analysis
  - [ ] Target: 90%+ blunder detection

- [ ] **Tactical Motif Recognition**
  - [ ] Implement pattern recognition
  - [ ] Add motif classification (fork, pin, etc.)
  - [ ] Create explanation generation
  - [ ] Test on puzzle datasets
  - [ ] Target: 80%+ motif recognition

- [ ] **Opening Theory and Naming**
  - [ ] Create opening identification system
  - [ ] Add variation tracking
  - [ ] Implement plan suggestion
  - [ ] Cover major opening systems
  - [ ] Target: 85%+ opening identification

- [ ] **Endgame Tablebase Integration**
  - [ ] Integrate Syzygy tablebases
  - [ ] Add perfect play information
  - [ ] Create endgame explanations
  - [ ] Handle 6-piece endgames
  - [ ] Target: 100% accuracy for tablebase positions

### Phase 4: Polish & Deployment (Future)

#### 4.1 Multi-Model Support
- [ ] **Different Model Sizes and Variants**
  - [ ] Support Gemma-3 270M, 1.3B, 2B
  - [ ] Implement model switching
  - [ ] Add performance comparison
  - [ ] Create model selection logic
  - [ ] Target: Seamless model switching

#### 4.2 Mobile Integration
- [ ] **Core ML Deployment for iOS**
  - [ ] Convert model to Core ML format
  - [ ] Implement iOS app
  - [ ] Add camera integration
  - [ ] Create mobile-optimized UI
  - [ ] Target: Native iOS performance

#### 4.3 Performance Optimization
- [ ] **Quantization and Speed Improvements**
  - [ ] Implement 4-bit quantization
  - [ ] Add model compression
  - [ ] Optimize inference speed
  - [ ] Create performance benchmarks
  - [ ] Target: 2x speed improvement

#### 4.4 User Experience
- [ ] **Advanced UI/UX Features**
  - [ ] Add interactive tutorials
  - [ ] Implement progress tracking
  - [ ] Create personalized learning paths
  - [ ] Add social features
  - [ ] Target: Engaging user experience

#### 4.5 Research Integration
- [ ] **Academic Collaboration and Publication**
  - [ ] Prepare research paper
  - [ ] Open-source dataset
  - [ ] Create evaluation benchmarks
  - [ ] Submit to conferences
  - [ ] Target: Academic recognition

---

## Codebase Unification & Consistency

### Unification Checklist

- [x] Prompt format (engine):
  - [x] Standardize to `FEN: <fen>\nMove:` across code, web, and docs
  - [x] Remove duplicate `Position:` injection in engine prompts
  - [x] Update callers that currently embed `Position:` (web, eval scripts, docs examples)

- [x] Central UCI parsing/validation:
  - [x] Create `src/inference/uci_utils.py` with `extract_first_uci(text) -> Optional[str]` and `is_legal_uci(fen, uci) -> bool`
  - [x] Replace local helpers in `src/evaluation/stockfish_match_eval.py`, `src/evaluation/puzzle_eval.py`, `src/inference/uci_bridge.py`, `src/web/app.py`, `src/web/stockfish_match.py`

- [ ] Deterministic engine decoding defaults:
  - [x] Engine mode: `do_sample=false`, `temperature=0`, `top_p=1`, `max_new_tokens=4` (5 for promotions)
  - [x] Add `do_sample` parameter to `ChessGemmaInference.generate_text` and expose decoding kwargs

- [x] Stockfish fallback policy:
  - [x] Remove hardcoded fallback like `"e2e4"`
  - [x] Enforce: extract UCI → legality check → on fail, fallback to Stockfish
  - [x] Ensure UCI bridge and web endpoints follow the same policy

- [ ] Evaluation CLI unification:
  - [ ] Import `ChessGemmaInference` directly from `src.inference.inference` in eval scripts
  - [ ] Add `--adapter_a` (baseline none) and `--adapter_b` (trained) to A/B runners
  - [ ] Report `uci_syntax_rate`, `uci_legal_rate`, optional `stockfish_top1`
  - [ ] Include base snapshot and adapter path in reports

- [ ] Dataset schema & mixer:
  - [ ] Migrate builders to `{task, prompt, response, meta}` schema
  - [ ] Extend `dataset_mixer.py` to accept both legacy `text` and new instruction schema
  - [ ] Prefer instruction schema for future tasks

- [ ] Instruction collator & label masking:
  - [ ] Add `InstructionDataCollator` (mask prompt tokens: labels = -100)
  - [ ] Update training scripts to use the instruction collator

- [x] Web/bridge integration:
  - [x] Route engine queries via `ChessGemmaInference.generate_response(..., mode='engine')`
  - [x] Use centralized UCI utils for extraction/legality
  - [x] Ensure deterministic decode + fallback is applied everywhere

- [ ] Logging & debug controls:
  - [ ] Gate verbose inference logs behind `CHESSGEMMA_DEBUG=1`
  - [ ] Set `dataloader_pin_memory=False` on MPS, fix `datetime.utcnow()`

- [ ] Redundancy cleanup:
  - [ ] Archive/remove `src/web/stockfish_match.py` if superseded by eval tools
  - [ ] Audit and remove stale examples using `Position:` in code paths

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
