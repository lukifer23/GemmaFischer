# Project Plan: GemmaFischer — Chess LLM Engine + Tutor

## Summary

**GemmaFischer** is a lightweight language model (based on Gemma 3B or 270M) designed to function as both:

- A **chess engine**, capable of interfacing with UCI backends, evaluating positions, and suggesting moves
- A **chess tutor/analyst**, able to explain concepts, annotate games, and provide historical insight through chain-of-thought (CoT) reasoning

**Platform**: Mac-only (M3 Pro) with MPS acceleration - no CUDA/CPU fallbacks.

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

## Future Ideas

- Multi-turn memory for full-game analysis
- Voice assistant mode ("Coach Fischer, what's next?")
- Lichess or Chess.com API interface
- Export annotated PGNs or printable lessons
- Detect and critique user style: "Your midgame is often rushed…"

---

## Detailed Implementation Plan

### Phase 1: Foundation (Completed) ✅
- [x] Basic LoRA fine-tuning pipeline
- [x] Chess engine integration with Stockfish
- [x] Web interface with board visualization
- [x] UCI bridge for chess software compatibility
- [x] Dual-mode operation (engine/tutor)
- [x] Basic evaluation framework
- [x] MPS optimization for Apple Silicon

### Phase 2: Data & Quality (In Progress) 🚧

#### 2.1 Dataset Overhaul
- [ ] **ChessInstruct v1.5 Refinement**
  - [ ] Filter out overly long sequences (>500 chars)
  - [ ] Ensure chess relevance (chess terms present)
  - [ ] Standardize question-answer format
  - [ ] Categorize by difficulty and topic
  - [ ] Target: 50k high-quality examples

- [ ] **Lichess Puzzle Dataset Integration**
  - [ ] Process 100k puzzles from Lichess dataset
  - [ ] Convert to Q&A format with explanations
  - [ ] Categorize by tactical theme (fork, pin, etc.)
  - [ ] Filter by difficulty rating (1000-2000)
  - [ ] Target: 50k puzzle examples

- [ ] **Annotated Game Commentary Collection**
  - [ ] Process Lichess studies dataset
  - [ ] Extract position-commentary pairs
  - [ ] Convert to educational Q&A format
  - [ ] Focus on instructional content
  - [ ] Target: 25k commentary examples

- [ ] **Opening Theory Database Integration**
  - [ ] Create opening identification questions
  - [ ] Add opening plan explanations
  - [ ] Include common variations
  - [ ] Cover major openings (Sicilian, Ruy Lopez, etc.)
  - [ ] Target: 10k opening examples

#### 2.2 Enhanced Evaluation Framework
- [ ] **Move Legality and Syntax Validation**
  - [ ] Implement 100% move legality checking
  - [ ] Add algebraic notation validation
  - [ ] Create automated test suite
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
- [ ] **Chain-of-Thought Reasoning Integration**
  - [ ] Create structured reasoning templates
  - [ ] Add step-by-step analysis examples
  - [ ] Implement tactical analysis patterns
  - [ ] Add positional evaluation frameworks
  - [ ] Target: 100% of tutor mode responses

- [ ] **Multi-Task Learning Optimization**
  - [ ] Implement task mixing strategy
  - [ ] Add curriculum learning phases
  - [ ] Balance different task types
  - [ ] Monitor task-specific performance
  - [ ] Target: Balanced performance across tasks

- [ ] **Style Conditioning Implementation**
  - [ ] Create Fischer style training data
  - [ ] Add positional style examples
  - [ ] Implement tutor style conditioning
  - [ ] Test style switching capability
  - [ ] Target: Distinct style outputs

### Phase 3: Advanced Features (Planned) 📋

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

### Phase 4: Polish & Deployment (Future) 🔮

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
