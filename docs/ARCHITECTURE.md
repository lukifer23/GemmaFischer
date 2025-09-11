# GemmaFischer Architecture

## System Overview

GemmaFischer is a comprehensive chess AI system that functions as both a chess engine (UCI-compatible) and a chess tutor/analyst. Built around fine-tuned language models with chain-of-thought reasoning, it provides both tactical analysis and educational explanations. The architecture follows a clear separation of concerns with distinct layers for training, inference, evaluation, and presentation.

**Platform**: Mac-only (M3 Pro) with MPS acceleration - no CUDA/CPU fallbacks.

### Key Design Principles
- **Modular Architecture**: Clean separation of concerns for easy extension
- **Dual-Mode Operation**: Engine mode (fast moves) and Tutor mode (explanations)
- **Hybrid Intelligence**: Combines LLM reasoning with chess engine precision
- **Retrieval-Augmented Generation**: Context enhancement through similar position lookup
- **Chain-of-Thought Reasoning**: Step-by-step analysis and explanation
- **Style Conditioning**: Historical player style emulation capabilities

## High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Interface │    │ Training System │    │ Expert Adapters │    │ Evaluation Suite│
│   (Flask + API) │    │  (Multi-Expert) │    │  (LoRA Models)  │    │  (Chess Metrics)│
│   http://localhost│    │                 │    │                 │    │                 │
│       :5001      │    │                 │    │                 │    │                 │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │                      │
          └──────────────────────┼──────────────────────┼──────────────────────┘
                                 │                      │
                    ┌─────────────┴─────────────┐       │
                    │   Dynamic Inference       │       │
                    │ (Adapter Switching + CoT) │       │
                    └─────────────┬─────────────┘       │
                                 │                      │
                    ┌─────────────┴─────────────┐       │
                    │    Expert Training        │       │
                    │   (UCI/Tutor/Director)    │       │
                    └─────────────┬─────────────┘       │
                                 │                      │
                    ┌─────────────┴─────────────┐       │
                    │    UCI Bridge Layer       │       │
                    │  (Engine + Tutor Modes)   │       │
                    └─────────────┬─────────────┘       │
                                 │                      │
                    ┌─────────────┴─────────────┐       │
                    │    Chess Engine Layer     │       │
                    │     (Stockfish)           │       │
                    └───────────────────────────┘       │
                                                       │
                                              ┌────────┴────────┐
                                              │ Dataset Pipeline │
                                              │ (100k+ samples)  │
                                              └─────────────────┘
```

## Component Details

### 1. Expert Training System (`src/training/`)

**Purpose**: Multi-expert LoRA fine-tuning system supporting UCI, Tutor, and Director modes with specialized training for each chess domain

**Key Components**:
- `train_lora_poc.py`: Main expert-aware training orchestrator
- `configs/`: Expert-specific configuration files (lora_uci.yaml, lora_tutor.yaml, lora_director_expert.yaml)
- `dataset_mixer.py`: Weighted dataset interleaving and curriculum phases
- `train.py`: Legacy single-expert training (maintained for compatibility)

**Architecture**:
```
Expert Training Pipeline
├── Expert Selection (UCI/Tutor/Director)
├── Dataset Loading (50k+ samples per expert)
├── Configuration Auto-Loading
├── Model Loading (Gemma-3 base model)
├── LoRA Adapter Attachment (Expert-specific)
├── MPS-Optimized Training (Apple Silicon)
├── Real-time Progress Monitoring
├── Checkpoint Management (per expert)
└── Adapter Saving (checkpoints/lora_{expert}/)
```

**Expert-Specific Features**:
- **UCI Expert**: Chess move generation with 50k UCI-format training samples
- **Tutor Expert**: Chess explanations with 50k tutor-format training samples
- **Director Expert**: Q&A reasoning with curated reasoning examples
- **Live Adapter Switching**: Dynamic model loading in web interface
- **Curriculum Training**: Optional phased learning schedules

**Performance Features**:
- **MPS Acceleration**: Native Apple Silicon optimization (2-3 steps/sec on M3 Pro)
- **Memory Efficient**: 4-6GB peak usage with gradient checkpointing
- **Resume Capability**: Training continuation from checkpoints
- **Multi-Expert Parallel**: Train different experts simultaneously
- **Real-time Monitoring**: System stats and progress tracking

### 2. Dynamic Inference Layer (`src/inference/`)

**Purpose**: Intelligent adapter switching and inference with expert model selection based on query type

**Key Components**:
- `inference.py`: Main inference engine with adapter management
- `chess_engine.py`: Stockfish integration and validation
- `uci_utils.py`: UCI move extraction and validation utilities
- `prompt_templates/`: Expert-specific prompt formatting

**Architecture**:
```
Dynamic Inference Pipeline
├── Query Analysis (UCI/Tutor/Director classification)
├── Expert Adapter Selection (Automatic routing)
├── Model Loading (Base + Expert LoRA adapter)
├── Prompt Construction (Expert-specific formatting)
├── Text Generation (Gemma-3 + Chain-of-Thought)
├── Response Post-processing (Expert-specific)
├── Chess Validation (Stockfish integration)
└── Fallback Handling (Multi-level safety nets)
```

**Expert-Specific Inference**:
- **UCI Expert**: Fast move generation with legal move validation
- **Tutor Expert**: Explanatory responses with step-by-step reasoning
- **Director Expert**: Q&A responses with tactical analysis
- **Adapter Switching**: Sub-second switching between experts
- **Memory Management**: Efficient model caching and unloading

### 3. UCI Bridge Layer (`src/inference/`)

**Purpose**: Full UCI protocol compatibility with chess software integration

**Key Components**:
- `uci_bridge.py`: UCI protocol implementation
- `inference.py`: Engine mode inference
- `chess_engine.py`: Stockfish fallback integration

**Architecture**:
```
UCI Bridge Pipeline
├── UCI Protocol Handler (position, go commands)
├── Expert Selection (UCI expert adapter)
├── Position Analysis (FEN processing)
├── Move Generation (Fast inference mode)
├── Legal Move Validation (Stockfish check)
├── UCI Output Formatting (bestmove command)
└── Error Recovery (Fallback strategies)
```

**UCI Features**:
- **Full Protocol Support**: Complete UCI v2 compliance
- **Fast Inference**: Optimized for tournament play
- **Move Validation**: Every move checked for legality
- **Stockfish Fallback**: Backup when model fails
- **Engine Integration**: Compatible with chess GUIs and tournaments

### 3. Chess Engine Integration (`src/inference/chess_engine.py`)

**Purpose**: Provide chess-specific validation and analysis

**Key Components**:
- `ChessEngineManager`: Main engine interface
- `MoveAnalysis`: Move validation results
- `PositionAnalysis`: Position evaluation

**Architecture**:
```
Chess Engine Layer
├── Stockfish Process Management
├── Move Validation
├── Position Analysis
├── Tactical Pattern Recognition
└── Natural Language Generation
```

**Key Features**:
- Concurrent move validation
- Position classification (opening/middle/endgame)
- Threat and opportunity detection
- Move quality assessment

### Unified Fallback Strategy

All move-generating entry points share a common safety net:

1. The language model proposes a move in text form.
2. The system extracts the first UCI token from the response.
3. The token is validated against the current FEN.
4. If extraction fails or the move is illegal, `ChessEngineManager` queries Stockfish for a legal move.

This policy is applied in the core inference module, the UCI bridge, and the web API to ensure that callers always receive a legal move.

### 4. Embedding System (Planned)

**Purpose**: Similar position retrieval and context enhancement

**Key Components**:
- `PositionEmbedder`: Generate embeddings for chess positions
- `VectorDatabase`: FAISS-based similarity search
- `ContextRetriever`: Find similar positions and extract context

**Architecture**:
```
Embedding System
├── Position Encoding (FEN → Vector)
├── Vector Database (FAISS)
├── Similarity Search
├── Context Extraction
└── Prompt Enhancement
```

**Planned Features**:
- Chess position vectorization
- Similar position lookup
- Historical game context retrieval
- Opening theory integration

### 5. Vision Module (Planned)

**Purpose**: Board image to FEN conversion

**Key Components**:
- `BoardDetector`: Find chessboard in images
- `PieceRecognizer`: Identify pieces and positions
- `FENGenerator`: Convert to FEN notation

**Architecture**:
```
Vision Pipeline
├── Image Preprocessing
├── Board Detection
├── Piece Recognition
├── Position Validation
└── FEN Generation
```

**Planned Features**:
- Real-world board image processing
- Piece detection and classification
- Perspective correction
- FEN validation and error handling

### 4. Web Interface & API (`src/web/`)

**Purpose**: Comprehensive web application providing full training, evaluation, and chess analysis capabilities

**Key Components**:
- `app.py`: Main Flask application with REST API
- `templates/`: HTML templates for web interface
- `chess_game.py`: Interactive chess board and game management
- `stockfish_match.py`: Evaluation match coordination

**Architecture**:
```
Web Interface Architecture
├── Flask Application (http://localhost:5001)
├── Training Dashboard (/training)
│   ├── Expert Selection (UCI/Tutor/Director)
│   ├── Real-time Progress Monitoring
│   ├── System Resource Tracking
│   └── Training Control (Start/Stop/Resume)
├── Chess Analysis Interface (/analysis)
│   ├── Interactive Chess Board
│   ├── Real-time Q&A System
│   ├── Move Validation & Suggestions
│   └── Expert Model Switching
├── Evaluation Suite (/evaluation)
│   ├── Stockfish Match Testing
│   ├── Puzzle Accuracy Evaluation
│   ├── Live Results Display
│   └── Performance Metrics
├── Dataset Management (/datasets)
│   ├── Data Cleaning Tools
│   ├── Stockfish Validation
│   ├── Processing Status
│   └── Quality Assurance
└── REST API Layer
    ├── Training Endpoints (/api/train/*)
    ├── Chess Endpoints (/api/game/*)
    ├── Evaluation Endpoints (/api/eval/*)
    └── Dataset Endpoints (/api/data/*)
```

**Core Features**:
- **Training Interface**: Complete GUI for training all expert models
- **Interactive Chess Board**: Click-to-move interface with legal move validation
- **Real-time Q&A**: Ask questions about any chess position with expert responses
- **Model Switching**: Dynamic switching between trained expert adapters
- **Evaluation Tools**: Built-in testing against Stockfish and puzzle databases
- **Dataset Processing**: Web-based data cleaning and validation tools
- **System Monitoring**: Real-time resource usage and performance tracking

**API Endpoints**:
- **Training**: `/api/train/start`, `/api/train/status`, `/api/train/stop`
- **Chess Game**: `/api/game/move`, `/api/game/analyze`, `/api/game/ai_move`
- **Evaluation**: `/api/eval/stockfish`, `/api/eval/puzzles`, `/api/eval/status`
- **Dataset**: `/api/data/clean`, `/api/data/status`
- **System**: `/api/health`, `/api/stats`, `/api/examples`

### 5. Evaluation Layer (`src/evaluation/`)

**Purpose**: Comprehensive evaluation suite for chess model performance assessment

**Key Components**:
- `stockfish_match_eval.py`: Stockfish vs model match evaluation
- `puzzle_eval.py`: Tactical puzzle accuracy testing
- `chess_evaluation.py`: General evaluation framework

**Architecture**:
```
Evaluation Pipeline
├── Stockfish Match Testing
│   ├── Position Generation (Mixed FENs)
│   ├── Model vs Stockfish Games
│   ├── Win/Loss/Draw Tracking
│   └── Performance Metrics
├── Puzzle Accuracy Testing
│   ├── Lichess Puzzle Database
│   ├── First-Move Accuracy
│   ├── Sequence Accuracy
│   └── Difficulty Analysis
├── Real-time Web Evaluation
│   ├── Live Results Display
│   ├── Progress Monitoring
│   └── Comparative Analysis
└── Performance Reporting
    ├── Accuracy Metrics
    ├── Response Time Analysis
    ├── Error Classification
    └── Benchmark Comparisons
```

**Key Features**:
- **Stockfish Match Evaluation**: Automated games against Stockfish at various depths
- **Puzzle Database Testing**: Accuracy testing on 1000+ rated chess puzzles
- **Real-time Evaluation**: Live evaluation progress in web interface
- **Comprehensive Metrics**: Win rates, accuracy scores, response times
- **Error Analysis**: Classification of model mistakes and failure modes

### 6. Dataset Pipeline (`data/`)

**Purpose**: Large-scale dataset processing and quality assurance for training

**Key Components**:
- `data/scripts/validate_and_augment.py`: Dataset validation and cleaning
- `data/processed/`: Cleaned training datasets (100k+ samples)
- `data/formatted/`: Expert-specific dataset symlinks

**Architecture**:
```
Dataset Processing Pipeline
├── Raw Data Collection
│   ├── ChessInstruct Dataset
│   ├── Lichess Puzzles
│   ├── Historical Games
│   └── Opening Theory
├── Data Validation & Cleaning
│   ├── Stockfish Move Validation
│   ├── Legal Move Verification
│   ├── Format Standardization
│   └── Quality Assurance
├── Expert-Specific Formatting
│   ├── UCI Expert Dataset (50k samples)
│   ├── Tutor Expert Dataset (50k samples)
│   ├── Director Expert Dataset (3.2MB)
│   └── Format Conversion
├── Quality Control
│   ├── Duplicate Removal
│   ├── Error Detection
│   ├── Balance Verification
│   └── Final Validation
└── Training-Ready Datasets
    ├── Symlink Management
    ├── Version Control
    └── Distribution
```

**Dataset Features**:
- **100k+ Training Samples**: Comprehensive chess knowledge base
- **Stockfish Validation**: All moves verified for legality
- **Expert Specialization**: Tailored datasets for each expert type
- **Quality Assurance**: Automated cleaning and validation pipeline
- **Symlink Management**: Clean dataset routing to training configs

## Data Flow

### Training Flow
1. Load ChessInstruct dataset
2. Convert to chat format
3. Initialize Gemma-3 with LoRA
4. Train with SFTTrainer
5. Save checkpoints and adapters

### Inference Flow
1. Load base model and adapter
2. Process user question
3. Generate response with Gemma-3
4. Validate moves with Stockfish (optional)
5. Return formatted response

### Web Interface Flow
1. Receive HTTP request
2. Load model (if not cached)
3. Generate response
4. Format for web display
5. Return JSON response

## Configuration Management

### Training Configuration
- YAML-based configuration files
- Environment-specific settings
- Hyperparameter management
- Checkpoint strategies

### Model Configuration
- LoRA parameters (rank, alpha, dropout)
- Target modules selection
- Device configuration (MPS/CPU)
- Memory optimization settings

## Error Handling

### Training Errors
- Memory overflow handling
- Checkpoint corruption recovery
- Data loading failures
- Model convergence issues

### Inference Errors
- Model loading failures
- Generation errors
- Chess engine unavailability
- Response formatting issues

### Web Interface Errors
- API endpoint failures
- Model unavailability
- Request validation errors
- Response timeout handling

## Performance Considerations

### Memory Management
- Gradient checkpointing
- Model quantization options
- Batch size optimization
- Memory monitoring

### Speed Optimization
- Model caching
- Concurrent processing
- MPS acceleration
- Response streaming

### Scalability
- Horizontal scaling options
- Load balancing strategies
- Database integration potential
- Cloud deployment considerations

## Security Considerations

### Model Security
- Adapter integrity validation
- Model versioning
- Access control
- Input sanitization

### Web Security
- Request validation
- Rate limiting
- CORS configuration
- Error message sanitization

## Monitoring and Logging

### Training Monitoring
- Loss tracking
- Learning rate scheduling
- Memory usage monitoring
- Checkpoint validation

### Inference Monitoring
- Response time tracking
- Model performance metrics
- Error rate monitoring
- Usage analytics

### System Monitoring
- Resource utilization
- Process health checks
- Error alerting
- Performance dashboards

## Data Flow Architecture

### Training Pipeline
```
Raw Data → Preprocessing → Formatting → LoRA Training → Model Checkpoint
    ↓
ChessInstruct → Chat Format → Dual-Mode Data → Unsloth Training → Adapter
```

### Inference Pipeline
```
User Input → Mode Selection → Prompt Construction → Model Generation → Validation
    ↓
Question → Engine/Tutor → FEN + Question → LLM Response → Stockfish Check
```

### Retrieval-Augmented Generation (Planned)
```
Position → Embedding → Similar Search → Context Retrieval → Enhanced Prompt
    ↓
FEN → Vector → FAISS Lookup → Historical Games → Context + Question
```

## Integration Patterns

### Chess Engine Integration
- **Validation**: Every move suggestion is validated by Stockfish
- **Analysis**: Position evaluation and tactical analysis
- **Fallback**: Stockfish provides backup when model fails

### Multi-Modal Integration (Planned)
- **Text**: Natural language questions and explanations
- **Position**: FEN notation and board states
- **Images**: Real-world chess board photos
- **Context**: Historical games and similar positions

### Style Conditioning (Planned)
- **Fischer Style**: Aggressive, tactical play
- **Positional Style**: Strategic, patient approach
- **Tutor Style**: Educational, explanatory tone
- **Engine Style**: Concise, move-focused responses

## Future Architecture Considerations

### Phase 2: Data & Quality
- **Dataset Overhaul**: High-quality chess Q&A curation
- **Enhanced Evaluation**: Comprehensive benchmarking suite
- **Training Improvements**: Advanced fine-tuning strategies

### Phase 3: Advanced Features
- **Embedding System**: Similar position retrieval and context enhancement
- **Vision Module**: Board image to FEN conversion
- **Enhanced Analysis**: Blunder detection, tactical motif recognition
- **Style Conditioning**: Historical player style emulation

### Phase 4: Polish & Deployment
- **Multi-Model Support**: Different model sizes and variants
- **Mobile Integration**: Core ML deployment for iOS devices
- **Performance Optimization**: Quantization and speed improvements
- **Research Integration**: Academic collaboration and publication
