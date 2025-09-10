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
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Interface │    │  Training Layer │    │ Evaluation Layer│
│   (Flask App)   │    │   (LoRA Fine-   │    │  (Chess Metrics)│
│                 │    │    tuning)      │    │                 │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────┴─────────────┐
                    │     Core Inference        │
                    │   (Gemma-3 + LoRA)        │
                    │  + Chain-of-Thought       │
                    └─────────────┬─────────────┘
                                 │
                    ┌─────────────┴─────────────┐
                    │    UCI Bridge Layer       │
                    │  (Engine + Tutor Modes)   │
                    └─────────────┬─────────────┘
                                 │
                    ┌─────────────┴─────────────┐
                    │    Chess Engine Layer     │
                    │     (Stockfish)           │
                    └───────────────────────────┘
```

## Component Details

### 1. Training Layer (`src/training/`)

**Purpose**: Fine-tune Gemma-3 model with chess-specific knowledge using LoRA, including chain-of-thought reasoning and dual-mode training

**Key Components**:
- `train.py`: Main training orchestrator
- `train_lora_poc.py`: Proof-of-concept training script
- `configs/`: Training configuration files
- `dataset_mixer.py`: Weighted dataset interleaving (multi-task)

**Architecture**:
```
Training Pipeline
├── Data Loading (ChessInstruct + Historical Games + Theory)
├── Data Preprocessing (Chat format + CoT reasoning)
├── Model Loading (Gemma-3 base model)
├── LoRA Adapter Attachment
├── Dual-Mode Training (Engine + Tutor modes)
├── Chain-of-Thought Integration
├── Checkpoint Management
└── Model Saving
```

**New Training Features**:
- **Dual-Purpose Training**: Separate modes for engine (UCI) and tutor (explanatory) outputs
- **Chain-of-Thought**: Step-by-step reasoning integration
- **Style Conditioning**: Historical player style emulation
- **Enhanced Datasets**: Historical games, theory books, annotated PGNs
 - **Curriculum Phases**: Optional phased schedule with per-phase dataset mixes

**Key Features**:
- Unsloth optimization for 2x speed improvement on M3 Pro
- MPS acceleration exclusively (no CUDA/CPU fallbacks)
- Gradient checkpointing for memory efficiency
- Resume functionality for long training runs
- Mac-only design optimized for Apple Silicon

### 2. UCI Bridge Layer (`src/inference/`)

**Purpose**: Provide UCI-compatible chess engine interface and dual-mode operation

**Key Components**:
- `inference.py`: Main inference script
- `chess_engine.py`: Stockfish integration
- `uci_bridge.py`: UCI protocol implementation (NEW)
- `prompt_templates/`: Mode-specific prompt templates (NEW)

**Architecture**:
```
UCI Bridge Pipeline
├── UCI Protocol Handler
├── Mode Selection (Engine vs Tutor)
├── Position Analysis (FEN → Prompt)
├── Model Inference (Gemma-3 + LoRA)
├── Response Processing
├── UCI Output Formatting
└── Fallback to Stockfish (if needed)
```

**New UCI Features**:
- **UCI Compatibility**: Full UCI protocol support for chess software integration
- **Dual Modes**: Engine mode (fast moves) and Tutor mode (explanations)
- **Chain-of-Thought**: Step-by-step reasoning in tutor mode
- **Style Conditioning**: Historical player style emulation
- **Strict UCI Output**: Engine-mode postprocessing ensures a single legal UCI move where possible
- **Fallback Support**: Stockfish integration when parsing/legality fails

### 3. Inference Layer (`src/inference/`)

**Purpose**: Generate chess-related responses using the fine-tuned model with enhanced reasoning

**Key Components**:
- `inference.py`: Main inference engine
- `chess_engine.py`: Stockfish integration

**Architecture**:
```
Inference Pipeline
├── Model Loading (Base + LoRA adapter)
├── Prompt Processing (Chat template + CoT)
├── Text Generation (Gemma-3 + Reasoning)
├── Response Post-processing
└── Chess Validation (Stockfish)
```

**Key Features**:
- Model caching for performance
- Chess-specific prompt formatting with chain-of-thought
- Move validation and analysis
- Error handling and fallbacks
- Dual-mode operation (engine/tutor)

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

### 4. Web Interface (`src/web/`)

**Purpose**: Provide interactive web interface for chess Q&A

**Key Components**:
- `app.py`: Flask application
- `templates/`: HTML templates

**Architecture**:
```
Web Interface
├── Flask Application
├── API Endpoints (/api/ask, /api/health)
├── Model Integration
├── Response Caching
└── Error Handling
```

**Key Features**:
- Real-time Q&A interface
- Model status monitoring
- Example question suggestions
- Response confidence scoring

### 5. Evaluation Layer (`src/evaluation/`)

**Purpose**: Assess model performance with chess-specific metrics

**Key Components**:
- `chess_evaluation.py`: Evaluation framework

**Architecture**:
```
Evaluation Pipeline
├── Test Question Loading
├── Model Response Generation
├── Chess Relevance Scoring
├── Move Syntax Validation
└── Performance Metrics
```

**Key Features**:
- Chess-specific evaluation metrics
- Move syntax validation
- Relevance scoring
- Comparative analysis

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
