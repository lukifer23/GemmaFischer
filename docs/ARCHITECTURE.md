# ChessGemma Architecture

## System Overview

ChessGemma is a comprehensive chess AI system featuring a Mixture of Experts (MoE) architecture that functions as both a chess engine (UCI-compatible) and a chess tutor/analyst. Built around fine-tuned Gemma-3 270M models with specialized LoRA adapters and intelligent routing, it provides tactical analysis, educational explanations, and dynamic expert selection. The architecture follows a clear separation of concerns with distinct layers for training, inference, evaluation, and presentation.

**Platform**: Mac-only (M3 Pro/Max) with MPS acceleration - optimized exclusively for Apple Silicon.

### Key Design Principles
- **Mixture of Experts (MoE)**: Dynamic routing between specialized expert models
- **Multi-Modal Operation**: Engine (UCI moves), Tutor (explanations), and Director (Q&A) modes
- **Hybrid Intelligence**: Combines LLM reasoning with chess engine precision and validation
- **Advanced Evaluation**: ELO estimation, move quality scoring, and comprehensive metrics
- **Data Standardization**: Automated validation and quality assurance pipeline
- **Comprehensive Logging**: Structured logging with performance monitoring and error tracking
- **Checkpoint Management**: Robust training resume with progress tracking and validation
- **MPS Memory Optimization**: Dynamic batch sizing and memory-efficient training
- **Position Embeddings**: Chess-aware similarity search and context enhancement

## High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Interface │    │ Training System │    │   MoE Router    │    │ Evaluation Suite│
│ (Flask + MoE UI)│    │  (LoRA Experts) │    │ (Expert Routing) │    │ (ELO + Quality)│
│   http://localhost│    │                 │    │                 │    │                 │
│       :5001      │    │                 │    │                 │    │                 │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │                      │
          └──────────────────────┼──────────────────────┼──────────────────────┘
                                 │                      │
                    ┌─────────────┴─────────────┐       │
                    │   MoE Inference Manager  │       │
                    │ (Dynamic Expert Selection│       │
                    │    + Ensemble Mode)      │       │
                    └─────────────┬─────────────┘       │
                                 │                      │
                    ┌─────────────┴─────────────┐       │
                    │   Specialized Experts     │       │
                    │ (UCI/Tutor/Director LoRA) │       │
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
                                              │ Data Standard-   │
                                              │  ization Pipeline│
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

### 2. Mixture of Experts (MoE) Inference Layer (`src/inference/`)

**Purpose**: Dynamic expert routing and intelligent adapter switching based on position characteristics and query analysis

**Key Components**:
- `inference.py`: Enhanced inference engine with MoE integration
- `moe_router.py`: Chess-specific MoE router with position feature extraction
- `chess_engine.py`: Stockfish integration for validation and analysis
- `uci_utils.py`: UCI move extraction and validation utilities

**MoE Architecture**:
```
MoE Inference Flow
├── Query Analysis
│   ├── Position Feature Extraction
│   │   ├── Material balance
│   │   ├── King safety
│   │   ├── Piece activity
│   │   ├── Pawn structure
│   │   └── Tactical motifs
│   └── Query Type Classification
│       ├── UCI move requests
│       ├── Explanatory queries
│       └── General Q&A
├── Expert Selection
│   ├── Primary Expert Routing
│   ├── Ensemble Mode Detection
│   └── Confidence Scoring
├── Inference Execution
│   ├── Adapter Loading/Caching
│   ├── Response Generation
│   └── Quality Validation
└── Response Enhancement
    ├── Routing Metadata
    ├── Confidence Scores
    └── Expert Attribution
```
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

### 5. Advanced Evaluation Layer (`src/evaluation/`)

**Purpose**: Comprehensive evaluation suite with ELO estimation, move quality scoring, and performance benchmarking

**Key Components**:
- `advanced_chess_eval.py`: ELO estimation and move quality analysis system
- `stockfish_match_eval.py`: Stockfish vs model tournament evaluation
- `puzzle_eval.py`: Tactical puzzle accuracy testing with quality metrics
- `chess_evaluation.py`: General evaluation framework
- `comprehensive_eval.py`: Multi-dimensional performance assessment

**Architecture**:
```
Advanced Evaluation Pipeline
├── ELO Rating Estimation
│   ├── Tournament Simulation (vs Stockfish)
│   ├── Performance Rating Calculation
│   ├── Confidence Interval Analysis
│   └── Rating Category Assessment
├── Move Quality Analysis
│   ├── Centipawn Loss Calculation
│   ├── Move Categorization (best/excellent/good/blunder)
│   ├── Stockfish Validation
│   └── Quality Distribution Analysis
├── Position Evaluation
│   ├── Static Position Assessment
│   ├── Evaluation Accuracy Metrics
│   ├── Phase-specific Analysis
│   └── Error Distribution Analysis
├── Comprehensive Benchmarking
│   ├── Multi-dataset Evaluation
│   ├── Cross-validation Testing
│   ├── Performance Regression Detection
│   └── Comparative Analysis
└── Reporting & Visualization
    ├── Structured JSON Reports
    ├── Performance Dashboards
    ├── Trend Analysis
    └── Automated Benchmarking
```

**Key Features**:
- **ELO Estimation**: Tournament-based rating calculation with confidence intervals
- **Move Quality Scoring**: Centipawn loss analysis and move categorization
- **Position Evaluation**: Stockfish-verified position assessment accuracy
- **Comprehensive Metrics**: Win rates, accuracy scores, quality distributions
- **Automated Benchmarking**: Continuous evaluation and performance tracking
- **Advanced Error Analysis**: Detailed classification of model mistakes and failure modes

### 6. Checkpoint Management System (`src/training/checkpoint_manager.py`)

**Purpose**: Robust training resume functionality with comprehensive progress tracking and validation

**Key Components**:
- `CheckpointManager`: Core checkpoint management with metadata tracking
- `CheckpointMetadata`: Structured metadata for training state
- `TrainingProgress`: Progress tracking across training sessions
- Custom training callbacks for automatic checkpointing

**Architecture**:
```
Checkpoint Management System
├── Checkpoint Discovery & Loading
│   ├── Latest Checkpoint Detection
│   ├── Metadata Validation
│   ├── Resume State Reconstruction
│   └── Integrity Verification
├── Automatic Checkpoint Creation
│   ├── Training Progress Capture
│   ├── Loss Metrics & Learning Rates
│   ├── Dataset & Model Metadata
│   └── System Information Logging
├── Progress Tracking & Reporting
│   ├── Training Statistics
│   ├── Estimated Completion Time
│   ├── Best Model Identification
│   └── Performance Trend Analysis
├── Checkpoint Maintenance
│   ├── Automatic Cleanup (Old Checkpoints)
│   ├── Storage Optimization
│   ├── Backup & Recovery
│   └── Version Management
└── Training Resume Functionality
    ├── State Restoration
    ├── Optimizer State Loading
    ├── DataLoader Resumption
    └── Progress Continuation
```

**Key Features**:
- **Robust Resume**: Automatic detection and resumption from latest checkpoints
- **Metadata Tracking**: Comprehensive training state with loss curves and metrics
- **Progress Monitoring**: Real-time training progress with completion estimates
- **Integrity Validation**: Checkpoint corruption detection and recovery
- **Smart Cleanup**: Automatic removal of old checkpoints while preserving best models
- **Multi-Expert Support**: Separate checkpoint management for each expert type

### 7. MPS Memory Optimization (`src/training/mps_optimizer.py`)

**Purpose**: Dynamic memory management and performance optimization for Apple Silicon MPS training

**Key Components**:
- `MPSMemoryOptimizer`: Core memory optimization with dynamic batch sizing
- `MPSDataLoaderOptimizer`: DataLoader optimizations for MPS efficiency
- Memory profiling and monitoring utilities
- Training configuration optimization

**Architecture**:
```
MPS Memory Optimization Pipeline
├── System Memory Assessment
│   ├── Available Memory Detection
│   ├── MPS Capability Verification
│   ├── Memory Usage Monitoring
│   └── Resource Allocation Planning
├── Dynamic Batch Size Calculation
│   ├── Model Memory Profiling
│   ├── Optimal Batch Size Determination
│   ├── Gradient Accumulation Adjustment
│   └── Memory Utilization Optimization
├── MPS-Specific Optimizations
│   ├── fp16 Training Enablement
│   ├── Gradient Checkpointing
│   ├── Memory-Efficient Attention
│   ├── Learning Rate Adjustments
│   └── Optimizer Selection
├── Performance Monitoring
│   ├── Real-time Memory Tracking
│   ├── Training Speed Optimization
│   ├── Bottleneck Identification
│   └── Automatic Adjustment
└── Memory Management Utilities
    ├── Cache Clearing Automation
    ├── Memory Leak Prevention
    ├── Efficient Data Loading
    └── Resource Cleanup
```

**Key Features**:
- **Dynamic Batch Sizing**: Automatic batch size calculation based on available memory
- **Memory Profiling**: Real-time memory usage monitoring and optimization
- **MPS Optimization**: Native Apple Silicon performance tuning
- **Gradient Checkpointing**: Memory-efficient training with reduced VRAM usage
- **Performance Monitoring**: Training speed and memory utilization tracking
- **Automatic Adjustment**: Self-tuning parameters based on system capabilities

### 8. Chess Position Embeddings (`src/inference/chess_embeddings.py`)

**Purpose**: Chess-aware position embeddings and similarity search for enhanced context and analysis

**Key Components**:
- `ChessPositionEmbedder`: Domain-specific embedding creation
- `ChessPositionRetriever`: Efficient similarity search and retrieval
- Position metadata extraction and analysis
- Retrieval-augmented generation support

**Architecture**:
```
Chess Position Embedding System
├── Feature Extraction Pipeline
│   ├── Material Balance Analysis
│   │   ├── Piece Values & Distribution
│   │   ├── Material Imbalance Calculation
│   │   └── Positional Material Value
│   ├── King Safety Assessment
│   │   ├── King Zone Attack Analysis
│   │   ├── Open File Detection
│   │   └── Castling Rights Evaluation
│   ├── Pawn Structure Analysis
│   │   ├── Pawn Islands & Chains
│   │   ├── Doubled Pawns Detection
│   │   └── Pawn Weaknesses
│   ├── Piece Mobility Calculation
│   │   ├── Legal Moves Counting
│   │   ├── Piece-Specific Activity
│   │   └── Development Assessment
│   ├── Tactical Motif Detection
│   │   ├── Pin Identification
│   │   ├── Check Patterns
│   │   └── Hanging Piece Analysis
│   └── Positional Feature Extraction
│       ├── Center Control
│       ├── Space Advantage
│       └── Coordination Metrics
├── Vector Embedding Creation
│   ├── Feature Normalization
│   ├── Dimensionality Reduction
│   ├── Unit Vector Normalization
│   └── Embedding Optimization
├── Similarity Search Engine
│   ├── Cosine Distance Calculation
│   ├── Top-K Retrieval
│   ├── Common Feature Identification
│   └── Relevance Scoring
└── Context Enhancement
    ├── Similar Position Retrieval
    ├── Response Augmentation
    ├── Analysis Enrichment
    └── Learning Context Provision
```

**Key Features**:
- **Chess-Domain Embeddings**: Position-aware vector representations using chess-specific features
- **Similarity Search**: Efficient retrieval of related chess positions
- **Context Enhancement**: Retrieval-augmented generation for improved analysis
- **Position Clustering**: Pattern recognition and strategic motif identification
- **Metadata Integration**: Rich position information for enhanced search capabilities
- **Performance Optimized**: Fast similarity calculations with pre-computed embeddings

### 9. Data Standardization Pipeline (`data/`)

**Purpose**: Automated dataset validation, quality assurance, and format standardization across 977K+ training samples

**Key Components**:
- `data/scripts/standardize_data_formats.py`: Comprehensive format standardization
- `data/scripts/validate_dataset_comprehensive.py`: Quality assessment and validation
- `data/standardized/`: Standardized datasets in consistent expert format
- `validation/`: Quality assessment reports and metrics

**Architecture**:
```
Data Standardization Pipeline
├── Multi-Format Input Processing
│   ├── Expert Format (task/prompt/response/meta)
│   ├── Legacy Text Format (conversational)
│   ├── Raw FEN Format (puzzle datasets)
│   └── Format Detection & Conversion
├── Comprehensive Validation
│   ├── Move Legality Verification
│   ├── FEN Position Validation
│   ├── Response Format Checking
│   └── Quality Score Assignment
├── Quality Assurance
│   ├── Position Complexity Analysis
│   ├── Move Type Distribution
│   ├── Metadata Completeness
│   └── Statistical Quality Metrics
├── Standardization & Cleaning
│   ├── Consistent JSON Schema
│   ├── Response Format Normalization
│   ├── Metadata Enrichment
│   └── Duplicate Removal
├── Expert-Specific Datasets
│   ├── UCI Expert (977K+ move generation samples)
│   ├── Tutor Expert (500K+ explanation samples)
│   ├── Director Expert (5.1K+ Q&A samples)
│   └── Quality-Filtered Outputs
└── Quality Control & Reporting
    ├── Automated Validation Reports
    ├── Quality Distribution Analysis
    ├── Error Classification
    └── Performance Benchmarking
```

**Dataset Features**:
- **977K+ Standardized Samples**: Multi-format datasets converted to consistent expert schema
- **Quality Assurance**: 93.1% average quality score with comprehensive validation
- **Expert Specialization**: Separate high-quality datasets for UCI, Tutor, and Director modes
- **Automated Processing**: Batch standardization and validation across all input formats
- **Stockfish Validation**: All moves verified for legality and position validity
- **Quality Metrics**: Position complexity analysis, move type distribution, metadata completeness

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
