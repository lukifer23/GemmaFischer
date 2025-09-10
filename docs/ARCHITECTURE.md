# ChessGemma Architecture

## System Overview

ChessGemma is a modular chess AI system built around fine-tuned language models, chess engine integration, and web interfaces. The architecture follows a clear separation of concerns with distinct layers for training, inference, evaluation, and presentation.

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
                    └─────────────┬─────────────┘
                                 │
                    ┌─────────────┴─────────────┐
                    │    Chess Engine Layer     │
                    │     (Stockfish)           │
                    └───────────────────────────┘
```

## Component Details

### 1. Training Layer (`src/training/`)

**Purpose**: Fine-tune Gemma-3 model with chess-specific knowledge using LoRA

**Key Components**:
- `train.py`: Main training orchestrator
- `train_lora_poc.py`: Proof-of-concept training script
- `configs/`: Training configuration files

**Architecture**:
```
Training Pipeline
├── Data Loading (ChessInstruct dataset)
├── Data Preprocessing (Chat format conversion)
├── Model Loading (Gemma-3 base model)
├── LoRA Adapter Attachment
├── Training Loop (SFTTrainer)
├── Checkpoint Management
└── Model Saving
```

**Key Features**:
- Unsloth optimization for 2x speed improvement
- MPS acceleration for Apple Silicon
- Gradient checkpointing for memory efficiency
- Resume functionality for long training runs

### 2. Inference Layer (`src/inference/`)

**Purpose**: Generate chess-related responses using the fine-tuned model

**Key Components**:
- `inference.py`: Main inference engine
- `chess_engine.py`: Stockfish integration

**Architecture**:
```
Inference Pipeline
├── Model Loading (Base + LoRA adapter)
├── Prompt Processing (Chat template)
├── Text Generation (Gemma-3)
├── Response Post-processing
└── Chess Validation (Stockfish)
```

**Key Features**:
- Model caching for performance
- Chess-specific prompt formatting
- Move validation and analysis
- Error handling and fallbacks

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

## Future Architecture Considerations

### Scalability Improvements
- Microservices architecture
- Container orchestration
- Database integration
- Caching layers

### Feature Enhancements
- Multi-model support
- Advanced chess analysis
- Real-time collaboration
- Mobile applications

### Performance Optimizations
- Model quantization
- Inference optimization
- Caching strategies
- Load balancing
