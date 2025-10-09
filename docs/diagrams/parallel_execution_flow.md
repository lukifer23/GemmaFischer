# Parallel Multi-Expert Execution Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Chess Analysis Request                        │
│                                                                 │
│  Question: "What is the best move for white?"                   │
│  Context: "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w" │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│              MoE Router (Optional Intelligent Selection)        │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Route Query Analysis                                   │    │
│  │                                                         │    │
│  │ • Position complexity assessment                       │    │
│  │ • Query intent classification                          │    │
│  │ • Expert confidence prediction                         │    │
│  │                                                         │    │
│  │ Output: [UCI, TUTOR, DIRECTOR]                         │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│              Parallel Expert Execution                          │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │   UCI       │  │   TUTOR     │  │  DIRECTOR   │              │
│  │   Expert    │  │   Expert    │  │   Expert    │              │
│  │             │  │             │  │             │              │
│  │ Thread 1    │  │ Thread 2    │  │ Thread 3    │              │
│  │             │  │             │  │             │              │
│  │ • Load UCI  │  │ • Load      │  │ • Load      │              │
│  │   Adapter   │  │   Tutor     │  │   Director  │              │
│  │ • Generate  │  │   Adapter   │  │   Adapter   │              │
│  │   Move      │  │ • Generate  │  │ • Generate  │              │
│  │ • 1.2s      │  │   Analysis  │  │   Strategy  │              │
│  │             │  │ • 2.1s      │  │ • 1.8s      │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└─────────────────────┼───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│              Response Aggregation & Formatting                   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Collect Results                                           │    │
│  │                                                         │    │
│  │ • UCI: "e2e4" (1.2s)                                   │    │
│  │ • TUTOR: "Detailed explanation..." (2.1s)              │    │
│  │ • DIRECTOR: "Strategic analysis..." (1.8s)             │    │
│  │                                                         │    │
│  │ Total Time: 2.1s (not 6.3s!)                           │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Structured JSON Response                       │
│                                                                 │
│  {                                                              │
│    "question": "...",                                           │
│    "experts": ["uci", "tutor", "director"],                     │
│    "total_time": 2.1,                                           │
│    "results": {                                                 │
│      "uci": {"response": "e2e4", "confidence": 0.92},           │
│      "tutor": {"response": "...", "confidence": 0.88},          │
│      "director": {"response": "...", "confidence": 0.85}        │
│    }                                                            │
│  }                                                              │
└─────────────────────────────────────────────────────────────────┘
```

## Thread Synchronization Details

```
┌─────────────────────────────────────────────────────────────────┐
│                Thread Safety Mechanisms                         │
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐   │
│  │   Thread 1      │  │   Thread 2      │  │   Thread 3      │   │
│  │   (UCI)         │  │   (TUTOR)       │  │   (DIRECTOR)    │   │
│  │                 │  │                 │  │                 │   │
│  │ 1. Acquire      │  │ 1. Wait for     │  │ 1. Wait for     │   │
│  │    adapter lock │  │    adapter lock │  │    adapter lock │   │
│  │                 │  │                 │  │                 │   │
│  │ 2. Load UCI     │  │ 2. Load TUTOR   │  │ 2. Load DIR     │   │
│  │    adapter      │  │    adapter      │  │    adapter      │   │
│  │                 │  │                 │  │                 │   │
│  │ 3. Generate     │  │ 3. Generate     │  │ 3. Generate     │   │
│  │    response     │  │    response     │  │    response     │   │
│  │                 │  │                 │  │                 │   │
│  │ 4. Release lock │  │ 4. Release lock │  │ 4. Release lock │   │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘   │
│                                                                 │
│  Shared Resources:                                               │
│  • Base Gemma Model (read-only after loading)                   │
│  • Adapter Registry (thread-safe dictionary)                    │
│  • Response Cache (RLock protected)                             │
│  • MPS Device Context (managed by PyTorch)                      │
└─────────────────────────────────────────────────────────────────┘
```

## Memory Management Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                Memory Layout & Management                        │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Base Model Memory (4GB)                                 │    │
│  │ • Gemma-3 270M weights                                   │    │
│  │ • Tokenizer                                               │    │
│  │ • KV cache templates                                      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Adapter Memory (2-6GB peak)                              │    │
│  │                                                         │    │
│  │ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │    │
│  │ │ UCI Adapter │ │Tutor Adptr │ │Dir Adapter │          │    │
│  │ │   ~2GB      │ │   ~2GB      │ │   ~2GB      │          │    │
│  │ │ Loaded on   │ │ Loaded on   │ │ Loaded on   │          │    │
│  │ │   demand    │ │   demand    │ │   demand    │          │    │
│  │ └─────────────┘ └─────────────┘ └─────────────┘          │    │
│  │                                                         │    │
│  │ Peak during parallel: 4GB + 6GB = 10GB                 │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Runtime Memory (~2GB)                                   │    │
│  │ • Request processing                                     │    │
│  │ • Response caching                                        │    │
│  │ • Thread overhead                                         │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## Performance Optimization Flow

```
┌─────────────────────────────────────────────────────────────────┐
│            Multi-Level Caching Strategy                         │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Level 1: Response Cache (Fastest)                       │    │
│  │ • Identical queries → Instant response                   │    │
│  │ • Hash: question + context + expert + params            │    │
│  │ • Hit Rate: 70-85%                                       │    │
│  └─────────────────────────────────────────────────────────┘    │
│                               │                                   │
│  ┌─────────────────────────────▼─────────────────────────────┐   │
│  │ Level 2: Feature Cache (Position Analysis)                │   │
│  │ • Board position features                                  │    │
│  │ • King safety scores                                       │    │
│  │ • Piece activity metrics                                   │    │
│  │ • Hit Rate: 80-90%                                         │    │
│  └─────────────────────────────────────────────────────────────┘   │
│                               │                                   │
│  ┌─────────────────────────────▼─────────────────────────────┐   │
│  │ Level 3: Adapter Cache (Model Weights)                    │   │
│  │ • LoRA adapter weights                                     │    │
│  │ • Persistent across requests                               │    │
│  │ • Lazy loading on first use                                │    │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Error Handling & Recovery

```
┌─────────────────────────────────────────────────────────────────┐
│                Error Isolation Architecture                      │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Individual Expert Failure Handling                      │    │
│  │                                                         │    │
│  │ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │    │
│  │ │ UCI Expert  │ │Tutor Expert │ │Dir Expert  │          │    │
│  │ │             │ │             │ │             │          │    │
│  │ │ Success    │ │ Failed     │ │ Success    │          │    │
│  │ │ 1.2s        │ │ Timeout     │ │ 1.8s        │          │    │
│  │ │ e2e4        │ │             │ │ Analysis    │          │    │
│  │ └─────────────┘ └─────────────┘ └─────────────┘          │    │
│  │                                                         │    │
│  │ Response includes successful experts + error for failed │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ System-Level Recovery                                     │    │
│  │                                                         │    │
│  │ • Timeout Protection (30s per expert)                   │    │
│  │ • Memory Limit Enforcement                               │    │
│  │ • Graceful Degradation                                   │    │
│  │ • Request Logging & Monitoring                           │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## Web API Integration

```
┌─────────────────────────────────────────────────────────────────┐
│                Web API Request Flow                             │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ POST /api/ask_parallel                                   │    │
│  │                                                         │    │
│  │ {                                                       │    │
│  │   "question": "What is the best move?",                 │    │
│  │   "context": "fen_string",                              │    │
│  │   "experts": ["uci", "tutor", "director"]              │    │
│  │ }                                                       │    │
│  └─────────────────────────────────────────────────────────┘    │
│                               │                                   │
│  ┌─────────────────────────────▼─────────────────────────────┐   │
│  │ Flask Request Processing                                 │   │
│  │ • Input validation                                        │    │
│  │ • RAG context enhancement                                 │    │
│  │ • Expert selection                                        │    │
│  └─────────────────────────────────────────────────────────────┘   │
│                               │                                   │
│  ┌─────────────────────────────▼─────────────────────────────┐   │
│  │ Parallel Inference Execution                             │   │
│  │ • Thread pool creation                                    │    │
│  │ • Expert distribution                                     │    │
│  │ • Synchronization                                          │    │
│  └─────────────────────────────────────────────────────────────┘   │
│                               │                                   │
│  ┌─────────────────────────────▼─────────────────────────────┐   │
│  │ Response Aggregation                                      │   │
│  │ • Result collection                                        │    │
│  │ • Performance metrics                                     │    │
│  │ • Error handling                                          │    │
│  └─────────────────────────────────────────────────────────────┘   │
│                               │                                   │
│  ┌─────────────────────────────▼─────────────────────────────┐   │
│  │ JSON Response Formatting                                 │   │
│  │ • Structured output                                       │    │
│  │ • Timing information                                      │    │
│  │ • Expert results                                          │    │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                Production Deployment                            │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Flask Web Server (localhost:5000)                       │    │
│  │ • Gunicorn/Waitress for production                       │    │
│  │ • CORS enabled for web clients                           │    │
│  │ • Request logging and monitoring                         │    │
│  └─────────────────────────────────────────────────────────┘    │
│                               │                                   │
│  ┌─────────────────────────────▼─────────────────────────────┐   │
│  │ Chess Model Service                                       │    │
│  │ • Singleton inference instance                            │    │
│  │ • Lazy model loading                                      │    │
│  │ • Adapter management                                      │    │
│  └─────────────────────────────────────────────────────────────┘   │
│                               │                                   │
│  ┌─────────────────────────────▼─────────────────────────────┐   │
│  │ MPS Acceleration Layer                                    │    │
│  │ • Apple Silicon optimized                                  │    │
│  │ • Memory management                                       │    │
│  │ • Performance monitoring                                  │    │
│  └─────────────────────────────────────────────────────────────┘   │
│                               │                                   │
│  ┌─────────────────────────────▼─────────────────────────────┐   │
│  │ Storage Layer                                             │    │
│  │ • Checkpoint management                                   │    │
│  │ • Cache persistence                                       │    │
│  │ • Log aggregation                                         │    │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

This architecture enables efficient parallel multi-expert execution while maintaining thread safety, proper resource management, and excellent performance on Apple Silicon hardware. The design scales from single expert queries to full parallel analysis with minimal overhead.
