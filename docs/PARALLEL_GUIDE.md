# Parallel Multi-Expert Analysis Guide

## Overview

GemmaFischer's parallel multi-expert execution allows you to query all three specialized experts (UCI, Tutor, Director) simultaneously, providing comprehensive chess analysis from multiple perspectives in a single request.

## Quick Start

### Web Interface
```bash
# Start the web server
./run_hybrid_webapp.sh
# Visit: http://localhost:5000

# Use the parallel endpoint instead of the regular /api/ask
curl -X POST http://localhost:5000/api/ask_parallel \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the best move for white?", "context": "fen_here"}'
```

### Python API
```python
from src.inference.inference import run_parallel_inference

# Get comprehensive analysis
results = run_parallel_inference(
    question="What is the best move for white?",
    context="r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
)

# Access individual expert responses
uci_move = results['uci']['response']
tutor_explanation = results['tutor']['response']
director_analysis = results['director']['response']
```

## Understanding Expert Roles

### UCI Expert
- **Purpose**: Chess move generation and tactical analysis
- **Output**: Raw UCI moves (e.g., "e2e4", "Nf3")
- **Strengths**: Fast, tactical accuracy, engine-like responses
- **Use Case**: "What move should I play?"

### Tutor Expert
- **Purpose**: Educational explanations and teaching
- **Output**: Step-by-step reasoning with chess concepts
- **Strengths**: Clear explanations, learning-focused, mistake analysis
- **Use Case**: "Why is this move good/bad?"

### Director Expert
- **Purpose**: Strategic analysis and high-level concepts
- **Output**: Opening theory, endgame principles, positional evaluation
- **Strengths**: Big-picture thinking, long-term planning
- **Use Case**: "What's the strategic idea here?"

## Use Cases

### 1. Position Evaluation
```bash
# Get complete position analysis
curl -X POST http://localhost:5000/api/ask_parallel \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Evaluate this position",
    "context": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 4"
  }'
```

**Expected Output:**
- **UCI**: Specific move recommendation (e.g., "d3d4")
- **Tutor**: "White should play d3-d4 to challenge Black's center control..."
- **Director**: "This is a closed center. White should focus on development and king safety..."

### 2. Learning Analysis
```bash
# Understand a tactical pattern
curl -X POST http://localhost:5000/api/ask_parallel \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Explain this tactical motif",
    "context": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 4"
  }'
```

### 3. Cross-Validation
```bash
# Check expert consistency
curl -X POST http://localhost:5000/api/ask_parallel \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Is this a good move?",
    "context": "position_after_move"
  }'
```

## Response Structure

```json
{
  "question": "What is the best move?",
  "context": "fen_string",
  "experts": ["uci", "tutor", "director"],
  "total_time": 3.45,
  "results": {
    "uci": {
      "response": "e2e4",
      "confidence": 0.92,
      "generation_time": 1.2,
      "model_loaded": true,
      "mode": "engine",
      "cached": false,
      "cache_hit_rate": 0.0
    },
    "tutor": {
      "response": "The move e2-e4 controls the center and develops the king pawn...",
      "confidence": 0.88,
      "generation_time": 2.1,
      "model_loaded": true,
      "mode": "tutor",
      "cached": false,
      "cache_hit_rate": 0.0
    },
    "director": {
      "response": "This is a standard opening move that follows classical principles...",
      "confidence": 0.85,
      "generation_time": 1.8,
      "model_loaded": true,
      "mode": "director",
      "cached": false,
      "cache_hit_rate": 0.0
    }
  }
}
```

## Performance Characteristics

### Timing
- **Sequential**: ~2.5 seconds per expert query
- **Parallel**: ~3.2 seconds for all three experts
- **Overhead**: ~1.3x response time for 3x richer analysis

### Memory Usage
- **Base Model**: ~4GB
- **Single Expert**: +2GB adapter
- **All Experts**: ~8GB peak (parallel loading)

### Concurrent Requests
- **Thread Safe**: Multiple parallel queries can run simultaneously
- **Isolation**: Each request maintains separate expert contexts
- **Resource Pooling**: Efficient adapter reuse across requests

## Advanced Usage

### Custom Expert Selection
```python
# Query only specific experts
results = run_parallel_inference(
    question="Analyze this opening",
    experts=['tutor', 'director']  # Skip UCI for strategic focus
)
```

### Error Handling
```python
# Parallel execution handles individual expert failures gracefully
results = run_parallel_inference(question="Complex analysis")

# Check for errors in individual responses
for expert, response in results.items():
    if 'error' in response:
        print(f"{expert} failed: {response['error']}")
    else:
        print(f"{expert}: {response['response'][:100]}...")
```

### Performance Monitoring
```python
results = run_parallel_inference(question="Performance test")

# Analyze response times
total_time = results.get('total_time', 0)
expert_times = {}
for expert, response in results.get('results', {}).items():
    expert_times[expert] = response.get('generation_time', 0)

print(f"Total parallel time: {total_time:.2f}s")
print(f"Expert timing: {expert_times}")
```

## Troubleshooting

### Common Issues

**Slow Response Times:**
- Check if all experts are trained and available
- Verify model is loaded and adapters are cached
- Consider reducing expert count for faster responses

**Memory Issues:**
- Parallel execution requires more memory than single expert
- Close other memory-intensive applications
- Use fewer experts or sequential execution if needed

**Inconsistent Results:**
- Experts may legitimately disagree on complex positions
- This is often educational - compare reasoning
- Use cross-validation to identify strong consensus

### Expert Availability

```bash
# Check which experts are available
python scripts/moe_health_check.py

# Train missing experts
python -m src.training.train_lora_poc --expert tutor --config auto --max_steps_override 1000
```

## Best Practices

### When to Use Parallel Analysis
- **Learning**: Understand positions from multiple perspectives
- **Validation**: Cross-check expert recommendations
- **Research**: Compare different analytical approaches
- **Debugging**: Identify expert inconsistencies or failures

### When to Use Single Expert
- **Speed Critical**: Tournament play or time-constrained analysis
- **Specific Focus**: Only need tactical moves or explanations
- **Resource Limited**: Memory or processing constraints

### Optimizing Performance
- **Cache Warming**: Run initial queries to load adapters
- **Batch Processing**: Group similar queries together
- **Selective Experts**: Use only needed experts for specific tasks
- **Concurrent Limits**: Monitor system resources during heavy usage

## Integration Examples

### Web Application
```javascript
// Frontend integration
async function analyzePosition(fen, question) {
    const response = await fetch('/api/ask_parallel', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, context: fen })
    });

    const data = await response.json();

    // Display results
    displayExpertResponse('uci', data.results.uci);
    displayExpertResponse('tutor', data.results.tutor);
    displayExpertResponse('director', data.results.director);

    // Show timing
    showPerformanceMetrics(data.total_time, data.results);
}
```

### Chess Engine Integration
```python
# UCI bridge with parallel analysis
class EnhancedUCIBridge:
    def analyze_position(self, fen):
        # Get parallel analysis
        results = run_parallel_inference(
            question="Analyze this position",
            context=fen
        )

        # Use UCI expert for move, others for explanation
        move = results['uci']['response']
        explanation = results['tutor']['response']

        return move, explanation
```

## Future Enhancements

### Planned Features
- **Ensemble Responses**: Combine expert outputs intelligently
- **Confidence Weighting**: Prioritize higher-confidence expert responses
- **Interactive Mode**: Real-time expert switching during analysis
- **Custom Expert Training**: User-defined expert specializations

### Research Directions
- **Expert Agreement Metrics**: Quantify expert consensus
- **Adaptive Routing**: Learn which experts to prefer for different query types
- **Multi-turn Conversations**: Maintain expert context across dialogue
- **Visual Analysis**: Expert commentary on board positions

---

Ready to try parallel multi-expert analysis? Start with the web interface at http://localhost:5000 and use /api/ask_parallel for comprehensive chess insights!
