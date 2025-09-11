# ChessGemma Evaluation Guide

## Overview

This guide covers the comprehensive evaluation framework for ChessGemma, including automated metrics, web-based evaluation tools, and performance benchmarking. The system supports real-time evaluation through the web interface and automated testing against Stockfish.

**Current Status**: Web-based evaluation tools active, Stockfish match testing available, baseline tactical accuracy ~2% (room for improvement).

## Current Evaluation Status

### Web-Based Evaluation Tools
The system provides comprehensive evaluation through the web interface:

```bash
# Start web interface for evaluation
python src/web/app.py
# Visit http://localhost:5001/evaluation
```

#### Available Evaluation Methods:
- **Stockfish Match Testing**: Compare model vs Stockfish on tactical positions
- **Puzzle Accuracy Evaluation**: Test on Lichess puzzle database (1000+ puzzles)
- **Real-time Results**: Live evaluation progress and performance metrics
- **Expert-Specific Testing**: Evaluate individual experts (UCI/Tutor/Director)

### Current Performance Metrics
Based on recent evaluation runs:

```bash
# Tactical Puzzle Evaluation Results
First Move Accuracy: 2% (baseline, room for improvement)
Legal Move Rate: 100% (Stockfish validation working)
Average Response Time: ~1 second per puzzle
Total Puzzles Tested: 100 (from Lichess 1000-2000 rating range)

# Stockfish Match Evaluation (Planned)
Win Rate: TBD (vs Stockfish depth 12)
Draw Rate: TBD
Loss Rate: TBD
```

### Evaluation Tools Status
```bash
Web Interface: ✓ Active (http://localhost:5001)
Stockfish Integration: ✓ Working (depth 12 evaluation)
Puzzle Database: ✓ Available (Lichess puzzles 1000-2000)
Expert Switching: ✓ Working (live adapter switching)
Real-time Monitoring: ✓ Active (live progress tracking)
```

## Automated Evaluation Metrics

### 1. Move Legality and Syntax

**Purpose**: Ensure the model never outputs illegal moves

**Metrics**:
- **Move Syntax Accuracy**: Percentage of moves in correct algebraic notation
- **Move Legality Rate**: Percentage of moves that are legal on the given board
- **Target**: 100% for both metrics

**Implementation**:
```python
def evaluate_move_legality(model, test_positions):
    """Evaluate move legality and syntax"""
    legal_moves = 0
    correct_syntax = 0
    total_moves = 0
    
    for position in test_positions:
        response = model.generate(position)
        moves = extract_moves_from_response(response)
        
        for move in moves:
            total_moves += 1
            if is_valid_notation(move):
                correct_syntax += 1
                if is_legal_move(move, position):
                    legal_moves += 1
    
    return {
        'syntax_accuracy': correct_syntax / total_moves,
        'legality_rate': legal_moves / total_moves
    }
```

### 2. Tactical Puzzle Success Rate

**Purpose**: Measure ability to solve chess puzzles

**Metrics**:
- **Puzzle Success Rate**: Percentage of puzzles solved correctly
- **First Move Accuracy**: Percentage where first move matches solution
- **Target**: 70%+ for basic puzzles, 50%+ for advanced

**Implementation**:
```python
def evaluate_tactical_puzzles(model, puzzle_dataset):
    """Evaluate puzzle solving ability"""
    correct_first_moves = 0
    correct_sequences = 0
    total_puzzles = len(puzzle_dataset)
    
    for puzzle in puzzle_dataset:
        position = puzzle['fen']
        solution = puzzle['solution']
        
        response = model.generate(position)
        predicted_moves = extract_moves_from_response(response)
        
        if predicted_moves and predicted_moves[0] == solution[0]:
            correct_first_moves += 1
        
        if predicted_moves == solution:
            correct_sequences += 1
    
    return {
        'first_move_accuracy': correct_first_moves / total_puzzles,
        'sequence_accuracy': correct_sequences / total_puzzles
    }
```

### 3. Positional Question Answering

**Purpose**: Test conceptual understanding of chess

**Metrics**:
- **Conceptual Accuracy**: Percentage of correct answers to chess concepts
- **Rule Knowledge**: Accuracy on basic chess rules
- **Target**: 90%+ for rules, 70%+ for concepts

**Test Questions**:
```python
CONCEPTUAL_QUESTIONS = [
    {
        "question": "Can you castle after moving the rook?",
        "answer": "No, once the rook moves, that rook cannot partake in castling",
        "category": "rules"
    },
    {
        "question": "What is the weakness of an isolated pawn?",
        "answer": "Cannot be defended by another pawn, becomes a target",
        "category": "strategy"
    },
    {
        "question": "What is the idea behind the Minority Attack?",
        "answer": "Advance fewer pawns to provoke weaknesses in opponent's structure",
        "category": "strategy"
    }
]
```

### 4. Opening Identification and Theory

**Purpose**: Test knowledge of chess openings

**Metrics**:
- **Opening Recognition**: Percentage of correctly identified openings
- **Theory Accuracy**: Correctness of opening plans and variations
- **Target**: 80%+ for common openings

**Implementation**:
```python
def evaluate_opening_knowledge(model, opening_dataset):
    """Evaluate opening theory knowledge"""
    correct_identifications = 0
    correct_plans = 0
    total_openings = len(opening_dataset)
    
    for opening in opening_dataset:
        moves = opening['moves']
        expected_name = opening['name']
        expected_plan = opening['plan']
        
        response = model.generate(f"Identify this opening: {moves}")
        
        if expected_name.lower() in response.lower():
            correct_identifications += 1
        
        if evaluate_plan_correctness(response, expected_plan):
            correct_plans += 1
    
    return {
        'identification_accuracy': correct_identifications / total_openings,
        'plan_accuracy': correct_plans / total_openings
    }
```

### 5. Stockfish Match Analysis

**Purpose**: Compare model moves to engine analysis with comprehensive match evaluation

**Metrics**:
- **Top Move Match**: Percentage matching Stockfish's #1 move
- **Top 3 Match**: Percentage in Stockfish's top 3 moves
- **Evaluation Accuracy**: How close model's evaluation is to engine's
- **Target**: 50%+ top move, 75%+ top 3 (current: baseline performance)

**Web Interface Tooling**:
```bash
# Start web interface for Stockfish evaluation
python src/web/app.py
# Visit http://localhost:5001/evaluation and run Stockfish match
```

**Command Line Tooling**:
```bash
# Automated Stockfish match evaluation
python src/evaluation/stockfish_match_eval.py \
  --file data/datasets/lichess_puzzles_1000_2000.jsonl \
  --limit 100 \
  --depth 12 \
  --out stockfish_match_after.json

# Current results format (from recent evaluation)
{
  "first_move_accuracy": 0.02,
  "legal_rate": 1.0,
  "avg_latency_sec": 0.978,
  "results": [...]
}
```

**Expert-Specific Evaluation**:
```bash
# Evaluate UCI Expert specifically
python src/evaluation/stockfish_match_eval.py \
  --expert uci \
  --file data/datasets/lichess_puzzles_1000_2000.jsonl \
  --limit 50 \
  --depth 12
```

## Explanation Quality Evaluation

### 1. Coherence and Structure

**Metrics**:
- **Logical Flow**: Does the explanation follow a clear structure?
- **Completeness**: Are all key points covered?
- **Clarity**: Is the explanation understandable?

**Implementation**:
```python
def evaluate_explanation_quality(model, test_questions):
    """Evaluate explanation coherence and structure"""
    scores = []
    
    for question in test_questions:
        response = model.generate(question)
        
        # Check for logical structure
        has_structure = check_explanation_structure(response)
        
        # Check completeness
        completeness = check_completeness(response, question)
        
        # Check clarity
        clarity = check_clarity(response)
        
        scores.append({
            'structure': has_structure,
            'completeness': completeness,
            'clarity': clarity
        })
    
    return scores
```

### 2. Chess Relevance

**Metrics**:
- **Chess Terms**: Presence of relevant chess terminology
- **Position Awareness**: Understanding of the specific position
- **Tactical Recognition**: Identification of key tactical elements

**Implementation**:
```python
def evaluate_chess_relevance(response):
    """Evaluate chess-specific relevance of response"""
    chess_terms = ['pawn', 'knight', 'bishop', 'rook', 'queen', 'king', 
                   'check', 'mate', 'castling', 'en passant', 'promotion']
    
    term_count = sum(1 for term in chess_terms if term in response.lower())
    relevance_score = min(term_count / 5, 1.0)  # Normalize to 0-1
    
    return relevance_score
```

## Benchmarking Against Prior Art

### 1. Baseline Comparisons

**Baseline 1: Base Gemma-3 Model**
- Run evaluation on untrained model
- Measure improvement from fine-tuning
- Target: Significant improvement across all metrics

**Baseline 2: Generic LLM**
- Compare to GPT-2 or similar size model
- Test if chess-specific training helps
- Target: Outperform generic models

**Baseline 3: Chess Engine**
- Compare move accuracy to Stockfish
- Test on tactical puzzles
- Target: Reasonable performance for 270M model

### 2. Human Baseline

**Target Performance**:
- **Beginner Level**: 1200-1400 Elo equivalent
- **Intermediate Level**: 1400-1600 Elo equivalent
- **Advanced Level**: 1600+ Elo equivalent

## User Experience Testing

### 1. Qualitative Assessment

**Evaluation Criteria**:
- **Helpfulness**: Does the explanation help understanding?
- **Accuracy**: Are the chess concepts correct?
- **Engagement**: Is the response engaging and educational?
- **Appropriateness**: Is the level appropriate for the user?

### 2. A/B Testing

**Test Scenarios**:
- **Engine vs Tutor Mode**: Compare response styles
- **Different Styles**: Test Fischer vs positional style
- **Complexity Levels**: Beginner vs advanced explanations

## Evaluation Pipeline

### 1. Automated Testing

```python
def run_comprehensive_evaluation(model):
    """Run full evaluation suite"""
    results = {}
    
    # Move legality
    results['move_legality'] = evaluate_move_legality(model, test_positions)
    
    # Tactical puzzles
    results['tactical_puzzles'] = evaluate_tactical_puzzles(model, puzzle_dataset)
    
    # Conceptual questions
    results['conceptual_qa'] = evaluate_conceptual_questions(model, concept_questions)
    
    # Opening knowledge
    results['opening_knowledge'] = evaluate_opening_knowledge(model, opening_dataset)
    
    # Stockfish comparison
    results['stockfish_match'] = evaluate_against_stockfish(model, test_positions)
    
    # Explanation quality
    results['explanation_quality'] = evaluate_explanation_quality(model, test_questions)
    
    return results
```

### 2. Continuous Evaluation

**Integration with Training**:
- Run evaluation after each epoch
- Track metrics over time
- Identify regression points
- Adjust training strategy based on results

**Performance Monitoring**:
- Real-time evaluation during inference
- User feedback collection
- Error rate tracking
- Response time monitoring

## Evaluation Datasets

### 1. Test Position Sets

**Tactical Puzzles**:
- Lichess puzzle dataset (5M+ puzzles)
- Curated set of 1000 puzzles by difficulty
- Mate in 1, 2, 3 puzzles
- Material gain puzzles

**Positional Questions**:
- Chess concept questions
- Rule-based questions
- Strategy questions
- Endgame technique questions

**Opening Positions**:
- Common opening positions
- Theory questions
- Plan identification
- Variation analysis

### 2. Evaluation Standards

**Move Legality**: 100% target
**Tactical Puzzles**: 70%+ for basic, 50%+ for advanced
**Conceptual QA**: 90%+ for rules, 70%+ for strategy
**Opening Knowledge**: 80%+ for common openings
**Stockfish Match**: 50%+ top move, 75%+ top 3

## Reporting and Visualization

### 1. Evaluation Reports

**Format**: JSON with detailed metrics
**Sections**:
- Executive summary
- Detailed metrics
- Comparison to baselines
- Recommendations for improvement

### 2. Performance Dashboards

**Real-time Metrics**:
- Current performance scores
- Trend analysis
- Error rate monitoring
- User satisfaction scores

**Training Progress**:
- Loss curves
- Evaluation metrics over time
- Best checkpoint identification
- Hyperparameter impact analysis

## Conclusion

This comprehensive evaluation framework ensures GemmaFischer meets high standards for both chess skill and educational value. Regular evaluation and continuous improvement will drive the project toward its goal of being both a competent chess engine and an effective tutor.
