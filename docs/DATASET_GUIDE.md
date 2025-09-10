# GemmaFischer Dataset Guide

## Overview

This guide covers comprehensive dataset preparation, curation, and management for GemmaFischer. High-quality training data is crucial for achieving both chess skill and educational value in the model.

## Dataset Categories

### 1. Chess Q&A Datasets

#### ChessInstruct v1.5 (Primary)
- **Source**: Thytu/ChessInstruct on Hugging Face
- **Size**: ~100k examples
- **Content**: Chess problems in text form
- **Format**: Question-answer pairs
- **Quality**: Mixed - needs curation

**Refinement Strategy**:
```python
def refine_chess_instruct(dataset):
    """Refine ChessInstruct dataset for better quality"""
    refined_data = []
    
    for example in dataset:
        # Filter out overly long sequences
        if len(example['answer']) > 500:
            continue
            
        # Ensure chess relevance
        if not contains_chess_terms(example['question']):
            continue
            
        # Standardize format
        refined_example = {
            'question': clean_question(example['question']),
            'answer': clean_answer(example['answer']),
            'category': classify_question(example['question'])
        }
        refined_data.append(refined_example)
    
    return refined_data
```

#### Implemented Refinement Pipeline
We provide a concrete refinement script that applies:
- Length filter (≤ 500 chars)
- Chess-term presence filter
- Standardized 3-message `conversations` format
- Topic and difficulty tagging

```bash
python scripts/refine_dataset.py
# Output: data/finetune/chess_finetune_refined.jsonl
```

#### Lichess Studies Dataset
- **Source**: Icannos/chess_studies on Hugging Face
- **Content**: Human-written study notes and commentary
- **Format**: Moves with explanatory text
- **Quality**: High - human expert commentary

**Processing**:
```python
def process_lichess_studies(studies_dataset):
    """Convert Lichess studies to Q&A format"""
    qa_pairs = []
    
    for study in studies_dataset:
        moves = study['moves']
        commentary = study['commentary']
        
        # Create position-based questions
        for i, move in enumerate(moves):
            position = get_position_after_move(moves[:i])
            comment = commentary[i] if i < len(commentary) else ""
            
            if comment:
                qa_pairs.append({
                    'question': f"Explain this position: {position.fen()}",
                    'answer': comment,
                    'category': 'positional_analysis'
                })
    
    return qa_pairs
```

### 2. PGN Game Data

#### Lichess Open Database
- **Source**: Lichess open database
- **Size**: Billions of games
- **Content**: Raw game data in PGN format
- **Usage**: Move prediction training

**Processing Strategy**:
```python
def process_pgn_games(pgn_file, sample_size=1000000):
    """Process PGN games for move prediction training"""
    games = []
    
    with open(pgn_file, 'r') as f:
        for game in pgn.read_game(f):
            if len(games) >= sample_size:
                break
                
            # Extract positions and moves
            board = chess.Board()
            for move in game.mainline_moves():
                position = board.fen()
                next_move = move.uci()
                
                games.append({
                    'position': position,
                    'move': next_move,
                    'game_info': {
                        'white_rating': game.headers.get('WhiteElo', 0),
                        'black_rating': game.headers.get('BlackElo', 0),
                        'result': game.headers.get('Result', '')
                    }
                })
                
                board.push(move)
    
    return games
```

#### Filtering for Quality
```python
def filter_high_quality_games(games, min_rating=1800):
    """Filter games by player rating and game quality"""
    filtered_games = []
    
    for game in games:
        white_rating = int(game['game_info']['white_rating'])
        black_rating = int(game['game_info']['black_rating'])
        
        # Only include games with strong players
        if white_rating >= min_rating and black_rating >= min_rating:
            filtered_games.append(game)
    
    return filtered_games
```

### 3. Chess Puzzle Datasets

#### Lichess Puzzle Dataset
- **Source**: Lichess/chess-puzzles on Hugging Face
- **Size**: ~5 million puzzles
- **Content**: FEN positions with solutions
- **Quality**: High - rated by difficulty

**Processing**:
```python
def process_lichess_puzzles(puzzle_dataset, max_puzzles=100000):
    """Convert Lichess puzzles to training format"""
    training_examples = []
    
    for puzzle in puzzle_dataset[:max_puzzles]:
        fen = puzzle['fen']
        solution = puzzle['solution']
        theme = puzzle['theme']
        rating = puzzle['rating']
        
        # Create training examples
        training_examples.append({
            'question': f"Find the best move in this position: {fen}",
            'answer': f"The best move is {solution[0]} because it {get_tactical_explanation(theme)}",
            'category': 'tactical_puzzle',
            'difficulty': rating,
            'theme': theme
        })
    
    return training_examples
```

#### Implemented Puzzle Ingestion
We include an ingestion script for Lichess puzzles (CSV), filtered by rating [1000, 2000], with motifs extraction and final UCI move line for tutor mode:

```bash
python scripts/ingest_lichess_puzzles.py
# Input: data/raw/lichess_puzzles.csv (columns: FEN, Moves, Rating, Themes)
# Output: data/datasets/lichess_puzzles_1000_2000.jsonl
```

If you have the official compressed archive (`lichess_db_puzzle.csv.zst`), place it at `data/raw/lichess_puzzles.csv.zst` or `data/raw/lichess/puzzles/lichess_puzzles.csv.zst` and install streaming support:

```bash
pip install zstandard
PUZZLES_LIMIT=50000 python scripts/ingest_lichess_puzzles.py
```

The script auto-detects `.zst` and `.csv` inputs in common locations and respects `PUZZLES_LIMIT` to cap the number of filtered rows ingested.

### 4. Annotated Game Datasets

#### Master Game Collections
- **Source**: Chess books, databases
- **Content**: Annotated games with commentary
- **Format**: PGN with comments
- **Quality**: Very high - expert analysis

**Processing**:
```python
def process_annotated_games(pgn_file):
    """Extract Q&A from annotated games"""
    qa_pairs = []
    
    with open(pgn_file, 'r') as f:
        for game in pgn.read_game(f):
            board = chess.Board()
            
            for node in game.mainline():
                if node.comment:
                    position = board.fen()
                    comment = node.comment
                    
                    # Create Q&A pairs
                    qa_pairs.append({
                        'question': f"Explain this position: {position}",
                        'answer': comment,
                        'category': 'game_commentary'
                    })
                
                if node.move:
                    board.push(node.move)
    
    return qa_pairs
```

### 5. Opening Theory Datasets

#### ECO Database
- **Source**: Encyclopedia of Chess Openings
- **Content**: Opening variations with names and plans
- **Format**: Structured opening data
- **Quality**: High - authoritative source

**Processing**:
```python
def process_opening_theory(eco_data):
    """Create opening theory training data"""
    training_examples = []
    
    for opening in eco_data:
        moves = opening['moves']
        name = opening['name']
        plan = opening['plan']
        
        # Create identification questions
        training_examples.append({
            'question': f"What opening is this: {moves}?",
            'answer': f"This is the {name}. The main plan is {plan}.",
            'category': 'opening_identification'
        })
        
        # Create plan questions
        training_examples.append({
            'question': f"What is the plan in the {name}?",
            'answer': plan,
            'category': 'opening_plan'
        })
    
    return training_examples
```

## Data Preparation Pipeline

### 1. Data Collection

```python
def collect_all_datasets():
    """Collect data from all sources"""
    datasets = {}
    
    # ChessInstruct
    datasets['chess_instruct'] = load_chess_instruct()
    
    # Lichess studies
    datasets['lichess_studies'] = load_lichess_studies()
    
    # PGN games
    datasets['pgn_games'] = load_pgn_games()
    
    # Puzzles
    datasets['puzzles'] = load_lichess_puzzles()
    
    # Annotated games
    datasets['annotated_games'] = load_annotated_games()
    
    # Opening theory
    datasets['opening_theory'] = load_opening_theory()
    
    return datasets
```

### 2. Normalization to Conversations
All datasets are normalized to the following `conversations` schema for instruction tuning:

```json
{
  "conversations": [
    {"role": "system", "content": "You are a chess tutor and engine."},
    {"role": "user", "content": "Position: <FEN>\nMode: Tutor\nAnalyze step-by-step."},
    {"role": "assistant", "content": "<analysis>\nBest move: e2e4"}
  ],
  "topic": "tactics|strategy|endgames|openings",
  "difficulty": "beginner|intermediate|advanced"
}
```

### 3. Quality Controls
- Enforce final UCI move line in tutor-mode answers (for extraction)
- Validate UCI syntax during ingestion where applicable
- Attach topic and difficulty labels for curriculum phases

### 2. Data Cleaning

```python
def clean_dataset(dataset):
    """Clean and standardize dataset"""
    cleaned = []
    
    for example in dataset:
        # Remove duplicates
        if example in cleaned:
            continue
            
        # Clean text
        example['question'] = clean_text(example['question'])
        example['answer'] = clean_text(example['answer'])
        
        # Validate chess content
        if not is_chess_related(example):
            continue
            
        # Standardize format
        example = standardize_format(example)
        cleaned.append(example)
    
    return cleaned
```

### 3. Data Formatting

```python
def format_for_training(dataset):
    """Format dataset for LoRA training"""
    formatted_data = []
    
    for example in dataset:
        # Create conversation format
        conversation = [
            {
                "role": "user",
                "content": example['question']
            },
            {
                "role": "assistant", 
                "content": example['answer']
            }
        ]
        
        formatted_data.append({
            "conversations": conversation,
            "category": example['category'],
            "difficulty": example.get('difficulty', 'medium')
        })
    
    return formatted_data
```

## Quality Control

### 1. Chess Relevance Filtering

```python
def filter_chess_relevance(dataset):
    """Filter for chess-relevant content"""
    chess_terms = [
        'pawn', 'knight', 'bishop', 'rook', 'queen', 'king',
        'check', 'mate', 'castling', 'en passant', 'promotion',
        'opening', 'middlegame', 'endgame', 'tactics', 'strategy'
    ]
    
    filtered = []
    for example in dataset:
        text = (example['question'] + ' ' + example['answer']).lower()
        if any(term in text for term in chess_terms):
            filtered.append(example)
    
    return filtered
```

### 2. Length Filtering

```python
def filter_by_length(dataset, min_length=10, max_length=500):
    """Filter examples by length"""
    filtered = []
    
    for example in dataset:
        answer_length = len(example['answer'])
        if min_length <= answer_length <= max_length:
            filtered.append(example)
    
    return filtered
```

### 3. Quality Scoring

```python
def score_data_quality(dataset):
    """Score data quality for prioritization"""
    scored_data = []
    
    for example in dataset:
        score = 0
        
        # Length score (prefer medium length)
        length = len(example['answer'])
        if 50 <= length <= 200:
            score += 2
        elif 20 <= length <= 300:
            score += 1
        
        # Chess term density
        chess_terms = count_chess_terms(example['answer'])
        score += min(chess_terms / 5, 2)
        
        # Structure score
        if has_clear_structure(example['answer']):
            score += 1
        
        example['quality_score'] = score
        scored_data.append(example)
    
    return sorted(scored_data, key=lambda x: x['quality_score'], reverse=True)
```

## Dataset Splits

### 1. Training/Validation/Test Split

```python
def create_dataset_splits(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Create train/validation/test splits"""
    # Shuffle dataset
    random.shuffle(dataset)
    
    n = len(dataset)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    splits = {
        'train': dataset[:train_end],
        'validation': dataset[train_end:val_end],
        'test': dataset[val_end:]
    }
    
    return splits
```

### 2. Category Balancing

```python
def balance_categories(dataset):
    """Balance dataset across categories"""
    categories = {}
    
    # Group by category
    for example in dataset:
        cat = example['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(example)
    
    # Balance categories
    min_size = min(len(cat_data) for cat_data in categories.values())
    balanced = []
    
    for cat_data in categories.values():
        balanced.extend(cat_data[:min_size])
    
    return balanced
```

## Data Augmentation

### 1. Question Variation

```python
def augment_questions(dataset):
    """Create variations of questions"""
    augmented = []
    
    for example in dataset:
        # Original
        augmented.append(example)
        
        # Variations
        variations = [
            f"Analyze this position: {example['question']}",
            f"What's happening here? {example['question']}",
            f"Explain this: {example['question']}"
        ]
        
        for variation in variations:
            augmented.append({
                'question': variation,
                'answer': example['answer'],
                'category': example['category']
            })
    
    return augmented
```

### 2. Answer Paraphrasing

```python
def augment_answers(dataset):
    """Create paraphrased answers"""
    augmented = []
    
    for example in dataset:
        # Original
        augmented.append(example)
        
        # Paraphrased versions
        paraphrases = [
            f"In this position, {example['answer'].lower()}",
            f"The key point is: {example['answer']}",
            f"Here's what's happening: {example['answer']}"
        ]
        
        for paraphrase in paraphrases:
            augmented.append({
                'question': example['question'],
                'answer': paraphrase,
                'category': example['category']
            })
    
    return augmented
```

## Dataset Statistics

### 1. Size and Distribution

```python
def analyze_dataset(dataset):
    """Analyze dataset statistics"""
    stats = {
        'total_examples': len(dataset),
        'categories': {},
        'avg_question_length': 0,
        'avg_answer_length': 0,
        'difficulty_distribution': {}
    }
    
    total_q_length = 0
    total_a_length = 0
    
    for example in dataset:
        # Category distribution
        cat = example['category']
        stats['categories'][cat] = stats['categories'].get(cat, 0) + 1
        
        # Length statistics
        total_q_length += len(example['question'])
        total_a_length += len(example['answer'])
        
        # Difficulty distribution
        diff = example.get('difficulty', 'medium')
        stats['difficulty_distribution'][diff] = stats['difficulty_distribution'].get(diff, 0) + 1
    
    stats['avg_question_length'] = total_q_length / len(dataset)
    stats['avg_answer_length'] = total_a_length / len(dataset)
    
    return stats
```

## Best Practices

### 1. Data Quality
- **Validate chess content**: Ensure all examples are chess-related
- **Check accuracy**: Verify answers are correct
- **Maintain consistency**: Use consistent formatting and terminology
- **Balance categories**: Ensure good representation across chess topics

### 2. Data Management
- **Version control**: Track dataset versions and changes
- **Documentation**: Document data sources and processing steps
- **Backup**: Keep multiple copies of important datasets
- **Metadata**: Include relevant metadata for each example

### 3. Training Considerations
- **Curriculum learning**: Start with simple examples, progress to complex
- **Multi-task learning**: Mix different types of chess tasks
- **Regular evaluation**: Test model performance on held-out data
- **Iterative improvement**: Continuously refine dataset based on model performance

## Conclusion

High-quality datasets are essential for training a successful chess AI. This guide provides comprehensive strategies for collecting, processing, and managing chess training data. Regular evaluation and iterative improvement will ensure the dataset continues to support model development effectively.
