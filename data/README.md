# GemmaFischer Data Directory

This directory contains all chess training data for the GemmaFischer project, organized by type and source.

## Directory Structure

```
data/
├── README.md                           # This file
├── raw/                               # Raw, unprocessed data
│   ├── lichess/                       # Lichess datasets
│   │   ├── puzzles/                   # Puzzle datasets
│   │   ├── games/                     # Game databases
│   │   └── studies/                   # Study collections
│   ├── chess_books/                   # Chess literature
│   ├── historical_games/              # Master game collections
│   ├── opening_theory/                # Opening databases
│   └── visual/                        # Board position images
├── processed/                         # Cleaned and standardized data
│   ├── qa_pairs/                      # Question-answer pairs
│   ├── conversations/                 # Chat format data
│   ├── positions/                     # FEN position data
│   └── visual/                        # Processed images
├── datasets/                          # Training-ready datasets
│   ├── chess_finetune_*.jsonl         # Current training datasets
│   ├── lichess_puzzles_*.jsonl        # Puzzle datasets
│   ├── opening_theory_*.jsonl         # Opening datasets
│   ├── endgame_*.jsonl                # Endgame datasets
│   ├── cot_reasoning_*.jsonl          # Chain-of-thought data
│   └── visual_training_*.jsonl        # Vision module data
├── validation/                        # Data quality validation
│   ├── quality_reports/               # Validation reports
│   └── filtered_data/                 # Quality-filtered datasets
└── scripts/                           # Data processing scripts
    ├── download_data.py               # Download raw datasets
    ├── process_lichess.py             # Process Lichess data
    ├── generate_cot_data.py           # Generate CoT examples
    ├── create_visual_data.py          # Create visual datasets
    └── validate_data.py               # Data quality validation
```

## Data Sources

### 1. Lichess Datasets
- **Puzzles**: 5M+ tactical puzzles with ratings and themes
- **Games**: Millions of rated games with player ratings
- **Studies**: Human-annotated chess studies and lessons

### 2. Chess Literature
- **Opening Theory**: Comprehensive opening databases
- **Endgame Theory**: Endgame positions and techniques
- **Strategic Concepts**: Positional and tactical principles

### 3. Historical Games
- **Master Games**: Annotated games from chess masters
- **Tournament Games**: High-level tournament games
- **Classic Games**: Famous historical games

### 4. Visual Data
- **Board Positions**: Rendered chess positions
- **Piece Recognition**: Training data for vision module
- **Board Detection**: Various board orientations and styles

## Data Formats

### Training Format (JSONL)
```json
{
  "text": "Question: [question]\nAnswer: [answer]",
  "conversations": [
    {"role": "system", "content": "[system prompt]"},
    {"role": "user", "content": "[question]"},
    {"role": "assistant", "content": "[answer]"}
  ],
  "category": "[category]",
  "difficulty": "[difficulty]",
  "source": "[source]",
  "metadata": {...}
}
```

### Chain-of-Thought Format
```json
{
  "text": "Question: [question]\nAnswer: [step-by-step reasoning]",
  "conversations": [
    {"role": "user", "content": "[question]"},
    {"role": "assistant", "content": "Let me analyze this step by step:\n1. [step 1]\n2. [step 2]\n3. [conclusion]"}
  ],
  "reasoning_steps": [...],
  "category": "cot_reasoning"
}
```

### Visual Training Format
```json
{
  "image_path": "path/to/board_image.png",
  "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  "question": "What is the position on this board?",
  "answer": "This is the starting position of a chess game.",
  "category": "visual_recognition"
}
```

## Quality Standards

### Data Quality Metrics
- **Chess Relevance**: 100% chess-related content
- **Accuracy**: Validated moves and positions
- **Completeness**: Complete question-answer pairs
- **Diversity**: Balanced across categories and difficulties
- **Length**: Appropriate length for training (10-500 characters)

### Validation Process
1. **Format Validation**: Correct JSON structure
2. **Chess Validation**: Valid FEN, moves, positions
3. **Content Validation**: Chess-relevant questions/answers
4. **Quality Scoring**: Length, clarity, educational value
5. **Deduplication**: Remove duplicate content

## Usage

### Download Raw Data
```bash
python data/scripts/download_data.py --source lichess --type puzzles
```

### Process Data
```bash
python data/scripts/process_lichess.py --input raw/lichess/puzzles --output datasets/
```

### Validate Data
```bash
python data/scripts/validate_data.py --dataset datasets/chess_finetune_full.jsonl
```

## Data Statistics

| Dataset | Size | Categories | Quality Score |
|---------|------|------------|---------------|
| ChessInstruct | 100k | Mixed | 7.2/10 |
| Lichess Puzzles | 5M+ | Tactics | 9.1/10 |
| Opening Theory | 50k | Openings | 8.8/10 |
| Endgame Data | 25k | Endgames | 9.3/10 |
| CoT Examples | 10k | Reasoning | 8.5/10 |
| Visual Data | 100k | Vision | 8.0/10 |

## Maintenance

- **Regular Updates**: Download fresh data monthly
- **Quality Monitoring**: Continuous validation and filtering
- **Version Control**: Track dataset versions and changes
- **Backup**: Maintain multiple copies of critical datasets
