# Chess Data Collection and Processing Usage Guide

This guide explains how to use the comprehensive chess data collection and processing system for GemmaFischer.

## Quick Start

### 1. Run the Complete Pipeline
```bash
# Run the entire data pipeline (download, process, validate, combine)
python data/scripts/master_data_pipeline.py

# Skip download step if you already have raw data
python data/scripts/master_data_pipeline.py --skip-download

# Create training configuration after pipeline
python data/scripts/master_data_pipeline.py --create-config
```

### 2. Individual Steps

#### Download Raw Data
```bash
# Download all available datasets
python data/scripts/download_data.py

# Download only Lichess puzzles (faster)
python data/scripts/download_data.py --source lichess --max_puzzles 100000

# Download with games (very large files)
python data/scripts/download_data.py --include_games
```

#### Process Lichess Data
```bash
# Process puzzles
python data/scripts/process_lichess.py --type puzzles --min_rating 1000 --max_rating 2000

# Process games
python data/scripts/process_lichess.py --type games --min_rating 1800

# Create combined dataset
python data/scripts/process_lichess.py --type combined --include_games
```

#### Generate Chain-of-Thought Data
```bash
# Generate all CoT examples
python data/scripts/generate_cot_data.py --type all --num_examples 2000

# Generate only tactical examples
python data/scripts/generate_cot_data.py --type tactical --num_examples 1000
```

#### Generate Visual Data
```bash
# Generate all visual data
python data/scripts/create_visual_data.py --type all --num_examples 2000

# Generate only board positions
python data/scripts/create_visual_data.py --type board_positions --num_examples 1000
```

#### Validate Datasets
```bash
# Validate all datasets
python data/scripts/validate_data.py --validate_all

# Validate specific dataset
python data/scripts/validate_data.py --input data/datasets/chess_finetune_full.jsonl --output data/validation/filtered_data/validated_dataset.jsonl
```

## Data Sources

### 1. Lichess Datasets
- **Puzzles**: 5M+ tactical puzzles with ratings and themes
- **Games**: Millions of rated games with player ratings
- **Studies**: Human-annotated chess studies and lessons

### 2. Generated Data
- **Chain-of-Thought**: Step-by-step reasoning examples
- **Visual Data**: Board position images and piece recognition
- **Opening Theory**: Comprehensive opening databases
- **Endgame Data**: Endgame positions and techniques
- **Historical Games**: Master game collections

## Data Quality Standards

### Validation Criteria
- **Format**: Valid JSON structure with required fields
- **Chess Relevance**: Contains chess-related content
- **Length**: Appropriate length for training (10-1000 characters)
- **Accuracy**: Valid FEN strings and chess moves
- **Quality Score**: Minimum quality threshold (default: 5.0/10)

### Quality Metrics
- **Chess Relevance**: Percentage of chess-related content
- **Length Distribution**: Average question/answer lengths
- **Category Balance**: Distribution across chess topics
- **Difficulty Levels**: Beginner, intermediate, advanced

## Output Formats

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
  "conversations": [...],
  "reasoning_steps": ["step1", "step2", "step3"],
  "category": "cot_reasoning"
}
```

### Visual Training Format
```json
{
  "text": "Question: [question]\nAnswer: [answer]",
  "conversations": [...],
  "image_path": "path/to/board_image.png",
  "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  "category": "visual_recognition"
}
```

## Directory Structure

```
data/
├── raw/                           # Raw, unprocessed data
│   ├── lichess/                   # Lichess datasets
│   ├── chess_books/               # Chess literature
│   ├── historical_games/          # Master games
│   ├── opening_theory/            # Opening databases
│   └── visual/                    # Visual data
├── processed/                     # Cleaned data
├── datasets/                      # Training-ready datasets
├── validation/                    # Quality validation
└── scripts/                       # Processing scripts
```

## Performance Considerations

### Memory Usage
- **Lichess Puzzles**: ~2GB uncompressed
- **Lichess Games**: ~10-20GB per month
- **Processing**: 4-8GB RAM recommended
- **Validation**: 2-4GB RAM for large datasets

### Processing Time
- **Download**: 10-30 minutes (depending on connection)
- **Puzzle Processing**: 5-15 minutes (1M puzzles)
- **Game Processing**: 30-60 minutes (10K games)
- **CoT Generation**: 2-5 minutes (2K examples)
- **Visual Generation**: 10-30 minutes (2K images)
- **Validation**: 5-15 minutes (per dataset)

### Storage Requirements
- **Raw Data**: 20-50GB (depending on sources)
- **Processed Data**: 5-10GB
- **Final Datasets**: 1-5GB
- **Total**: 30-70GB (recommended: 100GB free space)

## Troubleshooting

### Common Issues

#### Download Failures
```bash
# Check internet connection
# Verify disk space (need 50GB+ free)
# Try smaller datasets first
python data/scripts/download_data.py --max_puzzles 10000
```

#### Processing Errors
```bash
# Check Python dependencies
pip install -r requirements.txt

# Verify file permissions
chmod +x data/scripts/*.py

# Check available memory
# Close other applications
```

#### Validation Failures
```bash
# Lower quality threshold
python data/scripts/validate_data.py --min_quality 3.0

# Check specific dataset
python data/scripts/validate_data.py --input path/to/dataset.jsonl
```

### Dependencies
```bash
# Install required packages
pip install chess python-chess pillow requests zstandard

# For visual data generation
pip install pillow

# For data processing
pip install pandas numpy
```

## Best Practices

### 1. Incremental Processing
- Start with small datasets to test the pipeline
- Process data in batches to manage memory usage
- Validate each step before proceeding

### 2. Quality Control
- Always validate datasets before training
- Review quality reports and adjust thresholds
- Keep backup copies of validated datasets

### 3. Resource Management
- Monitor disk space during processing
- Use appropriate batch sizes for your system
- Process during off-peak hours for large downloads

### 4. Data Organization
- Keep raw data separate from processed data
- Use descriptive filenames with timestamps
- Document any custom processing steps

## Advanced Usage

### Custom Configuration
```python
# Modify pipeline configuration
config = {
    'lichess_puzzles': {
        'max_puzzles': 500000,
        'min_rating': 1200,
        'max_rating': 1800
    },
    'validation': {
        'min_quality_score': 6.0
    }
}
```

### Custom Data Sources
```python
# Add your own data processing
def process_custom_data(input_file, output_file):
    # Your custom processing logic
    pass
```

### Integration with Training
```bash
# Use generated datasets for training
python src/training/train.py --config src/training/configs/gemmafischer_combined.yaml
```

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the validation reports in `data/validation/`
3. Check the pipeline results in `data/pipeline_results.json`
4. Verify all dependencies are installed correctly

## Next Steps

After running the data pipeline:
1. **Review Quality Reports**: Check validation results
2. **Start Training**: Use the generated datasets for model training
3. **Monitor Performance**: Track training progress and adjust data as needed
4. **Iterate**: Refine datasets based on model performance
