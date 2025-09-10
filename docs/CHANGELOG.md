# Changelog
## [2.1.0] - 2025-09-10

### Added
- Unified inference API enhancements: engine-mode strict UCI postprocessing; improved error handling.
- UCI bridge fallback to Stockfish when parsing/legality fails.
- Dataset mixer (`src/training/dataset_mixer.py`) and curriculum phase support in `train_lora_poc.py`.
- Web debug endpoint `/api/debug/compare` to compare engine/tutor/Stockfish on a FEN.
- Stockfish match evaluator `src/evaluation/stockfish_match_eval.py`.

### Changed
- Training configs updated to optionally support `datasets:` mixture and `curriculum:` phases.
- Documentation updated across API, Architecture, Training, and Evaluation guides.

### Fixed
- Tests stabilized; inference tests pass (13/13).
- Import path for data prep tests by adding `data/__init__.py`.


All notable changes to ChessGemma will be documented in this file.

## [2.0.0] - 2025-01-27

### Major Reorganization
- **BREAKING**: Complete project structure reorganization
- **BREAKING**: Moved all source code to `src/` directory
- **BREAKING**: Consolidated documentation into `docs/` directory
- **BREAKING**: Reorganized data files into logical subdirectories

### New Features
- Comprehensive documentation system with architecture, training guide, and API reference
- Improved test coverage with proper unit tests
- Standardized configuration management
- Enhanced error handling and logging

### Architecture Changes
- **Training**: Moved to `src/training/` with consolidated configs
- **Inference**: Moved to `src/inference/` with chess engine integration
- **Evaluation**: Moved to `src/evaluation/` with comprehensive metrics
- **Web Interface**: Moved to `src/web/` with improved structure
- **Data Management**: Organized into `data/raw/`, `data/processed/`, `data/datasets/`

### Documentation
- **NEW**: `docs/ARCHITECTURE.md` - Complete system architecture documentation
- **NEW**: `docs/TRAINING_GUIDE.md` - Comprehensive training instructions
- **NEW**: `docs/API_REFERENCE.md` - Complete API documentation
- **UPDATED**: `README.md` - Consolidated and improved main documentation

### Testing
- **NEW**: `tests/test_inference.py` - Comprehensive inference testing
- **NEW**: `tests/test_data_prep.py` - Data preparation testing
- **IMPROVED**: Better test coverage and error handling

### File Organization
- **MOVED**: All training scripts to `src/training/`
- **MOVED**: All inference code to `src/inference/`
- **MOVED**: All evaluation code to `src/evaluation/`
- **MOVED**: Web application to `src/web/`
- **MOVED**: Configuration files to `src/training/configs/`
- **MOVED**: Dataset creation scripts to `data/`
- **ARCHIVED**: Old comparison and planning files to `archive/`

### Cleanup
- **REMOVED**: Duplicate inference scripts
- **REMOVED**: Outdated comparison files
- **REMOVED**: Redundant documentation files
- **CLEANED**: Inconsistent naming conventions

### Configuration
- **STANDARDIZED**: All configuration files use consistent YAML format
- **IMPROVED**: Better parameter organization and documentation
- **ENHANCED**: More flexible training configurations

## [1.0.0] - 2025-01-20

### Initial Release
- Basic LoRA fine-tuning pipeline for Gemma-3 270M
- Chess engine integration with Stockfish
- Web interface for chess Q&A
- Multiple training configurations
- Basic evaluation framework

### Features
- LoRA fine-tuning with Unsloth optimization
- Apple Silicon MPS acceleration
- Chess move validation and analysis
- Interactive web interface
- Multiple dataset support
- Checkpoint management

### Components
- Training pipeline with SFTTrainer
- Inference engine with model caching
- Chess engine integration
- Flask web application
- Evaluation metrics
- Data processing utilities

---

## Migration Guide

### For Users Upgrading from v1.x

1. **Update Import Paths**: All imports now use the new `src/` structure
   ```python
   # Old
   from inference import ChessGemmaInference
   
   # New
   from src.inference.inference import ChessGemmaInference
   ```

2. **Update Configuration Paths**: Config files moved to `src/training/configs/`
   ```bash
   # Old
   python train.py --config configs/lora_full.yaml
   
   # New
   python src/training/train.py --config src/training/configs/lora_full.yaml
   ```

3. **Update Data Paths**: Data files reorganized
   ```python
   # Old
   dataset_path = "data/finetune/chess_finetune.jsonl"
   
   # New
   dataset_path = "data/datasets/chess_finetune.jsonl"
   ```

4. **Update Web App**: Web app moved to `src/web/`
   ```bash
   # Old
   python run_web_app.py
   
   # New
   python src/web/app.py
   # or
   python src/web/run_web_app.py
   ```

### Breaking Changes

- **File Structure**: Complete reorganization of project structure
- **Import Paths**: All Python imports need to be updated
- **Configuration**: Config file locations changed
- **Data Paths**: Dataset file locations changed
- **Script Locations**: All scripts moved to appropriate directories

### Deprecated Features

- Old comparison files (moved to `archive/`)
- Duplicate inference scripts (removed)
- Outdated documentation files (consolidated)

---

## Future Roadmap

### Planned for v2.1.0
- [ ] Enhanced dataset generation tools
- [ ] Improved evaluation metrics
- [ ] Better error handling and recovery
- [ ] Performance optimizations

### Planned for v2.2.0
- [ ] Multi-model support
- [ ] Advanced chess analysis features
- [ ] Mobile app interface
- [ ] Cloud deployment options

### Planned for v3.0.0
- [ ] Microservices architecture
- [ ] Real-time collaboration features
- [ ] Advanced chess engine integration
- [ ] Machine learning pipeline improvements
