# Changelog
## [2.2.0] - 2025-09-11

### Major Features Added
- **Expert Training System**: Complete multi-expert LoRA adapter system (UCI, Tutor, Director)
- **Web Interface Overhaul**: Full-featured web application with training controls and real-time monitoring
- **Dataset Processing Pipeline**: 100,000+ training samples processed with Stockfish validation
- **Live Model Switching**: Dynamic adapter loading and switching between trained experts
- **Comprehensive Evaluation**: Tactical puzzle evaluation and Stockfish match testing

### Training & Models
- **UCI Expert**: 50,000 samples trained for chess move generation (multiple checkpoints: 600, 800, 1000 steps)
- **Tutor Expert**: 50,000 samples trained for chess explanations (200, 400 steps)
- **Director Expert**: Q&A reasoning model with 3.2MB dataset (500, 1000 steps)
- **Curriculum Training**: Advanced training with mixed datasets and phase-based learning
- **MPS Optimization**: Native Apple Silicon acceleration throughout training pipeline

### Web Interface Enhancements
- **Training Dashboard**: Real-time training monitoring with loss curves and system stats
- **Evaluation Tools**: Built-in Stockfish match and puzzle evaluation from web UI
- **Chess Analysis**: Interactive board with move validation and real-time Q&A
- **Dataset Management**: Web-based dataset cleaning and validation tools
- **API Endpoints**: Complete REST API for training, evaluation, and chess analysis
- **Live Adapter Switching**: Switch between different trained expert models in real-time

### Dataset & Data Processing
- **Processed Datasets**: 100,000+ clean samples (50k UCI + 50k Tutor)
- **Stockfish Validation**: Automatic move legality checking and repair
- **Expert-Specific Formatting**: Tailored data preparation for each expert type
- **Quality Assurance**: Comprehensive data validation and cleaning pipeline

### Performance & Evaluation
- **Current Metrics**: Baseline tactical accuracy ~2% (room for improvement)
- **Training Efficiency**: 2-3 steps/second on M3 Pro with MPS acceleration
- **Memory Optimization**: 4-6GB peak usage, optimized for Apple Silicon
- **Model Switching**: Sub-second adapter switching in web interface

### Technical Improvements
- **Checkpoint Management**: Extensive checkpoint history across all experts
- **Resume Functionality**: Training resume capabilities for interrupted sessions
- **System Monitoring**: Real-time resource usage tracking during training
- **Error Recovery**: Improved error handling and recovery mechanisms

### Documentation Updates
- **README.md**: Updated with current status, performance metrics, and web interface features
- **Training Commands**: Updated expert training commands with current configurations
- **Web Interface Guide**: Comprehensive documentation for all web features
- **Dataset Information**: Current dataset sizes and processing status

## [2.1.0] - 2025-01-28

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
