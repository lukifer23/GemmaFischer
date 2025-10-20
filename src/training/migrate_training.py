#!/usr/bin/env python3
"""
Training Script Migration Helper

Helps migrate from individual training scripts to the unified trainer.
Provides backward compatibility and migration assistance.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.unified_trainer import UnifiedChessTrainer


class TrainingMigrationHelper:
    """Helper for migrating from old training scripts to unified trainer."""
    
    def __init__(self):
        self.old_scripts = {
            "train_lora_poc.py": "Individual expert training",
            "train_chessgemmma.py": "Unified training orchestrator", 
            "expert_trainer.py": "Expert-specific training",
            "curriculum_trainer.py": "Curriculum learning"
        }
        
        self.unified_script = "unified_trainer.py"
    
    def analyze_old_scripts(self) -> Dict[str, Any]:
        """Analyze old training scripts for migration."""
        analysis = {
            "found_scripts": [],
            "missing_scripts": [],
            "migration_recommendations": []
        }
        
        training_dir = Path(__file__).parent
        
        for script_name in self.old_scripts.keys():
            script_path = training_dir / script_name
            if script_path.exists():
                analysis["found_scripts"].append({
                    "name": script_name,
                    "path": str(script_path),
                    "description": self.old_scripts[script_name]
                })
            else:
                analysis["missing_scripts"].append(script_name)
        
        # Generate migration recommendations
        analysis["migration_recommendations"] = self._generate_migration_recommendations(analysis["found_scripts"])
        
        return analysis
    
    def _generate_migration_recommendations(self, found_scripts: List[Dict[str, Any]]) -> List[str]:
        """Generate migration recommendations."""
        recommendations = []
        
        if not found_scripts:
            recommendations.append("No old training scripts found - you can use the unified trainer directly")
            return recommendations
        
        recommendations.append("The unified trainer consolidates all training functionality:")
        recommendations.append("  - Individual expert training (replaces train_lora_poc.py)")
        recommendations.append("  - Unified training orchestrator (replaces train_chessgemmma.py)")
        recommendations.append("  - Expert-specific training (replaces expert_trainer.py)")
        recommendations.append("  - Curriculum learning (integrated into unified trainer)")
        
        recommendations.append("\nMigration steps:")
        recommendations.append("1. Use 'python -m src.training.unified_trainer' instead of individual scripts")
        recommendations.append("2. Old checkpoints are compatible and will be automatically detected")
        recommendations.append("3. Configuration is now centralized in the unified trainer")
        
        return recommendations
    
    def create_migration_guide(self, output_path: str = "TRAINING_MIGRATION_GUIDE.md") -> None:
        """Create a migration guide for users."""
        analysis = self.analyze_old_scripts()
        
        guide_content = f"""# Training Script Migration Guide

## Overview

The ChessGemma training system has been consolidated into a single, unified trainer that replaces multiple individual training scripts.

## Migration Summary

### Old Scripts Found
"""
        
        if analysis["found_scripts"]:
            for script in analysis["found_scripts"]:
                guide_content += f"- **{script['name']}**: {script['description']}\n"
        else:
            guide_content += "- No old training scripts found\n"
        
        guide_content += f"""
### New Unified Trainer
- **unified_trainer.py**: Consolidated training system with all functionality

## Migration Steps

### 1. Update Training Commands

#### Old Commands
```bash
# Individual expert training
python -m src.training.train_lora_poc --expert uci --config auto --max_steps_override 1600

# Unified training orchestrator  
python -m src.training.train_chessgemmma --experts uci tutor

# Expert-specific training
python -m src.training.expert_trainer --expert uci --validate
```

#### New Commands
```bash
# Train all experts
python -m src.training.unified_trainer

# Train specific expert
python -m src.training.unified_trainer --expert uci

# Train with custom config
python -m src.training.unified_trainer --config custom_config.json

# Train without validation
python -m src.training.unified_trainer --no-validate

# Fresh training (no resume)
python -m src.training.unified_trainer --no-resume
```

### 2. Configuration Migration

The unified trainer uses built-in expert configurations that can be overridden with a custom config file:

```json
{{
  "experts": {{
    "uci": {{
      "max_steps": 1600,
      "batch_size": 1,
      "learning_rate": 2e-4,
      "timeout_minutes": 240
    }},
    "tutor": {{
      "max_steps": 1000,
      "batch_size": 1,
      "learning_rate": 2e-4,
      "timeout_minutes": 180
    }},
    "director": {{
      "max_steps": 1000,
      "batch_size": 1,
      "learning_rate": 2e-4,
      "timeout_minutes": 180
    }}
  }}
}}
```

### 3. Checkpoint Compatibility

- Existing checkpoints are fully compatible
- The unified trainer will automatically detect and resume from existing checkpoints
- No migration of checkpoint data is required

### 4. Feature Comparison

| Feature | Old Scripts | Unified Trainer |
|---------|-------------|-----------------|
| Individual expert training | Yes (train_lora_poc.py) | Yes (--expert option) |
| Multi-expert training | Yes (train_chessgemmma.py) | Yes (--expert all) |
| Curriculum learning | Yes (curriculum_trainer.py) | Yes (Built-in support) |
| Checkpoint management | Multiple scripts | Unified system |
| Error handling | Basic | Comprehensive |
| Monitoring | Basic | Advanced |
| Validation | Manual | Automatic |

## Benefits of Migration

1. **Simplified Interface**: Single command for all training needs
2. **Better Error Handling**: Comprehensive error recovery and logging
3. **Improved Monitoring**: Real-time metrics and progress tracking
4. **Unified Configuration**: Centralized configuration management
5. **Enhanced Validation**: Automatic model validation after training
6. **Timeout Protection**: Built-in timeout handling and recovery

## Backward Compatibility

- All existing checkpoints remain compatible
- Old training scripts are preserved but deprecated
- Gradual migration is supported

## Support

If you encounter issues during migration, check the unified trainer logs and refer to the comprehensive error handling system.
"""
        
        # Write migration guide
        with open(output_path, 'w') as f:
            f.write(guide_content)
        
        print(f"Migration guide created: {output_path}")
    
    def test_unified_trainer(self) -> bool:
        """Test if the unified trainer works correctly."""
        try:
            print("Testing unified trainer...")
            
            # Test initialization
            trainer = UnifiedChessTrainer()
            print("Unified trainer initialized successfully")
            
            # Test expert configs
            configs = trainer.expert_configs
            print(f"Found {len(configs)} expert configurations")
            
            for expert_name, config in configs.items():
                print(f"   - {expert_name}: {config.description}")
            
            print("Unified trainer test passed")
            return True
            
        except Exception as e:
            print(f"Unified trainer test failed: {e}")
            return False
    
    def archive_old_scripts(self, archive_dir: str = "archive/training_scripts") -> None:
        """Archive old training scripts."""
        archive_path = Path(archive_dir)
        archive_path.mkdir(parents=True, exist_ok=True)
        
        training_dir = Path(__file__).parent
        
        for script_name in self.old_scripts.keys():
            script_path = training_dir / script_name
            if script_path.exists():
                # Copy to archive
                archive_script_path = archive_path / script_name
                shutil.copy2(script_path, archive_script_path)
                print(f"Archived {script_name} to {archive_script_path}")
                
                # Add deprecation notice
                with open(script_path, 'r') as f:
                    content = f.read()
                
                deprecation_notice = '''"""
DEPRECATED: This script has been replaced by unified_trainer.py

Please use the unified trainer instead:
    python -m src.training.unified_trainer

For migration help, see TRAINING_MIGRATION_GUIDE.md
"""

'''
                
                with open(script_path, 'w') as f:
                    f.write(deprecation_notice + content)
                
                print(f"Added deprecation notice to {script_name}")


def main():
    """Main entry point for migration helper."""
    parser = argparse.ArgumentParser(
        description="ChessGemma Training Migration Helper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze old scripts and create migration guide
  python -m src.training.migrate_training --analyze

  # Test unified trainer
  python -m src.training.migrate_training --test

  # Archive old scripts
  python -m src.training.migrate_training --archive

  # Full migration process
  python -m src.training.migrate_training --migrate
        """
    )
    
    parser.add_argument('--analyze', action='store_true', help='Analyze old training scripts')
    parser.add_argument('--test', action='store_true', help='Test unified trainer')
    parser.add_argument('--archive', action='store_true', help='Archive old training scripts')
    parser.add_argument('--migrate', action='store_true', help='Run full migration process')
    parser.add_argument('--guide', type=str, default='TRAINING_MIGRATION_GUIDE.md', help='Migration guide output path')
    
    args = parser.parse_args()
    
    if not any([args.analyze, args.test, args.archive, args.migrate]):
        parser.print_help()
        return
    
    print("ChessGemma Training Migration Helper")
    print("=" * 50)
    
    helper = TrainingMigrationHelper()
    
    if args.analyze or args.migrate:
        print("Analyzing old training scripts...")
        analysis = helper.analyze_old_scripts()
        
        print(f"\nAnalysis Results:")
        print(f"   Found scripts: {len(analysis['found_scripts'])}")
        print(f"   Missing scripts: {len(analysis['missing_scripts'])}")
        
        if analysis['found_scripts']:
            print(f"\nFound old scripts:")
            for script in analysis['found_scripts']:
                print(f"   - {script['name']}: {script['description']}")
        
        print(f"\nMigration recommendations:")
        for rec in analysis['migration_recommendations']:
            print(f"   {rec}")
        
        # Create migration guide
        helper.create_migration_guide(args.guide)
    
    if args.test or args.migrate:
        print("\nTesting unified trainer...")
        success = helper.test_unified_trainer()
        if not success:
            print("Migration test failed - please fix issues before proceeding")
            sys.exit(1)
    
    if args.archive or args.migrate:
        print("\nArchiving old training scripts...")
        helper.archive_old_scripts()
    
    if args.migrate:
        print("\nMigration completed successfully!")
        print("See TRAINING_MIGRATION_GUIDE.md for detailed instructions")
        print("You can now use: python -m src.training.unified_trainer")


if __name__ == "__main__":
    main()
