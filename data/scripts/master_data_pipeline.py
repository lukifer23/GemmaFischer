#!/usr/bin/env python3
"""
Master Chess Data Pipeline

Orchestrates the complete data collection, processing, and validation pipeline:
1. Downloads raw datasets from multiple sources
2. Processes and converts data to training formats
3. Generates additional training data (CoT, visual)
4. Validates and filters for quality
5. Creates comprehensive training datasets

This is the main entry point for building the complete GemmaFischer dataset.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import time


class MasterDataPipeline:
    """Orchestrate the complete chess data pipeline."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.scripts_dir = data_dir / "scripts"
        self.datasets_dir = data_dir / "datasets"
        self.validation_dir = data_dir / "validation"
        
        # Ensure directories exist
        for dir_path in [self.datasets_dir, self.validation_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Pipeline configuration
        self.config = {
            'lichess_puzzles': {
                'enabled': True,
                'max_puzzles': 1000000,
                'min_rating': 1000,
                'max_rating': 2000
            },
            'lichess_games': {
                'enabled': False,  # Very large, enable only if needed
                'min_rating': 1800,
                'max_games': 10000
            },
            'cot_data': {
                'enabled': True,
                'tactical_examples': 1000,
                'positional_examples': 500,
                'endgame_examples': 300,
                'opening_examples': 400
            },
            'visual_data': {
                'enabled': True,
                'board_positions': 1000,
                'piece_recognition': 500,
                'board_detection': 300,
                'synthetic_boards': 200
            },
            'opening_theory': {
                'enabled': True
            },
            'endgame_data': {
                'enabled': True
            },
            'historical_games': {
                'enabled': True
            },
            'validation': {
                'enabled': True,
                'min_quality_score': 5.0
            }
        }
    
    def run_complete_pipeline(self, skip_download: bool = False) -> Dict[str, Any]:
        """Run the complete data pipeline."""
        print("🚀 Starting Master Chess Data Pipeline")
        print("=" * 60)
        
        start_time = time.time()
        pipeline_results = {
            'start_time': datetime.now().isoformat(),
            'steps_completed': [],
            'steps_failed': [],
            'datasets_created': [],
            'total_processing_time': 0
        }
        
        try:
            # Step 1: Download raw data
            if not skip_download:
                print("\n📥 Step 1: Downloading Raw Data")
                print("-" * 40)
                self._run_download_step(pipeline_results)
            else:
                print("\n⏭️  Step 1: Skipping Download (--skip-download)")
                pipeline_results['steps_completed'].append('download_skipped')
            
            # Step 2: Process Lichess data
            print("\n🔄 Step 2: Processing Lichess Data")
            print("-" * 40)
            self._run_lichess_processing_step(pipeline_results)
            
            # Step 3: Generate CoT data
            print("\n🧠 Step 3: Generating Chain-of-Thought Data")
            print("-" * 40)
            self._run_cot_generation_step(pipeline_results)
            
            # Step 4: Generate visual data
            print("\n👁️  Step 4: Generating Visual Data")
            print("-" * 40)
            self._run_visual_generation_step(pipeline_results)
            
            # Step 5: Validate datasets
            print("\n✅ Step 5: Validating Datasets")
            print("-" * 40)
            self._run_validation_step(pipeline_results)
            
            # Step 6: Create final combined dataset
            print("\n🎯 Step 6: Creating Final Combined Dataset")
            print("-" * 40)
            self._run_combination_step(pipeline_results)
            
            # Calculate total time
            pipeline_results['total_processing_time'] = time.time() - start_time
            pipeline_results['end_time'] = datetime.now().isoformat()
            
            # Save pipeline results
            self._save_pipeline_results(pipeline_results)
            
            print("\n🎉 Master Data Pipeline Complete!")
            print("=" * 60)
            print(f"⏱️  Total time: {pipeline_results['total_processing_time']:.1f} seconds")
            print(f"✅ Steps completed: {len(pipeline_results['steps_completed'])}")
            print(f"❌ Steps failed: {len(pipeline_results['steps_failed'])}")
            print(f"📊 Datasets created: {len(pipeline_results['datasets_created'])}")
            
            return pipeline_results
            
        except Exception as e:
            print(f"\n❌ Pipeline failed: {e}")
            pipeline_results['steps_failed'].append(f'pipeline_error: {e}')
            pipeline_results['total_processing_time'] = time.time() - start_time
            return pipeline_results
    
    def _run_download_step(self, results: Dict[str, Any]):
        """Run the data download step."""
        try:
            cmd = [
                sys.executable, str(self.scripts_dir / "download_data.py"),
                "--data_dir", str(self.data_dir.parent),  # Go up one level to the project root
                "--max_puzzles", str(self.config['lichess_puzzles']['max_puzzles'])
            ]
            
            if self.config['lichess_games']['enabled']:
                cmd.append("--include_games")
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.data_dir.parent)
            
            if result.returncode == 0:
                print("   ✅ Download step completed successfully")
                results['steps_completed'].append('download')
            else:
                print(f"   ❌ Download step failed: {result.stderr}")
                results['steps_failed'].append('download')
                
        except Exception as e:
            print(f"   ❌ Download step error: {e}")
            results['steps_failed'].append(f'download_error: {e}')
    
    def _run_lichess_processing_step(self, results: Dict[str, Any]):
        """Run the Lichess data processing step."""
        try:
            # Process puzzles
            if self.config['lichess_puzzles']['enabled']:
                cmd = [
                    sys.executable, str(self.scripts_dir / "process_lichess.py"),
                    "--data_dir", str(self.data_dir.parent),  # Go up one level to the project root
                    "--type", "puzzles",
                    "--min_rating", str(self.config['lichess_puzzles']['min_rating']),
                    "--max_rating", str(self.config['lichess_puzzles']['max_rating']),
                    "--max_items", str(self.config['lichess_puzzles']['max_puzzles'])
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.data_dir.parent)
                
                if result.returncode == 0:
                    print("   ✅ Lichess puzzles processed successfully")
                    results['steps_completed'].append('lichess_puzzles')
                    results['datasets_created'].append('lichess_puzzles')
                else:
                    print(f"   ❌ Lichess puzzles processing failed: {result.stderr}")
                    results['steps_failed'].append('lichess_puzzles')
            
            # Process games (if enabled)
            if self.config['lichess_games']['enabled']:
                cmd = [
                    sys.executable, str(self.scripts_dir / "process_lichess.py"),
                    "--data_dir", str(self.data_dir.parent),  # Go up one level to the project root
                    "--type", "games",
                    "--min_rating", str(self.config['lichess_games']['min_rating']),
                    "--max_items", str(self.config['lichess_games']['max_games'])
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.data_dir.parent)
                
                if result.returncode == 0:
                    print("   ✅ Lichess games processed successfully")
                    results['steps_completed'].append('lichess_games')
                    results['datasets_created'].append('lichess_games')
                else:
                    print(f"   ❌ Lichess games processing failed: {result.stderr}")
                    results['steps_failed'].append('lichess_games')
                    
        except Exception as e:
            print(f"   ❌ Lichess processing error: {e}")
            results['steps_failed'].append(f'lichess_processing_error: {e}')
    
    def _run_cot_generation_step(self, results: Dict[str, Any]):
        """Run the chain-of-thought data generation step."""
        if not self.config['cot_data']['enabled']:
            print("   ⏭️  CoT generation disabled")
            return
        
        try:
            cmd = [
                sys.executable, str(self.scripts_dir / "generate_cot_data.py"),
                "--data_dir", str(self.data_dir.parent),  # Go up one level to the project root
                "--type", "all",
                "--num_examples", str(sum([
                    self.config['cot_data']['tactical_examples'],
                    self.config['cot_data']['positional_examples'],
                    self.config['cot_data']['endgame_examples'],
                    self.config['cot_data']['opening_examples']
                ]))
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.data_dir.parent)
            
            if result.returncode == 0:
                print("   ✅ CoT data generated successfully")
                results['steps_completed'].append('cot_generation')
                results['datasets_created'].append('cot_reasoning')
            else:
                print(f"   ❌ CoT generation failed: {result.stderr}")
                results['steps_failed'].append('cot_generation')
                
        except Exception as e:
            print(f"   ❌ CoT generation error: {e}")
            results['steps_failed'].append(f'cot_generation_error: {e}')
    
    def _run_visual_generation_step(self, results: Dict[str, Any]):
        """Run the visual data generation step."""
        if not self.config['visual_data']['enabled']:
            print("   ⏭️  Visual generation disabled")
            return
        
        try:
            cmd = [
                sys.executable, str(self.scripts_dir / "create_visual_data.py"),
                "--data_dir", str(self.data_dir.parent),  # Go up one level to the project root
                "--type", "all",
                "--num_examples", str(sum([
                    self.config['visual_data']['board_positions'],
                    self.config['visual_data']['piece_recognition'],
                    self.config['visual_data']['board_detection'],
                    self.config['visual_data']['synthetic_boards']
                ]))
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.data_dir.parent)
            
            if result.returncode == 0:
                print("   ✅ Visual data generated successfully")
                results['steps_completed'].append('visual_generation')
                results['datasets_created'].append('visual_training')
            else:
                print(f"   ❌ Visual generation failed: {result.stderr}")
                results['steps_failed'].append('visual_generation')
                
        except Exception as e:
            print(f"   ❌ Visual generation error: {e}")
            results['steps_failed'].append(f'visual_generation_error: {e}')
    
    def _run_validation_step(self, results: Dict[str, Any]):
        """Run the data validation step."""
        if not self.config['validation']['enabled']:
            print("   ⏭️  Validation disabled")
            return
        
        try:
            cmd = [
                sys.executable, str(self.scripts_dir / "validate_data.py"),
                "--data_dir", str(self.data_dir.parent),  # Go up one level to the project root
                "--validate_all",
                "--min_quality", str(self.config['validation']['min_quality_score'])
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.data_dir.parent)
            
            if result.returncode == 0:
                print("   ✅ Data validation completed successfully")
                results['steps_completed'].append('validation')
            else:
                print(f"   ❌ Validation failed: {result.stderr}")
                results['steps_failed'].append('validation')
                
        except Exception as e:
            print(f"   ❌ Validation error: {e}")
            results['steps_failed'].append(f'validation_error: {e}')
    
    def _run_combination_step(self, results: Dict[str, Any]):
        """Run the final dataset combination step."""
        try:
            # Find all validated datasets
            dataset_files = list(self.datasets_dir.glob("*.jsonl"))
            
            if not dataset_files:
                print("   ❌ No datasets found to combine")
                results['steps_failed'].append('combination_no_datasets')
                return
            
            # Combine datasets
            combined_data = []
            dataset_stats = {}
            
            for dataset_file in dataset_files:
                print(f"   📊 Processing: {dataset_file.name}")
                
                count = 0
                with open(dataset_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            combined_data.append(json.loads(line))
                            count += 1
                
                dataset_stats[dataset_file.name] = count
                print(f"      Added {count:,} examples")
            
            # Shuffle combined data
            import random
            random.shuffle(combined_data)
            
            # Save combined dataset
            combined_file = self.datasets_dir / "gemmafischer_combined_dataset.jsonl"
            with open(combined_file, 'w', encoding='utf-8') as f:
                for example in combined_data:
                    f.write(json.dumps(example, ensure_ascii=False) + '\n')
            
            # Create metadata
            metadata = {
                'total_examples': len(combined_data),
                'source_datasets': dataset_stats,
                'creation_date': datetime.now().isoformat(),
                'description': 'Combined GemmaFischer training dataset',
                'pipeline_version': '1.0'
            }
            
            metadata_file = combined_file.with_suffix('.metadata.json')
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            print(f"   ✅ Combined dataset created: {combined_file}")
            print(f"   📊 Total examples: {len(combined_data):,}")
            print(f"   📁 Metadata: {metadata_file}")
            
            results['steps_completed'].append('combination')
            results['datasets_created'].append('combined_dataset')
            
        except Exception as e:
            print(f"   ❌ Combination error: {e}")
            results['steps_failed'].append(f'combination_error: {e}')
    
    def _save_pipeline_results(self, results: Dict[str, Any]):
        """Save pipeline results to file."""
        results_file = self.data_dir / "pipeline_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"📁 Pipeline results saved: {results_file}")
    
    def create_training_config(self, output_file: Path):
        """Create training configuration for the generated datasets."""
        print(f"📝 Creating training configuration...")
        
        # Find available datasets
        dataset_files = list(self.datasets_dir.glob("*.jsonl"))
        
        config = {
            'model': {
                'pretrained_model_path': "models/unsloth-gemma-3-270m-it/...",
                'max_seq_length': 2048,
                'dtype': "float16"
            },
            'training': {
                'output_dir': "checkpoints/gemmafischer_combined",
                'per_device_train_batch_size': 4,
                'gradient_accumulation_steps': 8,
                'learning_rate': 1e-4,
                'max_steps': 2000,
                'logging_steps': 50,
                'save_steps': 200,
                'fp16': False
            },
            'lora': {
                'r': 32,
                'lora_alpha': 64,
                'target_modules': ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
                'dropout': 0.05
            },
            'datasets': {
                'primary': str(self.datasets_dir / "gemmafischer_combined_dataset.jsonl"),
                'available_datasets': [str(f) for f in dataset_files]
            },
            'data_quality': {
                'min_quality_score': self.config['validation']['min_quality_score'],
                'validation_enabled': self.config['validation']['enabled']
            },
            'generation_date': datetime.now().isoformat(),
            'description': 'GemmaFischer training configuration for combined dataset'
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Training config created: {output_file}")


def main():
    """Main entry point for the master data pipeline."""
    parser = argparse.ArgumentParser(description="Master Chess Data Pipeline")
    parser.add_argument("--data_dir", default="data", help="Data directory path")
    parser.add_argument("--skip-download", action="store_true", help="Skip the download step")
    parser.add_argument("--create-config", action="store_true", help="Create training configuration")
    parser.add_argument("--config-file", default="src/training/configs/gemmafischer_combined.yaml", 
                       help="Training configuration output file")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    pipeline = MasterDataPipeline(data_dir)
    
    # Run the complete pipeline
    results = pipeline.run_complete_pipeline(skip_download=args.skip_download)
    
    # Create training configuration if requested
    if args.create_config:
        config_file = Path(args.config_file)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        pipeline.create_training_config(config_file)
    
    # Print final summary
    print("\n📋 Final Summary")
    print("=" * 60)
    print(f"✅ Successful steps: {len(results['steps_completed'])}")
    print(f"❌ Failed steps: {len(results['steps_failed'])}")
    print(f"📊 Datasets created: {len(results['datasets_created'])}")
    print(f"⏱️  Total time: {results['total_processing_time']:.1f} seconds")
    
    if results['steps_failed']:
        print(f"\n⚠️  Failed steps:")
        for step in results['steps_failed']:
            print(f"   - {step}")
    
    if results['datasets_created']:
        print(f"\n📁 Created datasets:")
        for dataset in results['datasets_created']:
            print(f"   - {dataset}")


if __name__ == "__main__":
    main()
