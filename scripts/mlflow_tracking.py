#!/usr/bin/env python3
"""MLflow integration for ChessGemma experiment tracking.

This script provides utilities for:
- Starting MLflow experiments
- Logging training metrics
- Tracking hyperparameters
- Comparing model versions
"""

import mlflow
import mlflow.pytorch
from pathlib import Path
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional


class ChessGemmaTracker:
    """MLflow tracker for ChessGemma experiments."""

    def __init__(self, experiment_name: str = "ChessGemma"):
        """Initialize MLflow experiment."""
        mlflow.set_experiment(experiment_name)
        print(f"MLflow experiment set to: {experiment_name}")

    def start_run(self, run_name: Optional[str] = None) -> str:
        """Start a new MLflow run."""
        if run_name is None:
            run_name = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        mlflow.start_run(run_name=run_name)
        run_id = mlflow.active_run().info.run_id
        print(f"Started MLflow run: {run_name} (ID: {run_id})")
        return run_id

    def log_config(self, config: Dict[str, Any]):
        """Log training configuration as parameters."""
        # Flatten nested config for MLflow
        def flatten_config(d: Dict, prefix: str = '') -> Dict:
            flat = {}
            for key, value in d.items():
                new_key = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    flat.update(flatten_config(value, new_key))
                else:
                    flat[new_key] = value
            return flat

        flat_config = flatten_config(config)
        for key, value in flat_config.items():
            mlflow.log_param(key, value)

        print(f"Logged {len(flat_config)} configuration parameters")

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log training metrics."""
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value, step=step)

    def log_model(self, model, model_name: str = "chess_gemma"):
        """Log the trained model."""
        try:
            mlflow.pytorch.log_model(model, model_name)
            print(f"Model logged as: {model_name}")
        except Exception as e:
            print(f"Could not log model: {e}")

    def log_artifacts(self, artifact_dir: str):
        """Log training artifacts (checkpoints, logs, etc.)."""
        if os.path.exists(artifact_dir):
            mlflow.log_artifacts(artifact_dir, "training_artifacts")
            print(f"Artifacts logged from: {artifact_dir}")

    def log_chess_evaluation(self, eval_results: Dict[str, Any]):
        """Log chess-specific evaluation results."""
        # Log main metrics
        mlflow.log_metric("chess.move_syntax_accuracy", eval_results.get('average_move_syntax_accuracy', 0))
        mlflow.log_metric("chess.relevance_score", eval_results.get('average_chess_relevance', 0))
        mlflow.log_metric("chess.total_moves", eval_results.get('total_moves_mentioned', 0))
        mlflow.log_metric("chess.valid_moves", eval_results.get('total_valid_moves', 0))

        # Log detailed results as JSON artifact
        eval_file = "chess_evaluation.json"
        with open(eval_file, 'w') as f:
            json.dump(eval_results, f, indent=2)
        mlflow.log_artifact(eval_file, "evaluations")

    def end_run(self):
        """End the current MLflow run."""
        mlflow.end_run()
        print("MLflow run ended")


def load_training_metrics(log_file: str) -> Dict[str, Any]:
    """Load training metrics from enhanced log file."""
    if not os.path.exists(log_file):
        return {}

    metrics = {}
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()

        if lines:
            # Get latest metrics
            latest = json.loads(lines[-1])
            metrics.update({
                'final_loss': latest.get('loss'),
                'final_eval_loss': latest.get('eval_loss'),
                'training_time': latest.get('time_elapsed', 0),
                'cpu_percent': latest.get('cpu_percent', 0),
                'progress_percent': latest.get('progress_percent', 0)
            })

        # Calculate training statistics
        all_losses = []
        all_eval_losses = []
        for line in lines:
            data = json.loads(line)
            if 'loss' in data and data['loss'] is not None:
                all_losses.append(data['loss'])
            if 'eval_loss' in data and data['eval_loss'] is not None:
                all_eval_losses.append(data['eval_loss'])

        if all_losses:
            metrics['avg_training_loss'] = sum(all_losses) / len(all_losses)
            metrics['min_training_loss'] = min(all_losses)
            metrics['max_training_loss'] = max(all_losses)

        if all_eval_losses:
            metrics['avg_eval_loss'] = sum(all_eval_losses) / len(all_eval_losses)
            metrics['min_eval_loss'] = min(all_eval_losses)

    except Exception as e:
        print(f"Could not load training metrics: {e}")

    return metrics


def main():
    """Example usage of ChessGemma MLflow tracking."""
    tracker = ChessGemmaTracker()

    # Example: Log a completed training run
    tracker.start_run("example_training_run")

    # Load config
    config_path = Path(__file__).parent.parent / 'configs' / 'lora_full.yaml'
    if config_path.exists():
        try:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            tracker.log_config(config)
        except Exception as e:
            print(f"Could not load config: {e}")

    # Load training metrics
    log_path = Path(__file__).parent.parent / 'checkpoints' / 'lora_full' / 'enhanced_train_log.jsonl'
    if log_path.exists():
        metrics = load_training_metrics(str(log_path))
        tracker.log_metrics(metrics)

    # Load chess evaluation results
    eval_path = Path(__file__).parent.parent / 'chess_evaluation_results.json'
    if eval_path.exists():
        try:
            with open(eval_path, 'r') as f:
                eval_results = json.load(f)
            tracker.log_chess_evaluation(eval_results)
        except Exception as e:
            print(f"Could not load evaluation results: {e}")

    tracker.end_run()
    print("MLflow logging complete!")


if __name__ == '__main__':
    main()
