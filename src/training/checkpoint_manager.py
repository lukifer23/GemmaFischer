#!/usr/bin/env python3
"""
Robust Checkpoint Management System for ChessGemma Training

Provides comprehensive checkpointing capabilities:
- Automatic checkpoint discovery and validation
- Training resume from latest checkpoint
- Progress tracking and metadata storage
- Checkpoint cleanup and optimization
- Recovery from training interruptions
- Multi-expert checkpoint management
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import hashlib
import threading

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(project_root))

try:
    from ..utils.logging_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """Metadata for a training checkpoint."""
    checkpoint_id: str
    expert_name: str
    step: int
    epoch: float
    global_step: int
    training_loss: float
    eval_loss: Optional[float] = None
    learning_rate: float = 0.0
    timestamp: str = ""
    duration_seconds: float = 0.0
    model_config: Dict[str, Any] = field(default_factory=dict)
    training_config: Dict[str, Any] = field(default_factory=dict)
    dataset_info: Dict[str, Any] = field(default_factory=dict)
    system_info: Dict[str, Any] = field(default_factory=dict)
    validation_metrics: Dict[str, Any] = field(default_factory=dict)
    is_best_checkpoint: bool = False
    checkpoint_hash: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CheckpointMetadata:
        """Create CheckpointMetadata from dictionary."""
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'checkpoint_id': self.checkpoint_id,
            'expert_name': self.expert_name,
            'step': self.step,
            'epoch': self.epoch,
            'global_step': self.global_step,
            'training_loss': self.training_loss,
            'eval_loss': self.eval_loss,
            'learning_rate': self.learning_rate,
            'timestamp': self.timestamp,
            'duration_seconds': self.duration_seconds,
            'model_config': self.model_config,
            'training_config': self.training_config,
            'dataset_info': self.dataset_info,
            'system_info': self.system_info,
            'validation_metrics': self.validation_metrics,
            'is_best_checkpoint': self.is_best_checkpoint,
            'checkpoint_hash': self.checkpoint_hash
        }


@dataclass
class TrainingProgress:
    """Tracks overall training progress across checkpoints."""
    expert_name: str
    total_steps: int
    completed_steps: int
    start_time: str
    last_checkpoint_time: str
    best_eval_loss: Optional[float] = None
    best_checkpoint_id: Optional[str] = None
    checkpoints_created: List[str] = field(default_factory=list)
    interruptions: List[Dict[str, Any]] = field(default_factory=list)
    estimated_completion_time: Optional[str] = None

    def update_progress(self, checkpoint: CheckpointMetadata) -> None:
        """Update progress based on checkpoint data."""
        self.completed_steps = checkpoint.global_step
        self.last_checkpoint_time = checkpoint.timestamp

        if checkpoint.eval_loss is not None:
            if self.best_eval_loss is None or checkpoint.eval_loss < self.best_eval_loss:
                self.best_eval_loss = checkpoint.eval_loss
                self.best_checkpoint_id = checkpoint.checkpoint_id
                checkpoint.is_best_checkpoint = True

        # Estimate completion time
        if self.completed_steps > 0 and self.total_steps > self.completed_steps:
            elapsed_time = datetime.now() - datetime.fromisoformat(self.start_time)
            remaining_steps = self.total_steps - self.completed_steps
            avg_time_per_step = elapsed_time.total_seconds() / self.completed_steps
            estimated_remaining = remaining_steps * avg_time_per_step
            completion_time = datetime.now() + timedelta(seconds=estimated_remaining)
            self.estimated_completion_time = completion_time.isoformat()


class CheckpointManager:
    """Comprehensive checkpoint management system."""

    def __init__(self, checkpoints_dir: Path, max_checkpoints_per_expert: int = 5):
        self.checkpoints_dir = checkpoints_dir
        self.max_checkpoints_per_expert = max_checkpoints_per_expert
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        # Thread safety
        self._lock = threading.RLock()

        # Progress tracking
        self.training_progress: Dict[str, TrainingProgress] = {}

        # Initialize from existing checkpoints
        self._load_existing_checkpoints()

    def _load_existing_checkpoints(self) -> None:
        """Load existing checkpoints and their metadata."""
        logger.info("Loading existing checkpoints...")

        for expert_dir in self.checkpoints_dir.iterdir():
            if not expert_dir.is_dir():
                continue

            expert_name = expert_dir.name
            self.training_progress[expert_name] = TrainingProgress(
                expert_name=expert_name,
                total_steps=0,  # Will be updated when training starts
                completed_steps=0,
                start_time=datetime.now().isoformat(),
                last_checkpoint_time=""
            )

            # Load checkpoints for this expert
            for checkpoint_dir in expert_dir.iterdir():
                if not checkpoint_dir.is_dir():
                    continue

                metadata_file = checkpoint_dir / "checkpoint_metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            data = json.load(f)
                            metadata = CheckpointMetadata.from_dict(data)

                            # Update progress
                            progress = self.training_progress[expert_name]
                            progress.completed_steps = max(progress.completed_steps, metadata.global_step)
                            progress.last_checkpoint_time = metadata.timestamp
                            progress.checkpoints_created.append(metadata.checkpoint_id)

                            if metadata.is_best_checkpoint:
                                progress.best_eval_loss = metadata.eval_loss
                                progress.best_checkpoint_id = metadata.checkpoint_id

                    except Exception as e:
                        logger.warning(f"Failed to load metadata for {checkpoint_dir}: {e}")

    def create_checkpoint(self, expert_name: str, step: int, epoch: float,
                         global_step: int, training_loss: float,
                         eval_loss: Optional[float] = None,
                         learning_rate: float = 0.0,
                         model_config: Optional[Dict[str, Any]] = None,
                         training_config: Optional[Dict[str, Any]] = None,
                         dataset_info: Optional[Dict[str, Any]] = None,
                         validation_metrics: Optional[Dict[str, Any]] = None,
                         trainer_state: Optional[Dict[str, Any]] = None) -> Tuple[Path, CheckpointMetadata]:
        """Create a new checkpoint with full metadata."""

        with self._lock:
            # Generate checkpoint ID
            timestamp = datetime.now()
            checkpoint_id = f"{expert_name}_{timestamp.strftime('%Y%m%d_%H%M%S')}_{global_step}"

            # Create checkpoint directory
            checkpoint_dir = self.checkpoints_dir / expert_name / checkpoint_id
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # Calculate duration
            progress = self.training_progress.get(expert_name)
            duration_seconds = 0.0
            if progress and progress.start_time:
                start_time = datetime.fromisoformat(progress.start_time)
                duration_seconds = (timestamp - start_time).total_seconds()

            # Create metadata
            metadata = CheckpointMetadata(
                checkpoint_id=checkpoint_id,
                expert_name=expert_name,
                step=step,
                epoch=epoch,
                global_step=global_step,
                training_loss=training_loss,
                eval_loss=eval_loss,
                learning_rate=learning_rate,
                timestamp=timestamp.isoformat(),
                duration_seconds=duration_seconds,
                model_config=model_config or {},
                training_config=training_config or {},
                dataset_info=dataset_info or {},
                system_info=self._get_system_info(),
                validation_metrics=validation_metrics or {}
            )

            # Calculate checkpoint hash for integrity checking
            metadata.checkpoint_hash = self._calculate_checkpoint_hash(checkpoint_dir)

            # Update progress
            if progress:
                progress.update_progress(metadata)
                progress.checkpoints_created.append(checkpoint_id)

            # Save metadata
            metadata_file = checkpoint_dir / "checkpoint_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2, default=str)

            # Save trainer state if provided
            if trainer_state:
                trainer_state_file = checkpoint_dir / "trainer_state.json"
                with open(trainer_state_file, 'w') as f:
                    json.dump(trainer_state, f, indent=2, default=str)

            logger.info(f"💾 Created checkpoint: {checkpoint_dir}")
            logger.info(f"   Step: {global_step}, Loss: {training_loss:.4f}")

            # Cleanup old checkpoints
            self._cleanup_old_checkpoints(expert_name)

            return checkpoint_dir, metadata

    def find_latest_checkpoint(self, expert_name: str) -> Optional[Tuple[Path, CheckpointMetadata]]:
        """Find the latest checkpoint for an expert."""

        with self._lock:
            expert_dir = self.checkpoints_dir / expert_name
            if not expert_dir.exists():
                return None

            latest_checkpoint = None
            latest_time = None

            for checkpoint_dir in expert_dir.iterdir():
                if not checkpoint_dir.is_dir():
                    continue

                metadata_file = checkpoint_dir / "checkpoint_metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            data = json.load(f)
                            metadata = CheckpointMetadata.from_dict(data)

                            checkpoint_time = datetime.fromisoformat(metadata.timestamp)
                            if latest_time is None or checkpoint_time > latest_time:
                                latest_time = checkpoint_time
                                latest_checkpoint = (checkpoint_dir, metadata)

                    except Exception as e:
                        logger.warning(f"Failed to read metadata for {checkpoint_dir}: {e}")

            return latest_checkpoint

    def resume_from_checkpoint(self, expert_name: str) -> Optional[Tuple[Path, CheckpointMetadata, Dict[str, Any]]]:
        """Find the best checkpoint to resume training from."""

        with self._lock:
            latest = self.find_latest_checkpoint(expert_name)
            if not latest:
                return None

            checkpoint_dir, metadata = latest

            # Load trainer state if available
            trainer_state = {}
            trainer_state_file = checkpoint_dir / "trainer_state.json"
            if trainer_state_file.exists():
                try:
                    with open(trainer_state_file, 'r') as f:
                        trainer_state = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load trainer state: {e}")

            logger.info(f"📂 Resuming from checkpoint: {checkpoint_dir}")
            logger.info(f"   Global step: {metadata.global_step}, Loss: {metadata.training_loss:.4f}")

            return checkpoint_dir, metadata, trainer_state

    def validate_checkpoint(self, checkpoint_dir: Path) -> bool:
        """Validate checkpoint integrity."""

        metadata_file = checkpoint_dir / "checkpoint_metadata.json"
        if not metadata_file.exists():
            return False

        try:
            with open(metadata_file, 'r') as f:
                data = json.load(f)
                metadata = CheckpointMetadata.from_dict(data)

            # Check if required files exist
            required_files = ['adapter_model.bin', 'adapter_config.json']
            for filename in required_files:
                if not (checkpoint_dir / filename).exists():
                    logger.warning(f"Missing required file: {filename}")
                    return False

            # Verify hash if available
            if metadata.checkpoint_hash:
                current_hash = self._calculate_checkpoint_hash(checkpoint_dir)
                if current_hash != metadata.checkpoint_hash:
                    logger.warning("Checkpoint hash mismatch - possible corruption")
                    return False

            return True

        except Exception as e:
            logger.warning(f"Checkpoint validation failed: {e}")
            return False

    def _cleanup_old_checkpoints(self, expert_name: str) -> None:
        """Clean up old checkpoints to save disk space."""

        expert_dir = self.checkpoints_dir / expert_name
        if not expert_dir.exists():
            return

        # Get all checkpoints for this expert
        checkpoints = []
        for checkpoint_dir in expert_dir.iterdir():
            if not checkpoint_dir.is_dir():
                continue

            metadata_file = checkpoint_dir / "checkpoint_metadata.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        data = json.load(f)
                        metadata = CheckpointMetadata.from_dict(data)
                        checkpoints.append((checkpoint_dir, metadata))
                except:
                    continue

        if len(checkpoints) <= self.max_checkpoints_per_expert:
            return

        # Sort by timestamp (newest first)
        checkpoints.sort(key=lambda x: x[1].timestamp, reverse=True)

        # Keep best checkpoint and most recent ones
        to_keep = set()

        # Always keep the best checkpoint
        best_checkpoint = None
        best_loss = float('inf')
        for checkpoint_dir, metadata in checkpoints:
            if metadata.eval_loss is not None and metadata.eval_loss < best_loss:
                best_loss = metadata.eval_loss
                best_checkpoint = checkpoint_dir

        if best_checkpoint:
            to_keep.add(best_checkpoint)

        # Keep the most recent checkpoints
        for checkpoint_dir, metadata in checkpoints[:self.max_checkpoints_per_expert]:
            to_keep.add(checkpoint_dir)

        # Remove old checkpoints
        for checkpoint_dir, metadata in checkpoints:
            if checkpoint_dir not in to_keep:
                try:
                    shutil.rmtree(checkpoint_dir)
                    logger.info(f"🗑️  Removed old checkpoint: {checkpoint_dir}")
                except Exception as e:
                    logger.warning(f"Failed to remove checkpoint {checkpoint_dir}: {e}")

    def _calculate_checkpoint_hash(self, checkpoint_dir: Path) -> str:
        """Calculate hash of checkpoint files for integrity checking."""
        hasher = hashlib.sha256()

        # Hash key files
        key_files = ['adapter_model.bin', 'adapter_config.json', 'checkpoint_metadata.json']
        for filename in key_files:
            file_path = checkpoint_dir / filename
            if file_path.exists():
                try:
                    with open(file_path, 'rb') as f:
                        while chunk := f.read(8192):
                            hasher.update(chunk)
                except:
                    pass

        return hasher.hexdigest()

    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for checkpoint metadata."""
        try:
            import platform
            import psutil

            return {
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'cpu_count': psutil.cpu_count(),
                'memory_gb': psutil.virtual_memory().total / (1024**3),
                'gpu_available': False,  # MPS detection would go here
                'disk_free_gb': psutil.disk_usage('/').free / (1024**3)
            }
        except:
            return {'error': 'Failed to collect system info'}

    def get_training_progress(self, expert_name: str) -> Optional[TrainingProgress]:
        """Get training progress for an expert."""
        return self.training_progress.get(expert_name)

    def list_checkpoints(self, expert_name: Optional[str] = None) -> Dict[str, List[CheckpointMetadata]]:
        """List all checkpoints, optionally filtered by expert."""

        result = {}

        if expert_name:
            experts = [expert_name]
        else:
            experts = [d.name for d in self.checkpoints_dir.iterdir() if d.is_dir()]

        for exp_name in experts:
            expert_dir = self.checkpoints_dir / exp_name
            if not expert_dir.exists():
                continue

            checkpoints = []
            for checkpoint_dir in expert_dir.iterdir():
                if not checkpoint_dir.is_dir():
                    continue

                metadata_file = checkpoint_dir / "checkpoint_metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            data = json.load(f)
                            metadata = CheckpointMetadata.from_dict(data)
                            checkpoints.append(metadata)
                    except:
                        continue

            # Sort by timestamp (newest first)
            checkpoints.sort(key=lambda c: c.timestamp, reverse=True)
            result[exp_name] = checkpoints

        return result

    def export_checkpoint_summary(self, output_file: Path) -> None:
        """Export a summary of all checkpoints."""

        summary = {
            'generated_at': datetime.now().isoformat(),
            'checkpoints_dir': str(self.checkpoints_dir),
            'experts': {}
        }

        all_checkpoints = self.list_checkpoints()

        for expert_name, checkpoints in all_checkpoints.items():
            progress = self.training_progress.get(expert_name)

            summary['experts'][expert_name] = {
                'total_checkpoints': len(checkpoints),
                'latest_checkpoint': checkpoints[0].to_dict() if checkpoints else None,
                'best_checkpoint': None,
                'training_progress': progress.__dict__ if progress else None,
                'checkpoint_list': [c.to_dict() for c in checkpoints[:10]]  # Last 10
            }

            # Find best checkpoint
            best_loss = float('inf')
            best_checkpoint = None
            for checkpoint in checkpoints:
                if checkpoint.eval_loss is not None and checkpoint.eval_loss < best_loss:
                    best_loss = checkpoint.eval_loss
                    best_checkpoint = checkpoint

            if best_checkpoint:
                summary['experts'][expert_name]['best_checkpoint'] = best_checkpoint.to_dict()

        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"📊 Checkpoint summary exported to: {output_file}")


def main():
    """Command-line interface for checkpoint management."""

    parser = argparse.ArgumentParser(description="Checkpoint Management for ChessGemma Training")
    parser.add_argument('--checkpoints-dir', type=Path, default=Path('checkpoints'),
                       help='Directory containing checkpoints')
    parser.add_argument('--command', choices=['list', 'validate', 'cleanup', 'summary', 'resume'],
                       default='list', help='Command to execute')
    parser.add_argument('--expert', help='Expert name (for specific operations)')
    parser.add_argument('--output', type=Path, help='Output file for summary')

    args = parser.parse_args()

    manager = CheckpointManager(args.checkpoints_dir)

    if args.command == 'list':
        checkpoints = manager.list_checkpoints(args.expert)
        for expert_name, expert_checkpoints in checkpoints.items():
            print(f"\n=== {expert_name} ===")
            for i, checkpoint in enumerate(expert_checkpoints[:5]):  # Show first 5
                print(f"  {i+1}. {checkpoint.checkpoint_id} (Step {checkpoint.global_step}, Loss: {checkpoint.training_loss:.4f})")

    elif args.command == 'validate':
        if not args.expert:
            print("Error: --expert required for validation")
            return

        latest = manager.find_latest_checkpoint(args.expert)
        if latest:
            checkpoint_dir, metadata = latest
            is_valid = manager.validate_checkpoint(checkpoint_dir)
            print(f"Checkpoint {checkpoint_dir.name}: {'VALID' if is_valid else 'INVALID'}")
        else:
            print(f"No checkpoints found for expert: {args.expert}")

    elif args.command == 'cleanup':
        if args.expert:
            manager._cleanup_old_checkpoints(args.expert)
            print(f"Cleaned up old checkpoints for: {args.expert}")
        else:
            for expert_name in manager.training_progress.keys():
                manager._cleanup_old_checkpoints(expert_name)
            print("Cleaned up old checkpoints for all experts")

    elif args.command == 'summary':
        output_file = args.output or Path('checkpoint_summary.json')
        manager.export_checkpoint_summary(output_file)

    elif args.command == 'resume':
        if not args.expert:
            print("Error: --expert required for resume")
            return

        resume_data = manager.resume_from_checkpoint(args.expert)
        if resume_data:
            checkpoint_dir, metadata, trainer_state = resume_data
            print(f"Resume from: {checkpoint_dir}")
            print(f"Global step: {metadata.global_step}")
            print(f"Training loss: {metadata.training_loss:.4f}")
            if metadata.eval_loss:
                print(f"Eval loss: {metadata.eval_loss:.4f}")
        else:
            print(f"No checkpoints found for expert: {args.expert}")


if __name__ == '__main__':
    main()
