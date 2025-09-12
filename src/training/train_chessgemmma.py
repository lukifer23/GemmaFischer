#!/usr/bin/env python3
"""
ChessGemma Unified Training Orchestrator

A comprehensive training orchestrator for the complete ChessGemma expert system.
Trains all experts (UCI, Tutor, Director) in sequence with intelligent checkpoint
management, MPS optimization, and comprehensive progress tracking.

Features:
- Sequential expert training with automatic checkpoint resume
- MPS memory optimization and dynamic batch sizing
- Comprehensive progress reporting and ETA calculations
- Intelligent resource management and cleanup
- Training validation and performance benchmarking
- Unified configuration management
- Error recovery and graceful failure handling
"""

from __future__ import annotations

import argparse
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import sys

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Import our enhanced training ecosystem
try:
    from .expert_trainer import ChessExpertTrainer
    from .checkpoint_manager import CheckpointManager
    from .mps_optimizer import MPSMemoryOptimizer, optimize_training_for_mps
    from ..utils.logging_config import get_logger, log_performance
    logger = get_logger(__name__)
except ImportError:
    # Fallback for basic functionality
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    log_performance = lambda func: func

    # Try to import what we can
    try:
        from .expert_trainer import ChessExpertTrainer
        EXPERT_TRAINER_AVAILABLE = True
    except ImportError as e:
        logger.warning(f"Expert trainer not available (likely missing torch): {e}")
        EXPERT_TRAINER_AVAILABLE = False
        ChessExpertTrainer = None

    try:
        from .checkpoint_manager import CheckpointManager
        CHECKPOINT_MANAGER_AVAILABLE = True
    except ImportError as e:
        logger.warning(f"Checkpoint manager not available: {e}")
        CHECKPOINT_MANAGER_AVAILABLE = False
        CheckpointManager = None

    try:
        from .mps_optimizer import MPSMemoryOptimizer
        MPS_OPTIMIZER_AVAILABLE = True
    except ImportError as e:
        logger.warning(f"MPS optimizer not available (likely missing torch): {e}")
        MPS_OPTIMIZER_AVAILABLE = False
        MPSMemoryOptimizer = None


@dataclass
class TrainingSession:
    """Represents a complete training session across all experts."""
    session_id: str
    start_time: str
    experts_to_train: List[str]
    completed_experts: List[str] = field(default_factory=list)
    failed_experts: List[str] = field(default_factory=list)
    total_training_time: float = 0.0
    system_info: Dict[str, Any] = field(default_factory=dict)
    mps_optimization_applied: bool = False
    checkpoint_resume_used: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'session_id': self.session_id,
            'start_time': self.start_time,
            'experts_to_train': self.experts_to_train,
            'completed_experts': self.completed_experts,
            'failed_experts': self.failed_experts,
            'total_training_time': self.total_training_time,
            'system_info': self.system_info,
            'mps_optimization_applied': self.mps_optimization_applied,
            'checkpoint_resume_used': self.checkpoint_resume_used
        }


@dataclass
class ExpertTrainingResult:
    """Result of training a single expert."""
    expert_name: str
    success: bool
    training_time: float
    checkpoint_created: Optional[str] = None
    final_metrics: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'expert_name': self.expert_name,
            'success': self.success,
            'training_time': self.training_time,
            'checkpoint_created': self.checkpoint_created,
            'final_metrics': self.final_metrics,
            'error_message': self.error_message
        }


class ChessGemmaTrainingOrchestrator:
    """
    Unified training orchestrator for the complete ChessGemma system.

    Manages the sequential training of all experts with:
    - Intelligent checkpoint management and resume
    - MPS memory optimization
    - Comprehensive progress tracking
    - Error recovery and reporting
    - Resource management
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the training orchestrator."""
        self.config_path = config_path or Path(__file__).parent / "configs" / "expert_training_config.json"
        self.project_root = project_root
        self.checkpoints_dir = self.project_root / "checkpoints"
        self.output_dir = self.project_root / "training_reports"

        # Initialize components
        self.checkpoint_manager = None
        self.mps_optimizer = None
        self.expert_trainer = None

        # Training session tracking
        self.current_session: Optional[TrainingSession] = None
        self.expert_results: List[ExpertTrainingResult] = []

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("🎭 ChessGemma Training Orchestrator initialized")
        logger.info(f"   Config: {self.config_path}")
        logger.info(f"   Checkpoints: {self.checkpoints_dir}")
        logger.info(f"   Reports: {self.output_dir}")

    def initialize_components(self) -> bool:
        """Initialize all available training components."""
        components_initialized = 0
        total_components = 3

        # Initialize checkpoint manager
        if CHECKPOINT_MANAGER_AVAILABLE:
            try:
                self.checkpoint_manager = CheckpointManager(self.checkpoints_dir)
                logger.info("✅ Checkpoint manager initialized")
                components_initialized += 1
            except Exception as e:
                logger.warning(f"⚠️  Checkpoint manager initialization failed: {e}")
                self.checkpoint_manager = None
        else:
            logger.warning("⚠️  Checkpoint manager not available")
            self.checkpoint_manager = None

        # Initialize MPS optimizer
        if MPS_OPTIMIZER_AVAILABLE:
            try:
                self.mps_optimizer = MPSMemoryOptimizer()
                logger.info("✅ MPS optimizer initialized")
                components_initialized += 1
            except Exception as e:
                logger.warning(f"⚠️  MPS optimizer initialization failed: {e}")
                self.mps_optimizer = None
        else:
            logger.warning("⚠️  MPS optimizer not available")
            self.mps_optimizer = None

        # Initialize expert trainer
        if EXPERT_TRAINER_AVAILABLE:
            try:
                self.expert_trainer = ChessExpertTrainer(str(self.config_path))
                logger.info("✅ Expert trainer initialized")
                components_initialized += 1
            except Exception as e:
                logger.warning(f"⚠️  Expert trainer initialization failed: {e}")
                self.expert_trainer = None
        else:
            logger.warning("⚠️  Expert trainer not available")
            self.expert_trainer = None

        if components_initialized == 0:
            logger.error("❌ No training components could be initialized")
            return False

        if components_initialized < total_components:
            logger.warning(f"⚠️  Only {components_initialized}/{total_components} components initialized")
            logger.warning("   Limited functionality available - some features may not work")

        return True

    @log_performance
    def train_all_experts(self, experts: List[str] = None, resume: bool = True,
                         validate: bool = True) -> Dict[str, Any]:
        """
        Train all experts in sequence.

        Args:
            experts: List of experts to train (default: ['uci', 'tutor', 'director'])
            resume: Whether to resume from checkpoints
            validate: Whether to validate after training

        Returns:
            Comprehensive training report
        """

        if experts is None:
            experts = ['uci', 'tutor', 'director']

        # Initialize training session
        session_id = f"chessgemma_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_session = TrainingSession(
            session_id=session_id,
            start_time=datetime.now().isoformat(),
            experts_to_train=experts,
            system_info=self._get_system_info(),
            mps_optimization_applied=bool(self.mps_optimizer),
            checkpoint_resume_used=resume
        )

        # Update system info with component availability
        self.current_session.system_info.update({
            'expert_trainer_available': EXPERT_TRAINER_AVAILABLE,
            'checkpoint_manager_available': CHECKPOINT_MANAGER_AVAILABLE,
            'mps_optimizer_available': MPS_OPTIMIZER_AVAILABLE
        })

        # Check if we have the required components
        if not self.expert_trainer:
            error_msg = "Expert trainer not available - cannot proceed with training"
            logger.error(f"❌ {error_msg}")
            return {
                'error': error_msg,
                'session_info': self.current_session.to_dict() if self.current_session else {},
                'training_summary': {'total_experts': 0, 'successful_experts': 0, 'failed_experts': 0}
            }

        logger.info("🚀 Starting ChessGemma unified training session")
        logger.info(f"   Session ID: {session_id}")
        logger.info(f"   Experts to train: {', '.join(experts)}")
        logger.info(f"   Resume enabled: {resume}")
        logger.info(f"   MPS optimization: {'enabled' if self.mps_optimizer else 'disabled'}")
        logger.info(f"   Checkpoint management: {'enabled' if self.checkpoint_manager else 'disabled'}")

        session_start_time = time.time()
        success_count = 0

        try:
            # Train each expert in sequence
            for i, expert_name in enumerate(experts, 1):
                logger.info(f"\n{'='*80}")
                logger.info(f"🎯 EXPERT {i}/{len(experts)}: {expert_name.upper()}")
                logger.info(f"{'='*80}")

                # Train individual expert
                result = self._train_single_expert(expert_name, resume)

                if result.success:
                    success_count += 1
                    self.current_session.completed_experts.append(expert_name)
                    logger.info(f"✅ {expert_name} training completed successfully")
                else:
                    self.current_session.failed_experts.append(expert_name)
                    logger.error(f"❌ {expert_name} training failed: {result.error_message}")

                self.expert_results.append(result)

                # Brief pause between experts for system cooling
                if i < len(experts):
                    logger.info("⏱️  Cooling period before next expert...")
                    time.sleep(30)  # 30 second pause

            # Update session timing
            self.current_session.total_training_time = time.time() - session_start_time

            # Run validation if requested
            if validate and success_count > 0:
                logger.info("\n🧪 Running post-training validation...")
                self._run_validation()

            # Generate final report
            report = self._generate_training_report()

            # Save session data
            self._save_session_data()

            # Final summary
            self._print_final_summary()

            return report

        except KeyboardInterrupt:
            logger.warning("🛑 Training interrupted by user")
            self._handle_interruption()
            return self._generate_training_report(interrupted=True)

        except Exception as e:
            logger.error(f"💥 Critical training error: {e}")
            self._handle_critical_error(e)
            return self._generate_training_report(error=str(e))

    def _train_single_expert(self, expert_name: str, resume: bool) -> ExpertTrainingResult:
        """Train a single expert with comprehensive error handling."""

        result = ExpertTrainingResult(expert_name=expert_name, success=False, training_time=0.0)

        try:
            start_time = time.time()

            # Check for existing checkpoints if resume is enabled
            if resume and self.checkpoint_manager:
                resume_data = self.checkpoint_manager.resume_from_checkpoint(expert_name)
                if resume_data:
                    checkpoint_dir, metadata, _ = resume_data
                    logger.info(f"📂 Resuming {expert_name} from step {metadata.global_step}")
                    result.checkpoint_created = str(checkpoint_dir)

            # Train the expert
            training_result = self.expert_trainer.train_expert(
                expert_name=expert_name,
                resume_from_checkpoint=resume
            )

            result.training_time = time.time() - start_time
            result.success = True
            result.final_metrics = training_result.__dict__ if hasattr(training_result, '__dict__') else {}

            # Get latest checkpoint info
            if self.checkpoint_manager:
                latest = self.checkpoint_manager.find_latest_checkpoint(expert_name)
                if latest:
                    checkpoint_dir, metadata = latest
                    result.checkpoint_created = str(checkpoint_dir)

            logger.info(f"✅ {expert_name} trained successfully in {result.training_time:.1f}s")

        except Exception as e:
            result.training_time = time.time() - start_time
            result.success = False
            result.error_message = str(e)
            logger.error(f"❌ {expert_name} training failed: {e}")

        return result

    def _run_validation(self) -> None:
        """Run validation on trained experts."""
        try:
            for expert_result in self.expert_results:
                if expert_result.success:
                    expert_name = expert_result.expert_name
                    logger.info(f"🧪 Validating {expert_name} expert...")

                    # Run basic validation
                    validation = self.expert_trainer.validate_expert_performance(expert_name, [])
                    logger.info(f"   {expert_name}: {validation.get('basic_validation', 'unknown')}")

        except Exception as e:
            logger.warning(f"Validation failed: {e}")

    def _generate_training_report(self, interrupted: bool = False, error: str = None) -> Dict[str, Any]:
        """Generate comprehensive training report."""

        end_time = datetime.now().isoformat()
        total_time = self.current_session.total_training_time if self.current_session else 0

        report = {
            'session_info': self.current_session.to_dict() if self.current_session else {},
            'training_summary': {
                'total_experts': len(self.current_session.experts_to_train) if self.current_session else 0,
                'successful_experts': len(self.current_session.completed_experts) if self.current_session else 0,
                'failed_experts': len(self.current_session.failed_experts) if self.current_session else 0,
                'total_training_time': total_time,
                'average_time_per_expert': total_time / max(len(self.expert_results), 1),
                'interrupted': interrupted,
                'error': error
            },
            'expert_results': [result.to_dict() for result in self.expert_results],
            'system_performance': self._get_system_performance_metrics(),
            'recommendations': self._generate_recommendations(),
            'generated_at': end_time
        }

        # Save detailed report
        if self.current_session:
            report_file = self.output_dir / f"training_report_{self.current_session.session_id}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"📊 Detailed training report saved: {report_file}")

        return report

    def _save_session_data(self) -> None:
        """Save session data for recovery."""
        if not self.current_session:
            return

        session_file = self.output_dir / f"session_{self.current_session.session_id}.json"
        with open(session_file, 'w') as f:
            json.dump(self.current_session.to_dict(), f, indent=2, default=str)

    def _print_final_summary(self) -> None:
        """Print final training summary."""
        if not self.current_session:
            return

        session = self.current_session
        total_time = session.total_training_time

        print("\n" + "="*80)
        print("🎭 CHESSGEMMA TRAINING SESSION COMPLETE")
        print("="*80)
        print(f"Session ID: {session.session_id}")
        print(f"Duration: {total_time:.1f} seconds ({total_time/3600:.1f} hours)")
        print()
        print("EXPERT TRAINING RESULTS:")
        print("-" * 40)

        for result in self.expert_results:
            status = "✅ SUCCESS" if result.success else "❌ FAILED"
            time_str = f"{result.training_time:.1f}s"
            print(f"  {result.expert_name.upper():8} | {status:10} | {time_str:8}")

        print()
        print("SUMMARY:")
        print(f"  Total Experts: {len(session.experts_to_train)}")
        print(f"  Successful: {len(session.completed_experts)}")
        print(f"  Failed: {len(session.failed_experts)}")
        print(f"  Success Rate: {len(session.completed_experts)/max(len(session.experts_to_train), 1)*100:.1f}%")

        if session.checkpoint_resume_used:
            print("  Checkpoint Resume: Enabled")
        if session.mps_optimization_applied:
            print("  MPS Optimization: Applied")

        print("="*80)

    def _get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        try:
            import psutil
            import platform

            return {
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'cpu_count': psutil.cpu_count(),
                'total_memory_gb': psutil.virtual_memory().total / (1024**3),
                'available_memory_gb': psutil.virtual_memory().available / (1024**3),
                'mps_available': hasattr(torch, 'backends') and torch.backends.mps.is_available(),
                'training_start_time': datetime.now().isoformat()
            }
        except:
            return {'error': 'Could not collect system info'}

    def _get_system_performance_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics."""
        try:
            import psutil

            return {
                'final_memory_usage': psutil.virtual_memory().percent,
                'cpu_usage': psutil.cpu_percent(interval=1),
                'disk_usage': psutil.disk_usage('/').percent
            }
        except:
            return {'error': 'Could not collect performance metrics'}

    def _generate_recommendations(self) -> List[str]:
        """Generate training recommendations based on results."""
        recommendations = []

        if not self.expert_results:
            return ["No training results available for analysis"]

        # Analyze success rates
        success_rate = sum(1 for r in self.expert_results if r.success) / len(self.expert_results)

        if success_rate < 1.0:
            recommendations.append("Consider investigating failures in failed experts")
            recommendations.append("Check system resources and memory availability")

        # Analyze training times
        avg_time = sum(r.training_time for r in self.expert_results) / len(self.expert_results)
        if avg_time > 3600:  # Over 1 hour per expert
            recommendations.append("Consider reducing training steps or batch sizes for faster iteration")
            recommendations.append("Enable gradient checkpointing for memory-efficient training")

        # MPS-specific recommendations
        if self.mps_optimizer and self.current_session and self.current_session.mps_optimization_applied:
            recommendations.append("MPS optimization is active - monitor memory usage during training")
            recommendations.append("Consider clearing MPS cache between expert trainings")

        # Checkpoint recommendations
        if self.checkpoint_manager and self.current_session and self.current_session.checkpoint_resume_used:
            recommendations.append("Checkpoint resume is working - training can be safely interrupted")
            recommendations.append("Consider regular checkpoint validation for data integrity")

        if not recommendations:
            recommendations.append("Training completed successfully - all systems operating optimally")
            recommendations.append("Consider increasing training steps for better model performance")

        return recommendations

    def _handle_interruption(self) -> None:
        """Handle training interruption gracefully."""
        logger.warning("Training session interrupted - saving progress...")

        if self.current_session:
            self.current_session.total_training_time = time.time() - time.mktime(
                datetime.fromisoformat(self.current_session.start_time).timetuple()
            )

        # Save current state
        self._save_session_data()
        self._generate_training_report(interrupted=True)

        logger.info("Progress saved - you can resume training later with --resume")

    def _handle_critical_error(self, error: Exception) -> None:
        """Handle critical training errors."""
        logger.critical(f"Critical training error: {error}")

        if self.current_session:
            self.current_session.failed_experts.extend(
                [e for e in self.current_session.experts_to_train
                 if e not in self.current_session.completed_experts + self.current_session.failed_experts]
            )

        # Save error state
        self._save_session_data()
        self._generate_training_report(error=str(error))


def main():
    """Main entry point for ChessGemma unified training."""

    parser = argparse.ArgumentParser(
        description="ChessGemma Unified Training Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all experts with automatic resume
  python -m src.training.train_chessgemmma

  # Train specific experts only
  python -m src.training.train_chessgemmma --experts uci tutor

  # Fresh training (no resume)
  python -m src.training.train_chessgemmma --no-resume

  # Train with validation
  python -m src.training.train_chessgemmma --validate
        """
    )

    parser.add_argument(
        '--experts',
        nargs='+',
        choices=['uci', 'tutor', 'director'],
        default=['uci', 'tutor', 'director'],
        help='Experts to train (default: all)'
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        default=True,
        help='Resume from existing checkpoints (default: enabled)'
    )

    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Start fresh training (disable resume)'
    )

    parser.add_argument(
        '--validate',
        action='store_true',
        help='Run validation after training each expert'
    )

    parser.add_argument(
        '--config',
        type=str,
        help='Path to training configuration file'
    )

    parser.add_argument(
        '--report-dir',
        type=str,
        help='Directory to save training reports'
    )

    args = parser.parse_args()

    # Handle resume flag conflict
    if args.no_resume:
        args.resume = False

    print("🎭 ChessGemma Unified Training Orchestrator")
    print("=" * 60)

    # Initialize orchestrator
    orchestrator = ChessGemmaTrainingOrchestrator(args.config)

    if args.report_dir:
        orchestrator.output_dir = Path(args.report_dir)

    # Initialize components
    if not orchestrator.initialize_components():
        print("❌ Failed to initialize training components")
        sys.exit(1)

    # Run training
    try:
        report = orchestrator.train_all_experts(
            experts=args.experts,
            resume=args.resume,
            validate=args.validate
        )

        # Exit with success/failure code
        success_count = report['training_summary']['successful_experts']
        total_count = report['training_summary']['total_experts']

        if success_count == total_count:
            print("\n🎉 All experts trained successfully!")
            sys.exit(0)
        elif success_count > 0:
            print(f"\n⚠️  Partial success: {success_count}/{total_count} experts trained")
            sys.exit(1)
        else:
            print("\n💥 Training failed completely")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
        sys.exit(130)  # Standard interrupt exit code
    except Exception as e:
        print(f"\n💥 Critical error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
