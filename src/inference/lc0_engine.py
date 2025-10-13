#!/usr/bin/env python3
"""
LC0 Engine Manager for GemmaFischer MoE System

Provides optimized LC0 (LeelaChess Zero) integration with Metal backend
for Apple Silicon, enabling neural network-based chess analysis.
"""

import os
import sys
import time
import logging
import threading
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

import chess
import chess.engine

# Configure logging
logger = logging.getLogger(__name__)

class LC0Backend(Enum):
    """LC0 backend options for different hardware."""
    METAL = "metal"      # Apple Silicon GPU (recommended)
    BLAS = "blas"        # CPU BLAS acceleration  
    EIGEN = "eigen"      # CPU Eigen library
    TRIVIAL = "trivial"  # Basic CPU (fallback)
    RANDOM = "random"    # Random moves (testing only)

@dataclass
class LC0Config:
    """Configuration for LC0 engine."""
    weights_file: Optional[str] = None
    backend: LC0Backend = LC0Backend.METAL
    threads: int = 2  # M3 Pro optimization
    nn_cache_size: int = 200000  # GPU memory optimization
    max_batch_size: int = 256
    temperature: float = 1.0  # Move selection temperature
    policy_softmax_temp: float = 1.36  # Policy softmax temperature
    
    # Search parameters
    search_time: float = 2.0  # seconds per move
    nodes_limit: Optional[int] = None
    depth_limit: Optional[int] = None
    
    # Advanced options
    cpuct: float = 3.4  # Exploration constant
    cpuct_base: float = 19652.0
    cpuct_factor: float = 1.0
    
    # Memory management
    ram_limit_mb: int = 1024  # RAM limit for search
    
@dataclass
class LC0Analysis:
    """LC0 position analysis result."""
    fen: str
    best_move: Optional[chess.Move] = None
    best_score: Optional[float] = None  # Centipawns
    principal_variation: List[chess.Move] = field(default_factory=list)
    search_time: float = 0.0
    nodes_searched: int = 0
    depth_reached: int = 0
    
    # Win/draw/loss probabilities (if available)
    wdl: Optional[Tuple[float, float, float]] = None
    
    # Additional info
    is_mate: bool = False
    mate_in: Optional[int] = None
    confidence: float = 0.0  # 0.0 to 1.0
    
    # Raw engine info
    raw_info: Dict[str, Any] = field(default_factory=dict)

class LC0EngineManager:
    """
    Optimized LC0 engine manager for Apple Silicon.
    
    Provides high-level interface for LC0 chess analysis with:
    - Metal GPU acceleration on Apple Silicon
    - Intelligent caching and memory management
    - Robust error handling and fallbacks
    - Performance monitoring and optimization
    """
    
    def __init__(self, config: Optional[LC0Config] = None):
        """
        Initialize LC0 engine manager.
        
        Args:
            config: LC0 configuration. If None, uses defaults optimized for M3 Pro.
        """
        self.config = config or self._create_default_config()
        self.engine_path = "/opt/homebrew/bin/lc0"
        self.engine: Optional[chess.engine.SimpleEngine] = None
        self._engine_lock = threading.RLock()
        
        # Performance monitoring
        self._analysis_count = 0
        self._total_analysis_time = 0.0
        self._cache_hits = 0
        
        # Initialize engine
        self._initialize_engine()
        
    def _create_default_config(self) -> LC0Config:
        """Create default configuration optimized for M3 Pro."""
        config = LC0Config()
        
        # Check for available weights
        weights_dir = Path("models/lc0_weights")
        if weights_dir.exists():
            # Look for any .pb.gz files
            weight_files = list(weights_dir.glob("*.pb.gz"))
            if weight_files:
                # Use the largest/most recent file
                best_weights = max(weight_files, key=lambda x: (x.stat().st_size, x.stat().st_mtime))
                config.weights_file = str(best_weights)
                logger.info(f"Found LC0 weights: {best_weights.name}")
            else:
                logger.warning("No LC0 weights found, using random weights (slower)")
                config.backend = LC0Backend.RANDOM
        else:
            logger.warning("LC0 weights directory not found, using random weights")
            config.backend = LC0Backend.RANDOM
            
        return config
    
    def _initialize_engine(self) -> None:
        """Initialize LC0 engine with optimized settings."""
        try:
            # Build engine options
            options = {
                'Threads': self.config.threads,
                'NNCacheSize': self.config.nn_cache_size,
                'Backend': self.config.backend.value,
                'PolicyTemperature': self.config.policy_softmax_temp,
            }
            
            # Add weights if available
            if self.config.weights_file and Path(self.config.weights_file).exists():
                options['WeightsFile'] = self.config.weights_file
                logger.info(f"Using LC0 weights: {self.config.weights_file}")
            else:
                logger.info("No weights file specified, LC0 will use built-in or random weights")
            
            # Add Metal-specific optimizations for Apple Silicon
            if self.config.backend == LC0Backend.METAL:
                # Metal backend specific options
                options.update({
                    'BLASCoreRatio': 1.0,  # Balance CPU/GPU usage
                })
            
            # Initialize engine
            with self._engine_lock:
                self.engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
                
                # Configure engine options
                valid_options = {}
                engine_options = self.engine.options
                
                for key, value in options.items():
                    if key in engine_options:
                        valid_options[key] = value
                        logger.debug(f"Setting LC0 option: {key} = {value}")
                    else:
                        logger.warning(f"LC0 option not supported: {key}")
                
                if valid_options:
                    self.engine.configure(valid_options)
                    logger.info(f"Configured LC0 with {len(valid_options)} options")
                
                # Test engine
                try:
                    board = chess.Board()
                    info = self.engine.analyse(board, chess.engine.Limit(depth=5))
                    logger.info("LC0 engine initialized successfully")
                    logger.debug(f"Test analysis score: {info.get('score', 'N/A')}")
                except Exception as e:
                    logger.warning(f"LC0 test analysis failed: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to initialize LC0 engine: {e}")
            self.engine = None
            raise
    
    def analyze_position(self, 
                        board: chess.Board, 
                        limit: chess.engine.Limit,
                        multipv: int = 1) -> LC0Analysis:
        """
        Analyze a chess position using LC0.
        
        Args:
            board: Chess board position
            limit: Search time/depth limit
            multipv: Number of principal variations to return
            
        Returns:
            LC0Analysis object with detailed results
        """
        if not self.engine:
            raise RuntimeError("LC0 engine not initialized")
            
        start_time = time.time()
        
        try:
            with self._engine_lock:
                # Configure multipv if supported
                if 'MultiPV' in self.engine.options and multipv > 1:
                    self.engine.configure({'MultiPV': multipv})
                
                # Analyze position
                info = self.engine.analyse(board, limit)
                
                # Extract results
                analysis = self._parse_analysis_result(board.fen(), info, time.time() - start_time)
                
                # Update performance stats
                self._analysis_count += 1
                self._total_analysis_time += analysis.search_time
                
                return analysis
                
        except Exception as e:
            logger.error(f"LC0 analysis failed: {e}")
            # Return basic analysis on failure
            return LC0Analysis(
                fen=board.fen(),
                search_time=time.time() - start_time,
                confidence=0.0
            )
    
    def find_best_move(self, 
                      fen: str, 
                      time_limit: float = 2.0,
                      depth_limit: Optional[int] = None) -> Optional[chess.Move]:
        """
        Find the best move for a position using LC0.
        
        Args:
            fen: FEN string of position
            time_limit: Time limit in seconds
            depth_limit: Optional depth limit
            
        Returns:
            Best move found, or None if analysis failed
        """
        try:
            board = chess.Board(fen)
            
            # Create search limit
            limit_kwargs = {'time': time_limit}
            if depth_limit:
                limit_kwargs['depth'] = depth_limit
            limit = chess.engine.Limit(**limit_kwargs)
            
            # Analyze position
            analysis = self.analyze_position(board, limit)
            
            return analysis.best_move
            
        except Exception as e:
            logger.error(f"Failed to find best move for {fen}: {e}")
            return None
    
    def _parse_analysis_result(self, fen: str, info: Dict[str, Any], search_time: float) -> LC0Analysis:
        """Parse LC0 analysis result into LC0Analysis object."""
        
        analysis = LC0Analysis(fen=fen, search_time=search_time)
        
        # Extract basic information
        analysis.raw_info = dict(info)  # Store raw info
        
        # Best move
        if 'pv' in info and info['pv']:
            analysis.best_move = info['pv'][0]
            analysis.principal_variation = list(info['pv'])
        
        # Score
        if 'score' in info:
            score = info['score']
            # Handle PovScore object correctly
            try:
                if hasattr(score, 'is_mate') and score.is_mate():
                    analysis.is_mate = True
                    analysis.mate_in = score.mate()
                    analysis.best_score = 10000 if score.mate() > 0 else -10000
                elif hasattr(score, 'score'):
                    # Get score from white's perspective
                    analysis.best_score = score.score()
                else:
                    # Fallback: try to get the score value directly
                    analysis.best_score = int(score)
            except Exception as e:
                logger.debug(f"Could not parse score {score}: {e}")
                analysis.best_score = 0
        
        # Search statistics
        analysis.nodes_searched = info.get('nodes', 0)
        analysis.depth_reached = info.get('depth', 0)
        
        # Win/draw/loss probabilities
        if 'wdl' in info:
            wdl = info['wdl']
            analysis.wdl = (wdl.white(), wdl.draws(), wdl.black())
        
        # Calculate confidence based on search depth and time
        confidence_factors = []
        
        # Depth confidence (higher depth = higher confidence)
        if analysis.depth_reached >= 10:
            confidence_factors.append(0.9)
        elif analysis.depth_reached >= 5:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.5)
            
        # Time confidence (more time = higher confidence)
        if search_time >= 1.0:
            confidence_factors.append(0.9)
        elif search_time >= 0.5:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.5)
            
        # Score confidence (mate scores are very confident)
        if analysis.is_mate:
            confidence_factors.append(0.95)
        elif abs(analysis.best_score or 0) > 300:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.6)
            
        analysis.confidence = sum(confidence_factors) / len(confidence_factors)
        
        return analysis
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get information about the LC0 engine."""
        info = {
            'engine_path': self.engine_path,
            'initialized': self.engine is not None,
            'config': {
                'backend': self.config.backend.value,
                'threads': self.config.threads,
                'weights_file': self.config.weights_file,
                'nn_cache_size': self.config.nn_cache_size,
            },
            'performance': {
                'total_analyses': self._analysis_count,
                'avg_analysis_time': self._total_analysis_time / max(self._analysis_count, 1),
                'cache_hit_rate': 0.0,  # TODO: implement caching
            }
        }
        
        # Add engine options if available
        if self.engine:
            try:
                info['engine_options'] = list(self.engine.options.keys())
            except:
                pass
                
        return info
    
    def is_healthy(self) -> bool:
        """Check if the LC0 engine is healthy and responsive."""
        if not self.engine:
            return False
            
        try:
            # Quick health check
            board = chess.Board()
            info = self.engine.analyse(board, chess.engine.Limit(depth=3))
            return 'score' in info
        except Exception:
            return False
    
    def cleanup(self) -> None:
        """Clean up engine resources."""
        with self._engine_lock:
            if self.engine:
                try:
                    self.engine.quit()
                except:
                    pass
                self.engine = None
    
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

# Convenience functions for easy integration
def create_lc0_engine(weights_file: Optional[str] = None, 
                     backend: str = "metal") -> LC0EngineManager:
    """
    Create an LC0 engine with sensible defaults.
    
    Args:
        weights_file: Path to network weights file (auto-discovered if None)
        backend: Backend to use ('metal', 'blas', 'eigen', 'random')
        
    Returns:
        Configured LC0EngineManager instance
    """
    config = LC0Config()
    
    # Set backend
    try:
        config.backend = LC0Backend(backend)
    except ValueError:
        logger.warning(f"Unknown backend '{backend}', using metal")
        config.backend = LC0Backend.METAL
    
    # Set weights if provided
    if weights_file:
        config.weights_file = weights_file
    
    return LC0EngineManager(config)

def test_lc0_basic() -> bool:
    """Test basic LC0 functionality."""
    try:
        with create_lc0_engine() as engine:
            # Test with starting position
            board = chess.Board()
            analysis = engine.analyze_position(board, chess.engine.Limit(depth=5))
            
            if analysis.best_move:
                print(f"✅ LC0 basic test passed. Best move: {analysis.best_move}")
                print(f"   Score: {analysis.best_score}, Depth: {analysis.depth_reached}")
                return True
            else:
                print("❌ LC0 analysis returned no move")
                return False
                
    except Exception as e:
        print(f"❌ LC0 basic test failed: {e}")
        return False

if __name__ == "__main__":
    # Quick test when run directly
    print("🧠 Testing LC0 Engine Manager...")
    success = test_lc0_basic()
    if success:
        print("🎉 LC0 integration ready!")
    else:
        print("❌ LC0 integration needs debugging")
