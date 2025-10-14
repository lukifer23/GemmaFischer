#!/usr/bin/env python3
"""Hybrid engine orchestration for LC0 + LLM explanations."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
import threading
from typing import Dict, List, Optional, Tuple

import chess

from .chess_engine import (
    ChessEngineManager,
    create_lc0_manager,
    create_stockfish_manager,
    PositionAnalysis,
    lc0_pool,
)
from ..config.config_manager import ChessEngineConfig
from ..utils.common import get_config_manager


@dataclass
class EngineSelection:
    name: str
    manager: ChessEngineManager
    time_limit: float
    depth: Optional[int]


@dataclass
class HybridEngineResult:
    fen: str
    engine_name: str
    best_move: Optional[str]
    principal_variation: List[str]
    evaluation_cp: Optional[int]
    mate_in: Optional[int]
    depth: int
    nodes: int
    engine_time: float
    fallback_used: bool = False
    raw_analysis: Dict[str, any] = field(default_factory=dict)


class HybridEngine:
    """Coordinate LC0 primary engine with optional Stockfish fallback."""

    def __init__(self, config: Optional[ChessEngineConfig] = None):
        config_manager = get_config_manager()
        if config is None and config_manager:
            try:
                config = config_manager().chess_engine
            except Exception:
                config = ChessEngineConfig()
        elif config is None:
            config = ChessEngineConfig()

        self.settings = config
        self.primary: Optional[EngineSelection] = None
        self.fallback: Optional[EngineSelection] = None
        self._engines_initialized = False
        self._init_lock = threading.RLock()

    def _initialize_engines(self) -> None:
        # Guard against concurrent initialization under parallel requests
        with self._init_lock:
            if self._engines_initialized and self.primary is not None:
                return

            cfg = self.settings

        # Primary engine
        primary_key = cfg.primary.lower()
        if primary_key == "lc0" and cfg.lc0.enabled:
            try:
                # Use the LC0 pool instead of creating new instances
                lc0_config = asdict(cfg.lc0)
                manager = lc0_pool.get_engine("primary", lc0_config)
                self.primary = EngineSelection(
                    name="LC0",
                    manager=manager,
                    time_limit=cfg.lc0.time_limit,
                    depth=cfg.lc0.depth,
                )
                logger.info("✅ LC0 primary engine initialized from pool")
            except Exception as exc:
                logger.warning("Failed to initialize LC0 from pool: %s", exc)
                self.primary = None
        elif primary_key == "stockfish":
            try:
                manager = create_stockfish_manager(asdict(cfg.fallback))
                self.primary = EngineSelection(
                    name="Stockfish",
                    manager=manager,
                    time_limit=cfg.fallback.time_limit,
                    depth=cfg.fallback.depth,
                )
            except Exception as exc:
                from ..utils.common import get_logger
                get_logger(__name__).warning("Failed to initialize Stockfish primary: %s", exc)
                self.primary = None

        # Fallback engine (Stockfish)
        if cfg.fallback.enabled:
            try:
                manager = create_stockfish_manager(asdict(cfg.fallback))
                self.fallback = EngineSelection(
                    name="Stockfish",
                    manager=manager,
                    time_limit=cfg.fallback.time_limit,
                    depth=cfg.fallback.depth,
                )
            except Exception as exc:
                from ..utils.common import get_logger
                get_logger(__name__).warning("Failed to initialize fallback engine: %s", exc)
                self.fallback = None

        # Ensure we have at least one engine
        if self.primary is None and self.fallback is not None:
            self.primary = self.fallback
            self.fallback = None

        self._engines_initialized = True

    def analyze(self, fen: str) -> HybridEngineResult:
        # Lazy initialization of engines
        if not self._engines_initialized:
            self._initialize_engines()

        if not self.primary:
            raise RuntimeError("No chess engine available for analysis.")

        analysis, elapsed, error = self._run_engine(self.primary, fen)
        used_selection = self.primary

        if (analysis.best_move is None or error is not None) and self.fallback and self.fallback is not self.primary:
            analysis, elapsed, error = self._run_engine(self.fallback, fen)
            used_selection = self.fallback

        pv = [mv.move for mv in analysis.top_moves] if analysis.top_moves else []
        nodes = analysis.evaluation.get('nodes', 0)
        depth = analysis.evaluation.get('depth', 0)

        return HybridEngineResult(
            fen=fen,
            engine_name=used_selection.name,
            best_move=analysis.best_move,
            principal_variation=pv,
            evaluation_cp=analysis.best_score,
            mate_in=analysis.mate_in,
            depth=depth,
            nodes=nodes,
            engine_time=elapsed,
            fallback_used=(used_selection is self.fallback and self.fallback is not None),
            raw_analysis={
                'evaluation': analysis.evaluation,
                'threats': analysis.threats,
                'opportunities': analysis.opportunities,
                'position_type': analysis.position_type,
                'error': str(error) if error else None,
            },
        )

    def health(self) -> Dict[str, Any]:
        """Return health information for primary and fallback engines."""
        # Lazy initialization of engines
        if not self._engines_initialized:
            self._initialize_engines()
            self._engines_initialized = True

        def _summary(selection: Optional[EngineSelection]) -> Optional[Dict[str, Any]]:
            if not selection:
                return None
            manager = selection.manager
            return {
                'name': selection.name,
                'engine_path': getattr(manager, 'engine_path', None),
                # Expose configured options when available for UI debugging
                'options': list(getattr(manager, 'engine', None).options.keys()) if getattr(manager, 'engine', None) is not None else [],
                'configured_threads': getattr(manager, 'engine_options', {}).get('Threads') if hasattr(manager, 'engine_options') else None,
                'configured_backend': getattr(manager, 'engine_options', {}).get('Backend') if hasattr(manager, 'engine_options') else None,
                'configured_weights': getattr(manager, 'engine_options', {}).get('WeightsFile') if hasattr(manager, 'engine_options') else None,
                'search_depth': selection.depth,
                'time_limit': selection.time_limit,
                'active': getattr(manager, 'engine', None) is not None,
            }

        return {
            'primary': _summary(self.primary),
            'fallback': _summary(self.fallback),
        }

    def _run_engine(
        self,
        selection: EngineSelection,
        fen: str,
    ) -> Tuple[PositionAnalysis, float, Optional[Exception]]:
        start = time.time()
        try:
            analysis = selection.manager.analyze_position(
                fen,
                depth=selection.depth if selection.depth is not None else 15,
                time_limit=selection.time_limit,
            )
            elapsed = time.time() - start
            return analysis, elapsed, None
        except Exception as exc:
            elapsed = time.time() - start
            return PositionAnalysis(fen=fen), elapsed, exc
