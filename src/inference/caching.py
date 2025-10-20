#!/usr/bin/env python3
"""
Caching System for ChessGemma

Provides efficient response caching, position caching, and performance optimization
for chess inference operations.
"""

from __future__ import annotations

import hashlib
import threading
import json
import os
from pathlib import Path
from collections import OrderedDict
from typing import Dict, Any, Optional, Tuple

# Import common utilities
from src.utils.common import get_logger, get_config_manager

# Get utility functions
logger = get_logger(__name__)
config_manager = get_config_manager()


class ChessInferenceCache:
    """High-performance caching system for chess inference operations."""

    def __init__(self, max_cache_size: Optional[int] = None):
        # Use configuration for cache sizes if available
        if config_manager and max_cache_size is None:
            try:
                config = config_manager()
                self.max_cache_size = config.cache.max_cache_size
                self._engine_cache_max = config.cache.engine_cache_max
            except:
                # Fall back to defaults
                self.max_cache_size = 512
                self._engine_cache_max = 1024
        else:
            # Use provided values or defaults
            self.max_cache_size = max_cache_size or 512
            self._engine_cache_max = 1024

        self._response_cache = OrderedDict()  # L1 cache for responses
        self._engine_cache = OrderedDict()    # L1 cache for engine moves
        self._kv_cache = {}                    # L1 cache for KV states

        # L2 (persistent) caches
        self._l2_cache_enabled = os.environ.get('CHESSGEMMA_L2_CACHE_ENABLED', '1') not in ('0', 'false', 'False')
        self._l2_cache_dir = Path(os.environ.get('CHESSGEMMA_L2_CACHE_DIR', 'cache/l2'))
        self._l2_response_cache: Optional[Dict[str, Any]] = None
        self._l2_engine_cache: Optional[Dict[str, Any]] = None

        # L2 cache size limits (smaller than L1 since disk I/O is slower)
        self._l2_max_response_cache = min(self.max_cache_size // 4, 256)
        self._l2_max_engine_cache = min(self._engine_cache_max // 4, 512)

        # Performance tracking with reduced overhead
        self._cache_hits = 0
        self._l2_cache_hits = 0
        self._total_requests = 0

        # Initialize L2 caches if enabled
        if self._l2_cache_enabled:
            self._initialize_l2_caches()

        # Thread safety - use faster Lock instead of RLock for better performance
        self._cache_lock = threading.Lock()

        logger.info(f"Chess Inference Cache initialized (max_size: {self.max_cache_size}, L2: {self._l2_cache_enabled})")

    def _initialize_l2_caches(self):
        """Initialize L2 (persistent) caches."""
        try:
            self._l2_cache_dir.mkdir(parents=True, exist_ok=True)

            # Initialize response cache
            response_cache_file = self._l2_cache_dir / "response_cache.json"
            if response_cache_file.exists():
                try:
                    with open(response_cache_file, 'r') as f:
                        self._l2_response_cache = json.load(f)
                    logger.debug(f"Loaded L2 response cache with {len(self._l2_response_cache)} entries")
                except Exception as e:
                    logger.warning(f"Failed to load L2 response cache: {e}")
                    self._l2_response_cache = {}
            else:
                self._l2_response_cache = {}

            # Initialize engine cache
            engine_cache_file = self._l2_cache_dir / "engine_cache.json"
            if engine_cache_file.exists():
                try:
                    with open(engine_cache_file, 'r') as f:
                        self._l2_engine_cache = json.load(f)
                    logger.debug(f"Loaded L2 engine cache with {len(self._l2_engine_cache)} entries")
                except Exception as e:
                    logger.warning(f"Failed to load L2 engine cache: {e}")
                    self._l2_engine_cache = {}
            else:
                self._l2_engine_cache = {}

        except Exception as e:
            logger.error(f"Failed to initialize L2 caches: {e}")
            self._l2_cache_enabled = False

    def _save_l2_cache(self, cache_type: str):
        """Save L2 cache to disk."""
        if not self._l2_cache_enabled:
            return

        try:
            if cache_type == "response" and self._l2_response_cache:
                cache_file = self._l2_cache_dir / "response_cache.json"
                with open(cache_file, 'w') as f:
                    json.dump(self._l2_response_cache, f, indent=None)  # Compact JSON for speed
            elif cache_type == "engine" and self._l2_engine_cache:
                cache_file = self._l2_cache_dir / "engine_cache.json"
                with open(cache_file, 'w') as f:
                    json.dump(self._l2_engine_cache, f, indent=None)
        except Exception as e:
            logger.debug(f"Failed to save L2 {cache_type} cache: {e}")

    def _get_l2_response(self, cache_key: str) -> Optional[Any]:
        """Get response from L2 cache."""
        if not self._l2_cache_enabled or not self._l2_response_cache:
            return None

        entry = self._l2_response_cache.get(cache_key)
        if entry:
            # Check TTL if present
            if 'expires' in entry:
                import time
                if time.time() > entry['expires']:
                    # Entry expired, remove it
                    del self._l2_response_cache[cache_key]
                    return None
            self._l2_cache_hits += 1
            return entry['data']
        return None

    def _put_l2_response(self, cache_key: str, data: Any, ttl_seconds: Optional[int] = None):
        """Store response in L2 cache."""
        if not self._l2_cache_enabled or not self._l2_response_cache:
            return

        entry = {'data': data}
        if ttl_seconds:
            import time
            entry['expires'] = time.time() + ttl_seconds

        self._l2_response_cache[cache_key] = entry

        # Maintain cache size limit
        if len(self._l2_response_cache) > self._l2_max_response_cache:
            # Remove oldest entries (simple FIFO eviction)
            excess = len(self._l2_response_cache) - self._l2_max_response_cache
            keys_to_remove = list(self._l2_response_cache.keys())[:excess]
            for key in keys_to_remove:
                del self._l2_response_cache[key]

    def _get_l2_engine_move(self, fen: str) -> Optional[str]:
        """Get engine move from L2 cache."""
        if not self._l2_cache_enabled or not self._l2_engine_cache:
            return None

        entry = self._l2_engine_cache.get(fen)
        if entry:
            if 'expires' in entry:
                import time
                if time.time() > entry['expires']:
                    del self._l2_engine_cache[fen]
                    return None
            self._l2_cache_hits += 1
            return entry['move']
        return None

    def _put_l2_engine_move(self, fen: str, move: str, ttl_seconds: Optional[int] = None):
        """Store engine move in L2 cache."""
        if not self._l2_cache_enabled or not self._l2_engine_cache:
            return

        entry = {'move': move}
        if ttl_seconds:
            import time
            entry['expires'] = time.time() + ttl_seconds

        self._l2_engine_cache[fen] = entry

        # Maintain cache size limit
        if len(self._l2_engine_cache) > self._l2_max_engine_cache:
            excess = len(self._l2_engine_cache) - self._l2_max_engine_cache
            keys_to_remove = list(self._l2_engine_cache.keys())[:excess]
            for key in keys_to_remove:
                del self._l2_engine_cache[key]

    def _create_cache_key(self, question: str, context: Optional[str], mode: str,
                         max_new_tokens: int, temperature: float, top_p: float) -> str:
        """Create a unique cache key for the request - optimized for speed."""
        # Use simpler key generation for better performance
        # Round floats to reduce cache fragmentation
        temp_rounded = round(temperature, 2)
        top_p_rounded = round(top_p, 2)

        # Create key more efficiently - avoid string concatenation in loop
        key_parts = [question, context or "", mode, str(max_new_tokens), str(temp_rounded), str(top_p_rounded)]
        key_string = "|".join(key_parts)

        # Use faster hash for better performance
        return hash(key_string) % (10**8)  # Simple hash for speed

    def _check_response_cache(self, cache_key: str) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
        """Check if response is cached - hierarchical caching with L1/L2 fallback."""
        # First check L1 cache
        cached = self._response_cache.get(cache_key)
        if cached is not None:
            # Update hit statistics
            self._cache_hits += 1
            self._response_cache.move_to_end(cache_key)

            # Calculate hit rate
            hit_rate = self._cache_hits / max(self._total_requests, 1)
            return cached.copy(), hit_rate

        # L1 miss - check L2 cache
        l2_cached = self._get_l2_response(cache_key)
        if l2_cached is not None:
            # Promote to L1 cache for faster future access
            self._cache_response(cache_key, l2_cached)

            # Calculate hit rate (including L2 hits)
            total_hits = self._cache_hits + self._l2_cache_hits
            hit_rate = total_hits / max(self._total_requests, 1)
            return l2_cached.copy(), hit_rate

        return None, None

    def _cache_response(self, cache_key: str, response: Dict[str, Any]):
        """Cache response for future use - hierarchical caching."""
        # Store in L1 cache
        self._response_cache[cache_key] = response.copy()

        # Maintain L1 cache size
        if len(self._response_cache) > self.max_cache_size:
            self._response_cache.popitem(last=False)

        # Also store in L2 cache for persistence (with TTL)
        if self._l2_cache_enabled:
            # Use default TTL from cache config or 1 hour
            ttl = getattr(self, '_cache_ttl_seconds', 3600)
            self._put_l2_response(cache_key, response, ttl)

    def _engine_cache_store(self, fen: Optional[str], move: str) -> None:
        """Store engine move in hierarchical cache."""
        if not fen:
            return

        # Store in L1 cache
        if fen in self._engine_cache:
            self._engine_cache.pop(fen, None)
        self._engine_cache[fen] = move

        # Maintain L1 cache size
        if len(self._engine_cache) > self._engine_cache_max:
            self._engine_cache.popitem(last=False)

        # Store in L2 cache for persistence
        if self._l2_cache_enabled:
            ttl = getattr(self, '_cache_ttl_seconds', 3600)
            self._put_l2_engine_move(fen, move, ttl)

    def _engine_cache_lookup(self, fen: Optional[str]) -> Optional[str]:
        """Lookup engine move from hierarchical cache."""
        if not fen:
            return None

        # Check L1 cache first
        mv = self._engine_cache.get(fen)
        if mv is not None:
            # Refresh LRU order
            self._engine_cache.pop(fen, None)
            self._engine_cache[fen] = mv
            return mv

        # L1 miss - check L2 cache
        l2_move = self._get_l2_engine_move(fen)
        if l2_move is not None:
            # Promote to L1 cache for faster future access
            self._engine_cache_store(fen, l2_move)
            return l2_move

        return None

    def save_l2_caches(self):
        """Save L2 caches to disk."""
        if not self._l2_cache_enabled:
            return

        try:
            self._save_l2_cache("response")
            self._save_l2_cache("engine")
            logger.debug("L2 caches saved to disk")
        except Exception as e:
            logger.debug(f"Failed to save L2 caches: {e}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics including hierarchical caching."""
        with self._cache_lock:
            total_requests = self._total_requests
            l1_hits = self._cache_hits
            l2_hits = self._l2_cache_hits
            total_hits = l1_hits + l2_hits

            l1_cache_size = len(self._response_cache)
            l1_engine_cache_size = len(self._engine_cache)

            l2_cache_size = len(self._l2_response_cache) if self._l2_response_cache else 0
            l2_engine_cache_size = len(self._l2_engine_cache) if self._l2_engine_cache else 0

            return {
                'total_requests': total_requests,
                'l1_cache_hits': l1_hits,
                'l2_cache_hits': l2_hits,
                'total_cache_hits': total_hits,
                'l1_hit_rate': l1_hits / max(total_requests, 1),
                'l2_hit_rate': l2_hits / max(total_requests, 1),
                'total_hit_rate': total_hits / max(total_requests, 1),
                'l1_response_cache_size': l1_cache_size,
                'l1_engine_cache_size': l1_engine_cache_size,
                'l2_response_cache_size': l2_cache_size,
                'l2_engine_cache_size': l2_engine_cache_size,
                'l2_cache_enabled': self._l2_cache_enabled,
                'max_l1_cache_size': self.max_cache_size,
                'max_l2_cache_size': self._l2_max_response_cache
            }

    def generate_response(
        self,
        question: str,
        context: Optional[str] = None,
        mode: str = "tutor",
        max_new_tokens: int = 200,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        do_sample: Optional[bool] = None,
        generate_func: callable = None
    ) -> Dict[str, Any]:
        """Generate response with caching support."""
        import time

        with self._cache_lock:
            self._total_requests += 1

        # Apply expert-specific decoding parameters
        temperature, top_p, do_sample = self._get_expert_decoding_params(mode, temperature, top_p, do_sample)

        # Check response cache for identical requests
        cache_key = self._create_cache_key(question, context, mode, max_new_tokens, temperature, top_p)
        cached_response, cache_hit_rate = self._check_response_cache(cache_key)
        if cached_response:
            cached_response["cached"] = True
            if cache_hit_rate is None:
                with self._cache_lock:
                    cache_hit_rate = self._cache_hits / max(self._total_requests, 1)
            cached_response["cache_hit_rate"] = cache_hit_rate
            return cached_response

        # Generate response if not cached
        start_time = time.time()
        try:
            if generate_func:
                response = generate_func(
                    question=question,
                    context=context,
                    mode=mode,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample
                )
            else:
                response = {
                    "response": "",
                    "confidence": 0.0,
                    "error": "No generation function provided"
                }

            # Add metadata
            response.update({
                "model_loaded": True,
                "mode": mode,
                "generation_time": time.time() - start_time,
                "cached": False,
                "cache_hit_rate": cache_hit_rate or 0.0,
            })

            # Cache the response
            self._cache_response(cache_key, response)

            return response

        except Exception as e:
            generation_time = time.time() - start_time
            return {
                "error": str(e),
                "response": "",
                "confidence": 0.0,
                "model_loaded": True,
                "mode": mode,
                "generation_time": generation_time,
                "cached": False,
                "cache_hit_rate": self._cache_hits / max(self._total_requests, 1)
            }

    def _get_expert_decoding_params(self, mode: str, temperature: Optional[float],
                                  top_p: Optional[float], do_sample: Optional[bool]) -> Tuple[float, float, bool]:
        """Get expert-specific decoding parameters based on mode."""
        # Expert-specific defaults
        expert_params = {
            "engine": {
                "temperature": 0.0,
                "top_p": 1.0,
                "do_sample": False
            },
            "tutor": {
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True
            },
            "director": {
                "temperature": 0.6,
                "top_p": 0.9,
                "do_sample": True
            }
        }

        # Get expert defaults
        normalized_mode = "engine" if mode == "uci" else mode
        expert_defaults = expert_params.get(normalized_mode, expert_params["tutor"])

        # Use provided values or expert defaults
        final_temperature = temperature if temperature is not None else expert_defaults["temperature"]
        final_top_p = top_p if top_p is not None else expert_defaults["top_p"]
        final_do_sample = do_sample if do_sample is not None else expert_defaults["do_sample"]

        return final_temperature, final_top_p, final_do_sample

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache performance statistics."""
        with self._cache_lock:
            total_requests = self._total_requests
            cache_hits = self._cache_hits
            response_cache_size = len(self._response_cache)
            engine_cache_size = len(self._engine_cache)

        return {
            'total_requests': total_requests,
            'cache_hits': cache_hits,
            'cache_hit_rate': cache_hits / max(total_requests, 1),
            'response_cache_size': response_cache_size,
            'engine_cache_size': engine_cache_size,
            'cache_max_size': self.max_cache_size,
        }

    def clear_caches(self):
        """Clear all performance caches."""
        with self._cache_lock:
            self._response_cache.clear()
            self._cache_hits = 0
        self._kv_cache.clear()
        logger.info("Inference caches cleared")

    def get_cache_info(self) -> Dict[str, Any]:
        """Get detailed cache information."""
        with self._cache_lock:
            return {
                'response_cache_size': len(self._response_cache),
                'engine_cache_size': len(self._engine_cache),
                'cache_hits': self._cache_hits,
                'total_requests': self._total_requests,
                'hit_rate': self._cache_hits / max(self._total_requests, 1),
                'max_size': self.max_cache_size,
            }
