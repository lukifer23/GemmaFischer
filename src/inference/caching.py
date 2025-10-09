#!/usr/bin/env python3
"""
Caching System for ChessGemma

Provides efficient response caching, position caching, and performance optimization
for chess inference operations.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from typing import Dict, Any, Optional, Tuple

# Import common utilities
from ..utils.common import get_logger, get_config_manager

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

        self._response_cache = OrderedDict()
        self._engine_cache = OrderedDict()
        self._kv_cache = {}

        # Performance tracking with reduced overhead
        self._cache_hits = 0
        self._total_requests = 0

        # Thread safety - use faster Lock instead of RLock for better performance
        self._cache_lock = threading.Lock()

        logger.info(f"Chess Inference Cache initialized (max_size: {self.max_cache_size})")

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
        """Check if response is cached and return a copy with current hit rate - optimized."""
        cached = self._response_cache.get(cache_key)
        if cached is None:
            return None, None

        # Update hit statistics outside the critical path for better performance
        self._cache_hits += 1
        self._response_cache.move_to_end(cache_key)

        # Calculate hit rate only when needed
        hit_rate = self._cache_hits / max(self._total_requests, 1)

        # Return cached response directly (shallow copy is still needed for safety)
        return cached.copy(), hit_rate

    def _cache_response(self, cache_key: str, response: Dict[str, Any]):
        """Cache response for future use - optimized."""
        self._response_cache[cache_key] = response.copy()

        # Maintain cache size - optimized eviction
        if len(self._response_cache) > self.max_cache_size:
            # Remove oldest item more efficiently
            self._response_cache.popitem(last=False)

    def _engine_cache_store(self, fen: Optional[str], move: str) -> None:
        """Store engine move in cache - optimized."""
        if not fen:
            return

        # Optimized LRU implementation
        if fen in self._engine_cache:
            self._engine_cache.pop(fen, None)
        self._engine_cache[fen] = move

        # Efficient eviction
        if len(self._engine_cache) > self._engine_cache_max:
            self._engine_cache.popitem(last=False)

    def _engine_cache_lookup(self, fen: Optional[str]) -> Optional[str]:
        """Lookup engine move from cache - optimized."""
        if not fen:
            return None

        mv = self._engine_cache.get(fen)
        if mv is None:
            return None

        # Refresh LRU order efficiently
        self._engine_cache.pop(fen, None)
        self._engine_cache[fen] = mv
        return mv

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
        logger.info("🧹 Inference caches cleared")

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
