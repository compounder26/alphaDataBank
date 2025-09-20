"""
Cache Utilities

Caching functionality for the dashboard to improve performance.
"""

import functools
import time
from typing import Any, Dict, Optional


class SimpleCache:
    """Simple in-memory cache with TTL support."""

    def __init__(self, default_ttl: int = 300):
        """
        Initialize cache.

        Args:
            default_ttl: Default time-to-live in seconds
        """
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = default_ttl

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        if key in self.cache:
            entry = self.cache[key]
            if time.time() < entry['expires']:
                return entry['value']
            else:
                del self.cache[key]
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with TTL."""
        ttl = ttl or self.default_ttl
        self.cache[key] = {
            'value': value,
            'expires': time.time() + ttl
        }

    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()

    def cleanup_expired(self) -> None:
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.cache.items()
            if current_time >= entry['expires']
        ]
        for key in expired_keys:
            del self.cache[key]


# Global cache instance
_cache = SimpleCache()


def cached(ttl: int = 300):
    """
    Decorator to cache function results.

    Args:
        ttl: Time-to-live in seconds
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"

            # Try to get from cache
            result = _cache.get(cache_key)
            if result is not None:
                return result

            # Execute function and cache result
            result = func(*args, **kwargs)
            _cache.set(cache_key, result, ttl)
            return result

        return wrapper
    return decorator


def clear_cache():
    """Clear all cached data."""
    _cache.clear()


def cleanup_cache():
    """Remove expired cache entries."""
    _cache.cleanup_expired()


def get_cache_stats() -> Dict[str, int]:
    """Get cache statistics."""
    total_entries = len(_cache.cache)
    current_time = time.time()
    expired_entries = sum(
        1 for entry in _cache.cache.values()
        if current_time >= entry['expires']
    )

    return {
        'total_entries': total_entries,
        'active_entries': total_entries - expired_entries,
        'expired_entries': expired_entries
    }