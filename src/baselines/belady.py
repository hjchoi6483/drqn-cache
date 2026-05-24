"""Belady baseline: offline upper bound using future trace (not online deployable)."""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict, List


class BeladyCacheSim:
    """Offline optimal upper-bound cache using full future trace (Belady MIN)."""

    def __init__(self, capacity: int, trace: List[int] | None = None):
        self.capacity = int(capacity)
        self.trace = trace
        self.reset(self.capacity)

    def reset(self, capacity: int | None = None):
        if capacity is not None:
            self.capacity = int(capacity)
        if self.trace is None:
            raise ValueError("BeladyCacheSim requires full trace via trace=... for offline evaluation.")

        self.positions: Dict[int, Deque[int]] = defaultdict(deque)
        for idx, key in enumerate(self.trace):
            self.positions[key].append(idx)
        self.cache = set()
        self.index = 0
        self.hits = 0
        self.misses = 0

    def _next_use(self, key: int) -> int:
        q = self.positions.get(key)
        if not q:
            return 10**18
        return q[0] if q else 10**18

    def _evict_one(self):
        victim = max(self.cache, key=lambda k: self._next_use(k))
        self.cache.remove(victim)

    def access(self, key: int) -> bool:
        if self.index >= len(self.trace) or self.trace[self.index] != key:
            raise ValueError("BeladyCacheSim access order must exactly match provided trace.")

        q = self.positions[key]
        if q and q[0] == self.index:
            q.popleft()

        if self.capacity <= 0:
            self.misses += 1
            self.index += 1
            return False

        if key in self.cache:
            self.hits += 1
            self.index += 1
            return True

        self.misses += 1
        if len(self.cache) >= self.capacity:
            self._evict_one()
        self.cache.add(key)
        self.index += 1
        return False

    def stats(self) -> Dict[str, float]:
        total = self.hits + self.misses
        hit_rate = (self.hits / total) * 100.0 if total else 0.0
        return {"hits": float(self.hits), "misses": float(self.misses), "hit_rate": float(hit_rate)}
