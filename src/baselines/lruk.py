from __future__ import annotations

from collections import deque
from typing import Deque, Dict


class LRUKCacheSim:
    """Online deterministic LRU-K simulator (default K=2) using access timestamps."""

    def __init__(self, capacity: int, k: int = 2):
        self.capacity = int(capacity)
        self.k = max(1, int(k))
        self.reset(self.capacity)

    def reset(self, capacity: int | None = None):
        if capacity is not None:
            self.capacity = int(capacity)
        self.clock = 0
        self.cache = set()
        self.history: Dict[int, Deque[int]] = {}
        self.hits = 0
        self.misses = 0

    def _record(self, key: int):
        if key not in self.history:
            self.history[key] = deque(maxlen=self.k)
        self.history[key].append(self.clock)

    def _evict_one(self):
        under_k = [k for k in self.cache if len(self.history.get(k, ())) < self.k]
        if under_k:
            victim = min(under_k, key=lambda x: self.history[x][-1])
        else:
            victim = min(self.cache, key=lambda x: self.history[x][0])
        self.cache.remove(victim)

    def access(self, key: int) -> bool:
        self.clock += 1
        if self.capacity <= 0:
            self._record(key)
            self.misses += 1
            return False

        if key in self.cache:
            self._record(key)
            self.hits += 1
            return True

        self._record(key)
        self.misses += 1
        self.cache.add(key)
        if len(self.cache) > self.capacity:
            self._evict_one()
        return False

    def stats(self) -> Dict[str, float]:
        total = self.hits + self.misses
        hit_rate = (self.hits / total) * 100.0 if total else 0.0
        return {"hits": float(self.hits), "misses": float(self.misses), "hit_rate": float(hit_rate)}
