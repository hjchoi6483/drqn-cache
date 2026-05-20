from __future__ import annotations

from collections import OrderedDict
from typing import Dict


class LFUCacheSim:
    """Online deterministic LFU simulator with LRU tie-breaks."""

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.reset(self.capacity)

    def reset(self, capacity: int | None = None):
        if capacity is not None:
            self.capacity = int(capacity)
        self.freq: Dict[int, int] = {}
        self.recency = OrderedDict()
        self.hits = 0
        self.misses = 0

    def _evict_one(self):
        min_freq = min(self.freq.values())
        for key in self.recency:
            if self.freq[key] == min_freq:
                del self.recency[key]
                del self.freq[key]
                return

    def access(self, key: int) -> bool:
        if self.capacity <= 0:
            self.misses += 1
            return False

        if key in self.freq:
            self.hits += 1
            self.freq[key] += 1
            self.recency.move_to_end(key)
            return True

        self.misses += 1
        self.freq[key] = 1
        self.recency[key] = True
        if len(self.freq) > self.capacity:
            self._evict_one()
        return False

    def stats(self) -> Dict[str, float]:
        total = self.hits + self.misses
        hit_rate = (self.hits / total) * 100.0 if total else 0.0
        return {"hits": float(self.hits), "misses": float(self.misses), "hit_rate": float(hit_rate)}
