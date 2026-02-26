from __future__ import annotations

from collections import OrderedDict
from typing import Dict


class LRUCacheSim:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.reset(self.capacity)

    def reset(self, capacity: int | None = None):
        if capacity is not None:
            self.capacity = int(capacity)
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0

    def access(self, key: int) -> bool:
        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
            return True

        self.misses += 1
        self.cache[key] = True
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return False

    def stats(self) -> Dict[str, float]:
        total = self.hits + self.misses
        hit_rate = (self.hits / total) * 100.0 if total else 0.0
        return {"hits": float(self.hits), "misses": float(self.misses), "hit_rate": float(hit_rate)}
