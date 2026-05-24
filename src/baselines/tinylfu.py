"""TinyLFU baseline: online frequency-aware admission policy."""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List


class _CountMinSketch:
    def __init__(self, width: int, depth: int):
        self.width = max(1, int(width))
        self.depth = max(1, int(depth))
        self.rows: List[List[int]] = [[0] * self.width for _ in range(self.depth)]
        self.seeds = [0x9E3779B1 ^ (i * 0x85EBCA77) for i in range(self.depth)]

    def _index(self, key: int, seed: int) -> int:
        h = hash((int(key), seed)) & 0x7FFFFFFF
        return h % self.width

    def increment(self, key: int, amount: int = 1):
        for row_i, seed in enumerate(self.seeds):
            self.rows[row_i][self._index(key, seed)] += amount

    def estimate(self, key: int) -> int:
        return min(self.rows[row_i][self._index(key, seed)] for row_i, seed in enumerate(self.seeds))

    def halve(self):
        for row in self.rows:
            for i, v in enumerate(row):
                row[i] = v >> 1


class TinyLFUCacheSim:
    """Online deterministic TinyLFU-style admission with Count-Min Sketch and doorkeeper."""

    def __init__(self, capacity: int, sketch_depth: int = 4):
        self.capacity = int(capacity)
        self.sketch_depth = int(sketch_depth)
        self.reset(self.capacity)

    def reset(self, capacity: int | None = None):
        if capacity is not None:
            self.capacity = int(capacity)
        width = max(64, self.capacity * 16)
        self.sketch = _CountMinSketch(width=width, depth=self.sketch_depth)
        self.doorkeeper = set()
        self.sample_size = max(10 * max(1, self.capacity), 1000)
        self.sampled = 0

        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.admissions = 0
        self.bypasses = 0

    def _observe(self, key: int):
        self.sampled += 1
        if key in self.doorkeeper:
            self.sketch.increment(key, 1)
        else:
            self.doorkeeper.add(key)

        if self.sampled >= self.sample_size:
            self.sketch.halve()
            self.doorkeeper.clear()
            self.sampled = 0

    def _estimated_freq(self, key: int) -> int:
        return self.sketch.estimate(key) + (1 if key in self.doorkeeper else 0)

    def access(self, key: int) -> bool:
        self._observe(key)
        if self.capacity <= 0:
            self.misses += 1
            return False

        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return True

        self.misses += 1
        if len(self.cache) < self.capacity:
            self.cache[key] = True
            self.admissions += 1
            return False

        victim = next(iter(self.cache))
        if self._estimated_freq(key) > self._estimated_freq(victim):
            self.cache.popitem(last=False)
            self.cache[key] = True
            self.admissions += 1
        else:
            self.bypasses += 1
        return False

    def stats(self) -> Dict[str, float]:
        total = self.hits + self.misses
        hit_rate = (self.hits / total) * 100.0 if total else 0.0
        return {
            "hits": float(self.hits),
            "misses": float(self.misses),
            "hit_rate": float(hit_rate),
            "admissions": float(self.admissions),
            "bypasses": float(self.bypasses),
        }
