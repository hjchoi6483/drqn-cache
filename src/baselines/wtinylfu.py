from __future__ import annotations

from collections import Counter, OrderedDict
from typing import Dict


class WTinyLFUCacheSim:
    """Online deterministic simplified W-TinyLFU with small LRU window + TinyLFU-gated main."""

    def __init__(self, capacity: int, window_ratio: float = 0.01):
        self.capacity = int(capacity)
        self.window_ratio = float(window_ratio)
        self.reset(self.capacity)

    def reset(self, capacity: int | None = None):
        if capacity is not None:
            self.capacity = int(capacity)
        if self.capacity > 0:
            self.window_cap = max(1, int(self.capacity * self.window_ratio), self.capacity // 100)
            self.window_cap = min(self.window_cap, self.capacity)
        else:
            self.window_cap = 0
        self.main_cap = max(0, self.capacity - self.window_cap)
        self.window = OrderedDict()
        self.main = OrderedDict()
        self.freq = Counter()
        self.hits = 0
        self.misses = 0

    def _admit_to_main(self, candidate: int):
        if self.main_cap <= 0:
            return
        if len(self.main) < self.main_cap:
            self.main[candidate] = True
            return
        victim = next(iter(self.main))
        if self.freq[candidate] >= self.freq[victim]:
            self.main.popitem(last=False)
            self.main[candidate] = True

    def access(self, key: int) -> bool:
        self.freq[key] += 1
        if self.capacity <= 0:
            self.misses += 1
            return False

        if key in self.window:
            self.hits += 1
            self.window.move_to_end(key)
            return True
        if key in self.main:
            self.hits += 1
            self.main.move_to_end(key)
            return True

        self.misses += 1
        self.window[key] = True
        if len(self.window) > self.window_cap:
            candidate, _ = self.window.popitem(last=False)
            self._admit_to_main(candidate)
        return False

    def stats(self) -> Dict[str, float]:
        total = self.hits + self.misses
        hit_rate = (self.hits / total) * 100.0 if total else 0.0
        return {"hits": float(self.hits), "misses": float(self.misses), "hit_rate": float(hit_rate)}
