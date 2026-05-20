from __future__ import annotations

from collections import Counter, OrderedDict, deque
from typing import Deque, Dict


class TinyLFUCacheSim:
    """Online deterministic simplified TinyLFU admission with LRU data cache."""

    def __init__(self, capacity: int, window_size: int = 1000, min_admit_count: int = 2):
        self.capacity = int(capacity)
        self.window_size = int(window_size)
        self.min_admit_count = int(min_admit_count)
        self.reset(self.capacity)

    def reset(self, capacity: int | None = None):
        if capacity is not None:
            self.capacity = int(capacity)
        self.cache = OrderedDict()
        self.total_freq = Counter()
        self.recent_freq = Counter()
        self.recent_window: Deque[int] = deque()
        self.hits = 0
        self.misses = 0
        self.bypasses = 0
        self.admissions = 0

    def _observe(self, key: int):
        self.total_freq[key] += 1
        self.recent_freq[key] += 1
        self.recent_window.append(key)
        if len(self.recent_window) > self.window_size:
            old = self.recent_window.popleft()
            self.recent_freq[old] -= 1
            if self.recent_freq[old] <= 0:
                del self.recent_freq[old]

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
        key_recent = self.recent_freq.get(key, 0)
        victim_recent = self.recent_freq.get(victim, 0)
        key_total = self.total_freq.get(key, 0)

        if key_recent >= victim_recent or key_total >= self.min_admit_count:
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
            "bypasses": float(self.bypasses),
            "admissions": float(self.admissions),
        }
