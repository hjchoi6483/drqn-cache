"""W-TinyLFU baseline: windowed TinyLFU admission with LRU window."""
from __future__ import annotations

from collections import OrderedDict, deque
from typing import Deque, Dict


class WTinyLFUCacheSim:
    """Deterministic simplified W-TinyLFU (window LRU + TinyLFU admission to main)."""

    def __init__(self, capacity: int, window_ratio: float = 0.01, recent_window: int = 10000):
        self.capacity = int(capacity)
        self.window_ratio = float(window_ratio)
        self.recent_window = int(recent_window)
        self.reset(self.capacity)

    def reset(self, capacity: int | None = None):
        if capacity is not None:
            self.capacity = int(capacity)
        self.window_cap = max(1, int(self.capacity * self.window_ratio)) if self.capacity > 0 else 0
        self.main_cap = max(0, self.capacity - self.window_cap)

        self.window = OrderedDict()
        self.main = OrderedDict()

        self.recent: Deque[int] = deque(maxlen=max(1, self.recent_window))
        self.freq: Dict[int, int] = {}

        self.hits = 0
        self.misses = 0

    def _observe(self, key: int):
        if len(self.recent) == self.recent.maxlen:
            old = self.recent[0]
            self.freq[old] = max(0, self.freq.get(old, 1) - 1)
            if self.freq[old] == 0:
                self.freq.pop(old, None)
        self.recent.append(key)
        self.freq[key] = self.freq.get(key, 0) + 1

    def _f(self, key: int) -> int:
        return int(self.freq.get(key, 0))

    def access(self, key: int) -> bool:
        self._observe(key)
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
        self.window.move_to_end(key)

        candidate = None
        if len(self.window) > self.window_cap:
            candidate, _ = self.window.popitem(last=False)

        if candidate is not None and self.main_cap > 0:
            if len(self.main) < self.main_cap:
                self.main[candidate] = True
            else:
                victim = next(iter(self.main))
                if self._f(candidate) >= self._f(victim):
                    self.main.popitem(last=False)
                    self.main[candidate] = True
        return False

    def stats(self) -> Dict[str, float]:
        total = self.hits + self.misses
        return {
            "hits": float(self.hits),
            "misses": float(self.misses),
            "hit_rate": float((self.hits / total) * 100.0 if total else 0.0),
        }
