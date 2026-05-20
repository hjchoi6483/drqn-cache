from __future__ import annotations

from collections import OrderedDict, deque
from typing import Deque, Dict


class TwoQCacheSim:
    """Online deterministic simplified 2Q simulator using A1in/A1out/Am."""

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.reset(self.capacity)

    def reset(self, capacity: int | None = None):
        if capacity is not None:
            self.capacity = int(capacity)
        self.a1in_target = max(1, self.capacity // 4) if self.capacity > 0 else 0
        self.a1out_max = max(0, self.capacity)
        self.a1in: Deque[int] = deque()
        self.a1in_set = set()
        self.am = OrderedDict()
        self.a1out: Deque[int] = deque()
        self.a1out_set = set()
        self.hits = 0
        self.misses = 0

    def _push_a1out(self, key: int):
        if key in self.a1out_set:
            return
        self.a1out.append(key)
        self.a1out_set.add(key)
        if len(self.a1out) > self.a1out_max:
            old = self.a1out.popleft()
            self.a1out_set.remove(old)

    def _ensure_space(self):
        resident = len(self.a1in_set) + len(self.am)
        if resident < self.capacity:
            return
        if len(self.a1in_set) > self.a1in_target:
            old = self.a1in.popleft()
            self.a1in_set.remove(old)
            self._push_a1out(old)
        elif self.am:
            self.am.popitem(last=False)
        elif self.a1in:
            old = self.a1in.popleft()
            self.a1in_set.remove(old)
            self._push_a1out(old)

    def access(self, key: int) -> bool:
        if self.capacity <= 0:
            self.misses += 1
            return False

        if key in self.am:
            self.hits += 1
            self.am.move_to_end(key)
            return True

        if key in self.a1in_set:
            self.hits += 1
            self.a1in_set.remove(key)
            self.a1in.remove(key)
            self._ensure_space()
            self.am[key] = True
            return True

        self.misses += 1
        if key in self.a1out_set:
            self.a1out_set.remove(key)
            self.a1out.remove(key)
            self._ensure_space()
            self.am[key] = True
            return False

        self._ensure_space()
        self.a1in.append(key)
        self.a1in_set.add(key)
        if len(self.a1in_set) > self.a1in_target:
            old = self.a1in.popleft()
            self.a1in_set.remove(old)
            self._push_a1out(old)
        return False

    def stats(self) -> Dict[str, float]:
        total = self.hits + self.misses
        hit_rate = (self.hits / total) * 100.0 if total else 0.0
        return {"hits": float(self.hits), "misses": float(self.misses), "hit_rate": float(hit_rate)}
