"""ARC baseline: adaptive replacement cache."""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict


class ARCCacheSim:
    """Adaptive Replacement Cache simulator (ARC).

    Maintains T1/T2 (resident) and B1/B2 (ghost) lists with adaptive target p.
    """

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.reset(self.capacity)

    def reset(self, capacity: int | None = None):
        if capacity is not None:
            self.capacity = int(capacity)
        self.t1 = OrderedDict()
        self.t2 = OrderedDict()
        self.b1 = OrderedDict()
        self.b2 = OrderedDict()
        self.p = 0.0
        self.hits = 0
        self.misses = 0

    def _replace(self, incoming_in_b2: bool):
        t1_len = len(self.t1)
        if t1_len >= 1 and (t1_len > self.p or (incoming_in_b2 and t1_len == self.p)):
            old, _ = self.t1.popitem(last=False)
            self.b1[old] = True
        else:
            if self.t2:
                old, _ = self.t2.popitem(last=False)
                self.b2[old] = True

    def access(self, key: int) -> bool:
        c = self.capacity

        if key in self.t1:
            self.hits += 1
            del self.t1[key]
            self.t2[key] = True
            return True

        if key in self.t2:
            self.hits += 1
            self.t2.move_to_end(key)
            return True

        self.misses += 1

        if key in self.b1:
            delta = 1.0 if len(self.b1) >= len(self.b2) else len(self.b2) / max(1, len(self.b1))
            self.p = min(float(c), self.p + delta)
            self._replace(incoming_in_b2=False)
            del self.b1[key]
            self.t2[key] = True
            return False

        if key in self.b2:
            delta = 1.0 if len(self.b2) >= len(self.b1) else len(self.b1) / max(1, len(self.b2))
            self.p = max(0.0, self.p - delta)
            self._replace(incoming_in_b2=True)
            del self.b2[key]
            self.t2[key] = True
            return False

        if len(self.t1) + len(self.b1) == c:
            if len(self.t1) < c:
                self.b1.popitem(last=False)
                self._replace(incoming_in_b2=False)
            else:
                self.t1.popitem(last=False)
        elif len(self.t1) + len(self.b1) < c and len(self.t1) + len(self.t2) + len(self.b1) + len(self.b2) >= c:
            if len(self.t1) + len(self.t2) + len(self.b1) + len(self.b2) >= 2 * c and self.b2:
                self.b2.popitem(last=False)
            self._replace(incoming_in_b2=False)

        self.t1[key] = True
        return False

    def stats(self) -> Dict[str, float]:
        total = self.hits + self.misses
        hit_rate = (self.hits / total) * 100.0 if total else 0.0
        return {"hits": float(self.hits), "misses": float(self.misses), "hit_rate": float(hit_rate)}
