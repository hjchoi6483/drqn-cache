from __future__ import annotations

from typing import Dict, Iterable, List

from .arc import ARCCacheSim
from .belady import BeladyCacheSim
from .lfu import LFUCacheSim
from .lru import LRUCacheSim
from .lruk import LRUKCacheSim
from .tinylfu import TinyLFUCacheSim
from .twoq import TwoQCacheSim


def build_baselines(names: Iterable[str], capacity: int, trace: List[int] | None = None) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for name in names:
        if name == "lru":
            out[name] = LRUCacheSim(capacity)
            continue
        if name == "arc":
            out[name] = ARCCacheSim(capacity)
            continue
        if name == "lfu":
            out[name] = LFUCacheSim(capacity)
            continue
        if name in {"lruk", "lru2"}:
            out[name] = LRUKCacheSim(capacity, k=2)
            continue
        if name == "2q":
            out[name] = TwoQCacheSim(capacity)
            continue
        if name == "tinylfu":
            out[name] = TinyLFUCacheSim(capacity)
            continue
        if name == "belady":
            out[name] = BeladyCacheSim(capacity, trace=trace)
            continue
        raise ValueError(f"Unknown baseline: {name}")
    return out
