from __future__ import annotations

from typing import Dict, Iterable

from .lru import LRUCacheSim


def build_baselines(names: Iterable[str], capacity: int) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for name in names:
        if name == "lru":
            out[name] = LRUCacheSim(capacity)
            continue
        raise ValueError(f"Unknown baseline: {name}")
    return out
