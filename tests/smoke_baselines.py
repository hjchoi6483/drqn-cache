from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.baselines.factory import build_baselines


def run_trace(baseline, trace):
    for req in trace:
        baseline.access(req)
    stats = baseline.stats()
    assert "hit_rate" in stats
    assert math.isfinite(float(stats["hit_rate"]))
    assert int(stats["hits"] + stats["misses"]) == len(trace)
    return stats


def main():
    trace = [1, 2, 3, 1, 2, 4, 1, 2, 3, 4]
    capacity = 2
    names = ["lru", "lfu", "lruk", "2q", "arc", "tinylfu", "wtinylfu", "belady"]

    baselines = build_baselines(names, capacity, trace=trace)
    results = {}
    for name in names:
        results[name] = run_trace(baselines[name], trace)

    assert results["belady"]["hit_rate"] >= results["lru"]["hit_rate"]

    for name in names:
        baselines[name].reset(capacity)
        assert baselines[name].stats()["hits"] == 0.0
        assert baselines[name].stats()["misses"] == 0.0

    print("smoke_baselines: ok")


if __name__ == "__main__":
    main()
