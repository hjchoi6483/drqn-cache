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
    names = ["lru", "lfu", "lruk", "2q", "arc", "tinylfu", "belady"]

    baselines = build_baselines(names, capacity, trace=trace)
    results = {}
    for name in names:
        results[name] = run_trace(baselines[name], trace)

    assert results["belady"]["hit_rate"] >= results["lru"]["hit_rate"]

    tiny = baselines["tinylfu"]
    tstats = tiny.stats()
    assert int(tstats["hits"] + tstats["misses"]) == len(trace)
    assert math.isfinite(float(tstats["hit_rate"]))
    assert len(tiny.cache) <= capacity

    keep_trace = [1] * 20 + list(range(100, 130)) + [1] * 20
    tiny2 = build_baselines(["tinylfu"], capacity=3, trace=keep_trace)["tinylfu"]
    for req in keep_trace:
        tiny2.access(req)
    assert 1 in tiny2.cache

    for name in names:
        baselines[name].reset(capacity)
        st = baselines[name].stats()
        assert st["hits"] == 0.0
        assert st["misses"] == 0.0

    ts2 = baselines["tinylfu"].stats()
    assert ts2["admissions"] == 0.0
    assert ts2["bypasses"] == 0.0

    print("baseline tests: ok")


if __name__ == "__main__":
    main()
