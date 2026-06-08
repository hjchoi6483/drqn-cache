from __future__ import annotations

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.traces.ycsb_loader import event_from_mapping, lookup_stream_from_events
from src.workload.builder import build_trace
from src.workload.ycsb.generate_ycsb_trace import generate_events


def assert_close_ratio(actual: float, expected: float, tolerance: float = 0.02) -> None:
    assert abs(actual - expected) <= tolerance, (actual, expected)


def op_counts(workload: str, operationcount: int = 1000) -> Counter[str]:
    events = generate_events(workload, recordcount=100, operationcount=operationcount, seed=7)
    assert len(events) == operationcount
    return Counter(str(event["operation"]) for event in events)


def main() -> None:
    c = op_counts("C")
    assert c == {"READ": 1000}

    b = op_counts("B")
    assert_close_ratio(b["READ"] / 1000, 0.95)
    assert_close_ratio(b["UPDATE"] / 1000, 0.05)

    a = op_counts("A")
    assert_close_ratio(a["READ"] / 1000, 0.50)
    assert_close_ratio(a["UPDATE"] / 1000, 0.50)

    f = op_counts("F")
    assert_close_ratio(f["READ"] / 1000, 0.50)
    assert_close_ratio(f["READ_MODIFY_WRITE"] / 1000, 0.50)

    d_events = generate_events("D", recordcount=100, operationcount=1000, seed=7)
    d_counts = Counter(str(event["operation"]) for event in d_events)
    assert d_counts["INSERT"] == 50
    inserted_keys = [int(event["key"]) for event in d_events if event["operation"] == "INSERT"]
    assert inserted_keys
    assert min(inserted_keys) == 101
    assert max(inserted_keys) == 150

    first = generate_events("D", recordcount=100, operationcount=200, seed=123)
    second = generate_events("D", recordcount=100, operationcount=200, seed=123)
    third = generate_events("D", recordcount=100, operationcount=200, seed=124)
    assert first == second
    assert first != third

    lookup = lookup_stream_from_events(
        [event_from_mapping(event) for event in generate_events("A", recordcount=100, operationcount=1000, seed=7)]
    )
    assert len(lookup) == 500

    synthetic = build_trace(
        {"NUM_REQUESTS": 128, "VOCAB_SIZE": 50},
        scenario="zipf",
        alpha=1.3,
        seed=0,
        set_seed_fn=lambda seed: None,
    )
    assert len(synthetic) == 128

    print("ycsb tests: ok")


if __name__ == "__main__":
    main()
