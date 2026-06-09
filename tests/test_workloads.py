from __future__ import annotations

import os
import random
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.workload.builder import build_trace
from src.workload.nonstationary import trace_zipf_hotshift, trace_zipf_shift
from src.workload.ycsb import trace_ycsb


# Small, CPU-only sizes for fast sanity checks (no torch, no training).
NUM_REQUESTS = 20_000
VOCAB_SIZE = 1_000


def set_seed(seed: int) -> None:
    """Torch-free seeding mirroring the runner's numpy/random seeding.

    Trace determinism depends only on numpy here, but we seed both for parity
    with the runner's set_seed contract.
    """
    random.seed(seed)
    np.random.seed(seed)


def assert_valid_trace(trace, expected_len: int, vocab: int, label: str) -> None:
    assert isinstance(trace, list), f"{label}: expected list, got {type(trace)}"
    assert len(trace) == expected_len, f"{label}: len {len(trace)} != {expected_len}"
    arr = np.asarray(trace)
    assert arr.min() >= 1, f"{label}: emitted id < 1 (min={arr.min()})"
    assert arr.max() <= vocab, f"{label}: emitted id > vocab ({arr.max()} > {vocab})"
    # id 0 is reserved by the env for the empty slot and must never appear.
    assert 0 not in set(arr.tolist()[:5000]), f"{label}: emitted reserved id 0"


def top_keys(seq, k: int) -> set:
    return {key for key, _ in Counter(seq).most_common(k)}


def test_validity_and_determinism() -> None:
    cases = [
        ("shift", lambda: trace_zipf_shift(NUM_REQUESTS, VOCAB_SIZE, 1.3, 1.8, shift_frac=0.7)),
        ("hotshift", lambda: trace_zipf_hotshift(NUM_REQUESTS, VOCAB_SIZE, 1.3, period=5_000)),
        ("ycsb_a", lambda: trace_ycsb(NUM_REQUESTS, VOCAB_SIZE, "a", zipf_const=0.99)),
        ("ycsb_b", lambda: trace_ycsb(NUM_REQUESTS, VOCAB_SIZE, "b", zipf_const=0.99)),
        ("ycsb_c", lambda: trace_ycsb(NUM_REQUESTS, VOCAB_SIZE, "c", zipf_const=0.99)),
        ("ycsb_d", lambda: trace_ycsb(NUM_REQUESTS, VOCAB_SIZE, "d", zipf_const=0.99)),
    ]
    for label, gen in cases:
        set_seed(0)
        first = gen()
        assert_valid_trace(first, NUM_REQUESTS, VOCAB_SIZE, label)

        set_seed(0)
        second = gen()
        assert first == second, f"{label}: not deterministic under identical seeding"
    print("validity + determinism: ok")


def test_shift_changes_distribution() -> None:
    set_seed(1)
    # Use clearly different skews so the regime change is unmistakable.
    trace = trace_zipf_shift(NUM_REQUESTS, VOCAB_SIZE, alpha_start=1.1, alpha_end=2.5, shift_frac=0.7)
    cut = int(NUM_REQUESTS * 0.7)
    early, late = trace[:cut], trace[cut:]

    # Mode frequency must differ markedly between the two regimes.
    early_mode_freq = Counter(early).most_common(1)[0][1] / len(early)
    late_mode_freq = Counter(late).most_common(1)[0][1] / len(late)
    assert abs(early_mode_freq - late_mode_freq) > 0.1, (
        f"shift: mode freq did not change enough ({early_mode_freq:.3f} vs {late_mode_freq:.3f})"
    )

    # Top-10 key sets should not be identical across the shift.
    assert top_keys(early, 10) != top_keys(late, 10), "shift: top-10 key sets unchanged"
    print("shift distribution change: ok")


def test_hotshift_rotates_hot_keys() -> None:
    set_seed(2)
    period = 5_000
    trace = trace_zipf_hotshift(NUM_REQUESTS, VOCAB_SIZE, alpha=1.3, period=period)
    first_block = trace[:period]
    later_block = trace[3 * period: 4 * period]
    assert top_keys(first_block, 10) != top_keys(later_block, 10), (
        "hotshift: hot-key set did not rotate between blocks"
    )
    print("hotshift rotation: ok")


def test_ycsb_d_temporal_drift() -> None:
    set_seed(3)
    trace = trace_ycsb(NUM_REQUESTS, VOCAB_SIZE, "d", zipf_const=0.99)
    quarter = NUM_REQUESTS // 4
    early_mean = float(np.mean(trace[:quarter]))
    late_mean = float(np.mean(trace[-quarter:]))
    assert late_mean > early_mean, (
        f"ycsb_d: newer keys do not dominate later (early_mean={early_mean:.1f}, late_mean={late_mean:.1f})"
    )
    print("ycsb_d temporal drift: ok")


def test_build_trace_dispatch() -> None:
    config = {
        "NUM_REQUESTS": NUM_REQUESTS,
        "VOCAB_SIZE": VOCAB_SIZE,
        "SHIFT_ALPHA_TO": 1.8,
        "SHIFT_FRAC": 0.7,
        "HOTSHIFT_PERIOD": 5_000,
        "YCSB_ZIPF_CONST": 0.99,
    }
    new_scenarios = {
        "zipf": 1.3,
        "shift": 1.3,
        "hotshift": 1.3,
        "ycsb_a": 0.99,
        "ycsb_b": 0.99,
        "ycsb_c": 0.99,
        "ycsb_d": 0.99,
    }
    for scenario, alpha in new_scenarios.items():
        trace = build_trace(config, scenario, alpha, seed=0, set_seed_fn=set_seed)
        assert_valid_trace(trace, NUM_REQUESTS, VOCAB_SIZE, f"dispatch:{scenario}")

    # build_trace must remain deterministic via the injected set_seed_fn.
    a = build_trace(config, "ycsb_d", 0.99, seed=7, set_seed_fn=set_seed)
    b = build_trace(config, "ycsb_d", 0.99, seed=7, set_seed_fn=set_seed)
    assert a == b, "dispatch: build_trace not deterministic for fixed seed"

    raised = False
    try:
        build_trace(config, "definitely_not_a_scenario", 1.0, seed=0, set_seed_fn=set_seed)
    except ValueError:
        raised = True
    assert raised, "dispatch: unknown scenario did not raise ValueError"
    print("build_trace dispatch: ok")


def main() -> None:
    test_validity_and_determinism()
    test_shift_changes_distribution()
    test_hotshift_rotates_hot_keys()
    test_ycsb_d_temporal_drift()
    test_build_trace_dispatch()
    print("workload tests: ok")


if __name__ == "__main__":
    main()
