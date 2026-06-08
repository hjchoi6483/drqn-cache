from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List

YCSB_WORKLOADS: Dict[str, Dict[str, float]] = {
    "A": {"READ": 0.50, "UPDATE": 0.50},
    "B": {"READ": 0.95, "UPDATE": 0.05},
    "C": {"READ": 1.00},
    "D": {"READ": 0.95, "INSERT": 0.05},
    "F": {"READ": 0.50, "READ_MODIFY_WRITE": 0.50},
}

LOOKUP_OPS = {"READ", "READ_MODIFY_WRITE"}
READ_HIT_RATE_OPS = {"READ", "READ_MODIFY_WRITE"}


def _operation_schedule(workload: str, operationcount: int, rng: random.Random) -> List[str]:
    """Return a deterministically shuffled schedule with exact rounded mix counts."""
    mix = YCSB_WORKLOADS[workload]
    ops: List[str] = []
    names = list(mix)
    remaining = int(operationcount)
    for name in names[:-1]:
        count = int(round(operationcount * mix[name]))
        count = max(0, min(count, remaining))
        ops.extend([name] * count)
        remaining -= count
    ops.extend([names[-1]] * remaining)
    rng.shuffle(ops)
    return ops


def _zipf_like_sample(alpha: float, rng: random.Random) -> int:
    return max(1, int(rng.random() ** (-1.0 / (alpha - 1.0))))


def _zipf_key(recordcount: int, alpha: float, rng: random.Random) -> int:
    if recordcount <= 0:
        raise ValueError("recordcount must be positive")
    # The existing synthetic generator maps unbounded Zipf samples back into a
    # finite key universe. Reuse the same bounded style for YCSB A/B/C/F.
    return int((_zipf_like_sample(alpha, rng) % recordcount) + 1)


def _latest_key(max_key: int, alpha: float, rng: random.Random) -> int:
    if max_key <= 0:
        raise ValueError("latest distribution requires at least one key")
    # Rank 1 is the newest key, rank 2 the next-newest, and so on. Folding the
    # unbounded Zipf sample keeps generation fast while biasing reads toward
    # newly inserted records.
    rank = int((_zipf_like_sample(alpha, rng) - 1) % max_key) + 1
    return int(max_key - rank + 1)


def _event(index: int, operation: str, key: int, workload: str) -> Dict[str, object]:
    is_lookup = operation in LOOKUP_OPS
    count_read = operation in READ_HIT_RATE_OPS
    return {
        "index": int(index),
        "timestep": int(index),
        "workload": workload,
        "operation": operation,
        "key": int(key),
        "is_cache_lookup": bool(is_lookup),
        "count_read_hit_rate": bool(count_read),
    }


def generate_events(
    workload: str,
    recordcount: int,
    operationcount: int,
    seed: int,
    zipf_alpha: float = 1.2,
) -> List[Dict[str, object]]:
    """Generate YCSB-core-style events as dictionaries.

    The generator models YCSB operations and access distributions for cache
    experiments; it is not a database benchmark driver. UPDATE and INSERT events
    are emitted in the trace but are not marked as cache lookups. YCSB-E scan
    support is intentionally excluded until the simulator can model ranges.
    """
    workload = workload.upper()
    if workload not in YCSB_WORKLOADS:
        raise ValueError(f"workload must be one of {sorted(YCSB_WORKLOADS)}")
    if recordcount <= 0:
        raise ValueError("recordcount must be positive")
    if operationcount < 0:
        raise ValueError("operationcount must be non-negative")
    if zipf_alpha <= 1.0:
        raise ValueError("zipf_alpha must be > 1.0")

    py_rng = random.Random(seed)
    ops = _operation_schedule(workload, operationcount, py_rng)

    events: List[Dict[str, object]] = []
    max_key = int(recordcount)
    for index, op in enumerate(ops):
        if workload == "D" and op == "INSERT":
            max_key += 1
            key = max_key
        elif workload == "D":
            key = _latest_key(max_key, zipf_alpha, py_rng)
        else:
            key = _zipf_key(recordcount, zipf_alpha, py_rng)
        events.append(_event(index, op, key, workload))
    return events


def write_jsonl_trace(events: Iterable[Dict[str, object]], out: str | Path) -> None:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate YCSB-style cache trace JSONL.")
    p.add_argument("--workload", choices=sorted(YCSB_WORKLOADS), required=True)
    p.add_argument("--recordcount", type=int, required=True)
    p.add_argument("--operationcount", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--zipf_alpha", type=float, default=1.2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    events = generate_events(
        workload=args.workload,
        recordcount=args.recordcount,
        operationcount=args.operationcount,
        seed=args.seed,
        zipf_alpha=args.zipf_alpha,
    )
    write_jsonl_trace(events, args.out)
    meta = {
        "workload": args.workload,
        "recordcount": args.recordcount,
        "operationcount": args.operationcount,
        "seed": args.seed,
        "zipf_alpha": args.zipf_alpha,
        "format": "jsonl",
        "note": "YCSB-E scan/range workload is TODO future work.",
    }
    Path(f"{args.out}.meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
