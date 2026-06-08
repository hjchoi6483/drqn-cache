from __future__ import annotations

import os
from typing import Callable, Dict, List

from .zipf import trace_zipf
from .ycsb.generate_ycsb_trace import generate_events, write_jsonl_trace
from src.traces.ycsb_loader import load_ycsb_lookup_stream


ConfigDict = Dict[str, object]


def build_trace(
    config: ConfigDict,
    scenario: str,
    alpha: float,
    seed: int,
    set_seed_fn: Callable[[int], None],
) -> List[int]:
    set_seed_fn(seed)
    nreq = int(config["NUM_REQUESTS"])
    vocab = int(config["VOCAB_SIZE"])

    if scenario == "zipf":
        return trace_zipf(nreq, vocab, alpha)
    if scenario.startswith("ycsb_"):
        workload = scenario.split("_", 1)[1].upper()
        trace_path = str(config.get("TRACE_PATH") or "")
        if not trace_path:
            trace_dir = os.path.join(str(config.get("OUT_DIR", "out")), "traces")
            os.makedirs(trace_dir, exist_ok=True)
            trace_path = os.path.join(
                trace_dir,
                f"ycsb_{workload}_records{vocab}_ops{nreq}_seed{seed}_alpha{float(alpha)}.jsonl",
            )
            if not os.path.exists(trace_path):
                events = generate_events(
                    workload=workload,
                    recordcount=vocab,
                    operationcount=nreq,
                    seed=seed,
                    zipf_alpha=float(alpha),
                )
                write_jsonl_trace(events, trace_path)
        return load_ycsb_lookup_stream(trace_path, read_hit_rate_only=True)
    raise ValueError(scenario)
