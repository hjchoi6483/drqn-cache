from __future__ import annotations

from typing import Callable, Dict, List

from .nonstationary import trace_zipf_hotshift, trace_zipf_shift
from .ycsb import trace_ycsb
from .zipf import trace_zipf


ConfigDict = Dict[str, object]

# YCSB scenarios reuse the loop's float ``alpha`` slot to carry the Zipfian
# constant (for run-id readability). If the slot holds a plausible YCSB constant
# we use it directly; otherwise we fall back to the config default.
YCSB_ALPHA_AS_CONST_RANGE = (0.0, 1.2)


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

    if scenario == "shift":
        # The loop's alpha slot is the start (early/stationary) skew.
        alpha_start = float(alpha)
        alpha_end = float(config["SHIFT_ALPHA_TO"])
        shift_frac = float(config.get("SHIFT_FRAC", 0.7))
        return trace_zipf_shift(nreq, vocab, alpha_start, alpha_end, shift_frac)

    if scenario == "hotshift":
        period = int(config["HOTSHIFT_PERIOD"])
        return trace_zipf_hotshift(nreq, vocab, float(alpha), period)

    if scenario in ("ycsb_a", "ycsb_b", "ycsb_c", "ycsb_d"):
        workload = scenario.split("_")[1]
        lo, hi = YCSB_ALPHA_AS_CONST_RANGE
        if lo < float(alpha) <= hi:
            zipf_const = float(alpha)
        else:
            zipf_const = float(config.get("YCSB_ZIPF_CONST", 0.99))
        return trace_ycsb(nreq, vocab, workload, zipf_const=zipf_const)

    raise ValueError(scenario)
