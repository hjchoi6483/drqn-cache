from __future__ import annotations

from typing import Callable, Dict, List

from .hotshift import trace_hotshift
from .zipf import trace_zipf


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
    if scenario == "hotshift":
        return trace_hotshift(
            num_requests=nreq,
            vocab_size=vocab,
            alpha=alpha,
            phases=int(config["HOTSHIFT_PHASES"]),
            offset_step_mode=str(config["HOTSHIFT_OFFSET_STEP_MODE"]),
            offset_step_custom=int(config["HOTSHIFT_OFFSET_STEP_CUSTOM"]),
        )
    raise ValueError(scenario)
