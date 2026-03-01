from __future__ import annotations

from typing import Callable, Dict, List

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
    raise ValueError(scenario)
