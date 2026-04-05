from __future__ import annotations

from typing import Callable, Dict, List, Tuple

from .zipf import trace_zipf, trace_zipf_phase_shift, trace_zipf_random_jump


ConfigDict = Dict[str, object]


def build_trace(
    config: ConfigDict,
    scenario: str,
    alpha: float,
    seed: int,
    set_seed_fn: Callable[[int], None],
) -> List[int]:
    reqs, _ = build_trace_with_meta(config, scenario, alpha, seed, set_seed_fn)
    return reqs


def build_trace_with_meta(
    config: ConfigDict,
    scenario: str,
    alpha: float,
    seed: int,
    set_seed_fn: Callable[[int], None],
) -> Tuple[List[int], Dict[str, object]]:
    set_seed_fn(seed)
    nreq = int(config["NUM_REQUESTS"])
    vocab = int(config["VOCAB_SIZE"])

    if scenario in {"zipf", "zipf_static"}:
        return trace_zipf(nreq, vocab, alpha), {
            "scenario": "zipf_static",
            "alpha": float(alpha),
        }

    if scenario == "zipf_phase_shift":
        reqs, meta = trace_zipf_phase_shift(
            num_requests=nreq,
            vocab_size=vocab,
            phase_alphas=[float(x) for x in config["PHASE_ALPHAS"]],
            switch_every=int(config["ALPHA_SWITCH_EVERY"]),
            mode=str(config["ALPHA_SCHEDULE_MODE"]),
        )
        meta["scenario"] = scenario
        return reqs, meta

    if scenario == "zipf_random_jump":
        reqs, meta = trace_zipf_random_jump(
            num_requests=nreq,
            vocab_size=vocab,
            phase_alphas=[float(x) for x in config["PHASE_ALPHAS"]],
            switch_every=int(config["ALPHA_SWITCH_EVERY"]),
            mode=str(config["ALPHA_SCHEDULE_MODE"]),
            jump_prob=float(config["ALPHA_JUMP_PROB"]),
        )
        meta["scenario"] = scenario
        return reqs, meta

    raise ValueError(scenario)
