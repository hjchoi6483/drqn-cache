from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def trace_zipf(num_requests: int, vocab_size: int, alpha: float) -> List[int]:
    samples = np.random.zipf(a=alpha, size=num_requests)
    ids = (samples % vocab_size) + 1
    return ids.astype(np.int64).tolist()


def _sample_phase_zipf(phase_len: int, vocab_size: int, alpha: float) -> List[int]:
    return trace_zipf(phase_len, vocab_size, alpha)


def trace_zipf_phase_shift(
    num_requests: int,
    vocab_size: int,
    phase_alphas: List[float],
    switch_every: int,
    mode: str,
) -> Tuple[List[int], Dict[str, object]]:
    if not phase_alphas:
        raise ValueError("phase_alphas must not be empty")
    if switch_every <= 0:
        raise ValueError("switch_every must be > 0")

    req_ids: List[int] = []
    used_alphas: List[float] = []
    num_phases = int(np.ceil(num_requests / switch_every))

    for pidx in range(num_phases):
        if mode == "cyclic":
            alpha = float(phase_alphas[pidx % len(phase_alphas)])
        elif mode == "random":
            alpha = float(np.random.choice(phase_alphas))
        else:
            raise ValueError(f"Unknown alpha schedule mode: {mode}")

        phase_len = min(switch_every, num_requests - len(req_ids))
        req_ids.extend(_sample_phase_zipf(phase_len, vocab_size, alpha))
        used_alphas.append(alpha)

    meta = {
        "phase_alphas": [float(x) for x in phase_alphas],
        "switch_every": int(switch_every),
        "mode": str(mode),
        "used_phase_alphas": used_alphas,
    }
    return req_ids, meta


def _pick_random_alpha_prefer_diff(phase_alphas: List[float], prev_alpha: float | None) -> float:
    choices = [float(a) for a in phase_alphas]
    if prev_alpha is None:
        return float(np.random.choice(choices))
    diff_choices = [a for a in choices if a != float(prev_alpha)]
    if diff_choices:
        return float(np.random.choice(diff_choices))
    return float(np.random.choice(choices))


def trace_zipf_random_jump(
    num_requests: int,
    vocab_size: int,
    phase_alphas: List[float],
    switch_every: int,
    mode: str,
    jump_prob: float,
) -> Tuple[List[int], Dict[str, object]]:
    if not phase_alphas:
        raise ValueError("phase_alphas must not be empty")
    if switch_every <= 0:
        raise ValueError("switch_every must be > 0")

    jump_prob = float(np.clip(jump_prob, 0.0, 1.0))
    req_ids: List[int] = []
    used_alphas: List[float] = []
    num_phases = int(np.ceil(num_requests / switch_every))

    current_alpha: float = float(phase_alphas[0])
    current_idx = 0

    for pidx in range(num_phases):
        if pidx > 0:
            do_jump = bool(np.random.rand() < jump_prob)
            if do_jump:
                current_alpha = _pick_random_alpha_prefer_diff(phase_alphas, current_alpha)
                current_idx = int(np.argmin(np.abs(np.asarray(phase_alphas, dtype=np.float64) - current_alpha)))
            else:
                if mode == "cyclic":
                    current_idx = (current_idx + 1) % len(phase_alphas)
                    current_alpha = float(phase_alphas[current_idx])
                elif mode == "random":
                    current_alpha = _pick_random_alpha_prefer_diff(phase_alphas, current_alpha)
                    current_idx = int(np.argmin(np.abs(np.asarray(phase_alphas, dtype=np.float64) - current_alpha)))
                else:
                    raise ValueError(f"Unknown alpha schedule mode: {mode}")

        phase_len = min(switch_every, num_requests - len(req_ids))
        req_ids.extend(_sample_phase_zipf(phase_len, vocab_size, current_alpha))
        used_alphas.append(float(current_alpha))

    meta = {
        "phase_alphas": [float(x) for x in phase_alphas],
        "switch_every": int(switch_every),
        "mode": str(mode),
        "jump_prob": float(jump_prob),
        "used_phase_alphas": used_alphas,
    }
    return req_ids, meta
