from __future__ import annotations

from typing import List

import numpy as np


def hotshift_offset_step(vocab_size: int, mode: str, custom_step: int) -> int:
    if mode == "half_plus_one":
        return (vocab_size // 2) + 1
    if mode == "custom":
        if custom_step <= 0:
            raise ValueError("HOTSHIFT_OFFSET_STEP_CUSTOM must be positive.")
        return custom_step
    raise ValueError(mode)


def trace_hotshift(
    num_requests: int,
    vocab_size: int,
    alpha: float,
    phases: int,
    offset_step_mode: str,
    offset_step_custom: int,
) -> List[int]:
    ranks = np.random.zipf(a=alpha, size=num_requests)
    ranks0 = ((ranks - 1) % vocab_size).astype(np.int64)

    phase_len = max(1, num_requests // phases)
    step = hotshift_offset_step(vocab_size, offset_step_mode, offset_step_custom)

    out = np.empty((num_requests,), dtype=np.int64)
    for i in range(num_requests):
        p = min(i // phase_len, phases - 1)
        offset = (p * step) % vocab_size
        out[i] = ((ranks0[i] + offset) % vocab_size) + 1

    return out.tolist()
