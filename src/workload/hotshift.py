from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return abs(a)


def _to_coprime(step: int, vocab_size: int) -> int:
    s = max(1, int(step)) % vocab_size
    if s == 0:
        s = 1
    while _gcd(s, vocab_size) != 1:
        s = (s + 1) % vocab_size
        if s == 0:
            s = 1
    return s


def hotshift_offset_step(vocab_size: int, mode: str, custom_step: int, rng: np.random.Generator) -> int:
    if mode == "half_plus_one":
        return _to_coprime((vocab_size // 2) + 1, vocab_size)
    if mode == "custom":
        if custom_step <= 0:
            raise ValueError("HOTSHIFT_OFFSET_STEP_CUSTOM must be positive.")
        return _to_coprime(custom_step, vocab_size)
    if mode == "coprime_stride":
        if custom_step > 0:
            return _to_coprime(custom_step, vocab_size)
        # deterministic-ish fallback from RNG/size
        raw = int(rng.integers(1, vocab_size))
        return _to_coprime(raw, vocab_size)
    if mode == "random_stride":
        return _to_coprime(int(rng.integers(1, vocab_size)), vocab_size)
    raise ValueError(mode)


def _parse_phase_skew(spec: str | Sequence[float]) -> List[float]:
    if isinstance(spec, str):
        values = [v.strip() for v in spec.split(",") if v.strip()]
        if not values:
            return [1.0]
        return [max(0.1, float(v)) for v in values]
    vals = [max(0.1, float(v)) for v in spec]
    return vals or [1.0]


def trace_hotshift(
    num_requests: int,
    vocab_size: int,
    alpha: float,
    phases: int,
    offset_step_mode: str,
    offset_step_custom: int,
    phase_skew: str | Sequence[float] = "1.0",
    mix_ratio: float = 0.9,
    transition: str = "abrupt",
    random_seed: int | None = None,
) -> List[int]:
    phases = max(1, int(phases))
    mix_ratio = float(np.clip(mix_ratio, 0.0, 1.0))
    rng = np.random.default_rng(random_seed)

    phase_len = max(1, int(math.ceil(num_requests / phases)))
    skew = _parse_phase_skew(phase_skew)
    step = hotshift_offset_step(vocab_size, offset_step_mode, offset_step_custom, rng)

    out = np.empty((num_requests,), dtype=np.int64)
    for i in range(num_requests):
        p = min(i // phase_len, phases - 1)
        curr_alpha = max(1.01, alpha * skew[p % len(skew)])
        curr_offset = (p * step) % vocab_size

        base_rank = int(rng.zipf(a=curr_alpha))
        base_id0 = (base_rank - 1) % vocab_size

        if rng.random() > mix_ratio:
            # cold request path to induce realistic churn
            cold_jump = int(rng.integers(vocab_size // 4, vocab_size))
            base_id0 = (base_id0 + cold_jump) % vocab_size

        if transition == "smooth" and p < phases - 1:
            phase_progress = (i % phase_len) / max(1, phase_len - 1)
            next_offset = ((p + 1) * step) % vocab_size
            blend = 0.5 - 0.5 * math.cos(math.pi * phase_progress)
            eff_offset = int(round((1.0 - blend) * curr_offset + blend * next_offset)) % vocab_size
        else:
            eff_offset = curr_offset

        out[i] = ((base_id0 + eff_offset) % vocab_size) + 1

    return out.tolist()
