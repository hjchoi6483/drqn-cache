from __future__ import annotations

import importlib.util
import random
from typing import List

if importlib.util.find_spec("numpy") is not None:
    import numpy as np
else:
    np = None


def _zipf_like_sample(alpha: float) -> int:
    if alpha <= 1.0:
        raise ValueError("alpha must be > 1.0")
    # Continuous Pareto inverse-CDF approximation used only when NumPy is not
    # installed in a lightweight smoke-test environment.
    return max(1, int(random.random() ** (-1.0 / (alpha - 1.0))))


def trace_zipf(num_requests: int, vocab_size: int, alpha: float) -> List[int]:
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if np is not None:
        samples = np.random.zipf(a=alpha, size=num_requests)
        ids = (samples % vocab_size) + 1
        return ids.astype(np.int64).tolist()
    return [(_zipf_like_sample(alpha) % vocab_size) + 1 for _ in range(num_requests)]
