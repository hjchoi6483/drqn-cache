from __future__ import annotations

from typing import List

import numpy as np


def trace_zipf(num_requests: int, vocab_size: int, alpha: float) -> List[int]:
    samples = np.random.zipf(a=alpha, size=num_requests)
    ids = (samples % vocab_size) + 1
    return ids.astype(np.int64).tolist()
