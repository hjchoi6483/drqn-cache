"""Non-stationary synthetic workload generators (Experiment C).

These traces are designed so that, under the existing ``TRAIN_RATIO`` split,
training observes an early/stationary regime while evaluation observes a
non-stationary regime. They support the paper's "non-stationary" claim without
retuning: the same Zipf-tuned ``best_params.json`` is reused for evaluation.

Two scenarios are provided:

``trace_zipf_shift``
    The request distribution's Zipf skew changes partway through the stream.
    The early (stationary) segment covers training; the skew change lands inside
    the evaluation portion so the learned policy is tested on a distribution it
    did not train on.

``trace_zipf_hotshift``
    The Zipf *shape* (skew) is held fixed, but the rank->key mapping is rotated
    every ``period`` requests via a fresh permutation. This keeps the marginal
    frequency profile Zipf while changing *which* concrete keys are hot, modeling
    a drifting working set (e.g., shifting targets of interest).

Both generators map Zipf samples to ids with ``(samples % vocab_size) + 1`` --
exactly like :func:`src.workload.zipf.trace_zipf` -- so ids stay in
``1..vocab_size`` and id ``0`` (the env's reserved "empty slot") is never emitted.
Determinism is inherited from the caller, which must seed numpy via
``set_seed_fn(seed)`` before construction.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np


def trace_zipf_shift(
    num_requests: int,
    vocab_size: int,
    alpha_start: float,
    alpha_end: float,
    shift_frac: float = 0.7,
) -> List[int]:
    """Zipf trace whose skew changes once, partway through the stream.

    The first ``floor(num_requests * shift_frac)`` requests are drawn from a Zipf
    distribution with parameter ``alpha_start``; the remainder from ``alpha_end``.
    Each sample is mapped to an id with ``(sample % vocab_size) + 1``, matching
    :func:`trace_zipf`. With the default ``shift_frac=0.7`` and ``TRAIN_RATIO=0.8``
    the early stationary regime covers training while the skew change is observed
    during evaluation.
    """
    n_first = int(num_requests * shift_frac)
    n_second = num_requests - n_first

    first = np.random.zipf(a=alpha_start, size=n_first)
    second = np.random.zipf(a=alpha_end, size=n_second)
    samples = np.concatenate([first, second])
    ids = (samples % vocab_size) + 1
    return ids.astype(np.int64).tolist()


def trace_zipf_hotshift(
    num_requests: int,
    vocab_size: int,
    alpha: float,
    period: int,
    n_hot: Optional[int] = None,
) -> List[int]:
    """Zipf trace with a fixed shape but periodically rotated hot-key identities.

    A fixed Zipf shape ``alpha`` is used throughout. Zipf samples are mapped to a
    rank ``r = sample % vocab_size`` and emitted as ``perm[r]``, where ``perm`` is
    a permutation of ``1..vocab_size``. Every ``period`` requests ``perm`` is
    regenerated, so the marginal frequency profile stays Zipf while the concrete
    keys that are popular rotate over time.

    This implementation uses a *full* permutation rotation, so every rank (head
    and tail) is remapped at each rotation; ``n_hot`` is accepted for interface
    symmetry but ignored under full-permutation rotation.
    """
    del n_hot  # full-permutation rotation remaps all ranks; n_hot is unused.

    ids = np.empty(num_requests, dtype=np.int64)
    pos = 0
    while pos < num_requests:
        block = min(period, num_requests - pos)
        # Fresh permutation of 1..vocab_size for this block.
        perm = np.random.permutation(vocab_size) + 1
        samples = np.random.zipf(a=alpha, size=block)
        ranks = samples % vocab_size
        ids[pos: pos + block] = perm[ranks]
        pos += block

    return ids.tolist()
