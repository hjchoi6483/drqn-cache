"""Synthetic YCSB-style access-sequence generators (Experiment A).

These traces emulate the access patterns of the standard YCSB core workloads
without any external download, producing a sequence of integer key ids suitable
for the existing cache simulator.

Page-access simplification (deliberate)
---------------------------------------
The cache simulator consumes only a sequence of *touched* key ids. A YCSB
operation (read or update) touches exactly one key/page, and the read/write
distinction does not change *which* page is touched. We therefore collapse every
YCSB operation to a single key access and ignore the read/update ratio. This is
consistent with treating the buffer cache as page-access driven. Under this
model, workloads ``a`` (50/50), ``b`` (95/5), and ``c`` (100/0) differ only in
their read/update mix and are therefore *statistically equivalent* access-key
sequences; they are still generated as separate scenarios so the paper can
report them distinctly and so future write-aware extensions slot in cleanly.

Bounded Zipf sampler
--------------------
We do NOT use ``numpy.random.zipf`` (which is unbounded and then modded, which
distorts the tail). Instead we precompute normalized cumulative weights
``p_i proportional to 1 / i**zipf_const`` for ``i = 1..vocab_size`` and sample via
inverse-CDF (``np.searchsorted`` on the cumulative array with uniform randoms).
This gives a faithful, properly bounded Zipfian over ``1..vocab_size`` and stays
vectorized for large request counts.

Workloads
---------
``a``/``b``/``c``
    Keys drawn from a bounded Zipfian over ``1..vocab_size`` with constant
    ``zipf_const`` (YCSB's default ~0.99).
``d`` (read-latest, non-stationary)
    Recently inserted keys are most likely to be read. See
    :func:`_trace_ycsb_latest` for the exact mechanism.

All generators return ``List[int]`` of length ``num_requests`` with ids in
``1..vocab_size`` (id ``0`` -- the env's reserved "empty slot" -- is never
emitted). Determinism is inherited from the caller, which must seed numpy via
``set_seed_fn(seed)`` before construction.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np


def _bounded_zipf_cdf(vocab_size: int, zipf_const: float) -> np.ndarray:
    """Normalized cumulative weights for a bounded Zipf over ``1..vocab_size``.

    ``weight_i proportional to 1 / i**zipf_const`` for ``i = 1..vocab_size``. The
    returned array has length ``vocab_size`` and is suitable for inverse-CDF
    sampling via :func:`np.searchsorted`.
    """
    ranks = np.arange(1, vocab_size + 1, dtype=np.float64)
    weights = 1.0 / np.power(ranks, zipf_const)
    cdf = np.cumsum(weights)
    cdf /= cdf[-1]
    return cdf


def _sample_bounded_zipf(cdf: np.ndarray, size: int) -> np.ndarray:
    """Inverse-CDF sample ``size`` ids in ``1..len(cdf)`` from a bounded Zipf.

    Returns an int64 array of ids (rank index + 1).
    """
    u = np.random.random(size=size)
    idx = np.searchsorted(cdf, u, side="right")
    # searchsorted can return len(cdf) for u extremely close to 1.0; clamp it.
    np.clip(idx, 0, len(cdf) - 1, out=idx)
    return (idx + 1).astype(np.int64)


def _trace_ycsb_zipfian(num_requests: int, vocab_size: int, zipf_const: float) -> List[int]:
    """Stationary bounded-Zipf key sequence (YCSB a/b/c access model)."""
    cdf = _bounded_zipf_cdf(vocab_size, zipf_const)
    ids = _sample_bounded_zipf(cdf, num_requests)
    return ids.tolist()


def _trace_ycsb_latest(num_requests: int, vocab_size: int, zipf_const: float) -> List[int]:
    """YCSB-D read-latest: popularity drifts toward newer keys over time.

    Mechanism
    ---------
    The active population grows from a small seed up to ``vocab_size`` by
    inserting one new key id (the next integer in ``1..vocab_size``) every
    ``insert_every`` requests, until the full vocabulary has been inserted. At
    each request a read targets a key near the most-recently-inserted end: we draw
    a bounded-Zipf rank ``r`` (``r = 0`` is most popular) over the *current*
    population size and map it so that rank 0 selects the newest key and larger
    ranks select progressively older keys. Concretely, with current population
    size ``m`` and inserted ids ``1..m``, the read key is ``m - r``.

    Because the newest id increases over time and reads concentrate on the newest
    end, the popular set drifts toward larger ids as the stream advances --
    capturing YCSB-D's temporal non-stationarity. The full bounded-Zipf CDF is
    precomputed once; since the population only changes every ``insert_every``
    requests, requests are processed in vectorized population-blocks (one
    ``searchsorted`` per block) for speed.
    """
    # Insert schedule: spread vocab_size insertions across the stream, starting
    # from a small seed population so there is always something to read.
    seed_pop = max(1, min(vocab_size, 10))
    remaining_inserts = max(0, vocab_size - seed_pop)
    insert_every = max(1, num_requests // max(1, remaining_inserts)) if remaining_inserts else num_requests + 1

    full_cdf = _bounded_zipf_cdf(vocab_size, zipf_const)

    ids = np.empty(num_requests, dtype=np.int64)
    i = 0
    while i < num_requests:
        # population(i) = min(vocab_size, seed_pop + i // insert_every); all
        # requests in this block share the same population value.
        population = min(vocab_size, seed_pop + i // insert_every)
        if population >= vocab_size:
            block_end = num_requests
        else:
            block_end = min(num_requests, (population - seed_pop + 1) * insert_every)
        block = block_end - i

        # Bounded-Zipf rank over the current population, renormalized from the
        # precomputed full CDF prefix, sampled by inverse-CDF.
        prefix = full_cdf[:population]
        u = np.random.random(size=block) * prefix[-1]
        r = np.searchsorted(prefix, u, side="right")
        np.clip(r, 0, population - 1, out=r)
        # rank 0 -> newest key (id == population); larger rank -> older key.
        ids[i:block_end] = population - r
        i = block_end

    return ids.tolist()


def trace_ycsb(
    num_requests: int,
    vocab_size: int,
    workload: str,
    zipf_const: float = 0.99,
    seed_for_latest: Optional[int] = None,
) -> List[int]:
    """Generate a synthetic YCSB-style access-key sequence.

    ``workload`` is one of ``{"a", "b", "c", "d"}``. ``a``/``b``/``c`` produce a
    stationary bounded-Zipf sequence (statistically equivalent under the
    page-access model); ``d`` produces the read-latest non-stationary sequence.
    ``seed_for_latest`` is accepted for interface symmetry; determinism is driven
    by the caller's ``set_seed_fn`` so it is unused here.
    """
    del seed_for_latest  # determinism comes from the caller's numpy seeding.

    workload = workload.lower()
    if workload in ("a", "b", "c"):
        return _trace_ycsb_zipfian(num_requests, vocab_size, zipf_const)
    if workload == "d":
        return _trace_ycsb_latest(num_requests, vocab_size, zipf_const)
    raise ValueError(f"Unknown YCSB workload: {workload}")
