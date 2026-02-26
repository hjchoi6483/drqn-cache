from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def _phase_histograms(trace: Sequence[int], vocab_size: int, phases: int) -> List[np.ndarray]:
    phases = max(1, int(phases))
    arr = np.asarray(trace, dtype=np.int64)
    phase_len = max(1, int(np.ceil(len(arr) / phases)))
    hists: List[np.ndarray] = []
    for p in range(phases):
        s = p * phase_len
        e = min(len(arr), (p + 1) * phase_len)
        if s >= len(arr):
            break
        seg = arr[s:e]
        counts = np.bincount(seg - 1, minlength=vocab_size).astype(np.float64)
        probs = counts / max(1.0, counts.sum())
        hists.append(probs)
    return hists


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-12
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * (kl_pm + kl_qm))


def _topk_overlap(p: np.ndarray, q: np.ndarray, k: int) -> float:
    k = max(1, min(k, len(p)))
    p_top = set(np.argpartition(-p, k - 1)[:k].tolist())
    q_top = set(np.argpartition(-q, k - 1)[:k].tolist())
    return float(len(p_top & q_top) / k)


def workload_diagnostics(
    trace: Sequence[int],
    vocab_size: int,
    phases: int,
    top_k: int = 100,
) -> Dict[str, float]:
    hists = _phase_histograms(trace, vocab_size=vocab_size, phases=phases)
    if len(hists) < 2:
        return {
            "diag_phase_count": float(len(hists)),
            "diag_js_mean": 0.0,
            "diag_js_max": 0.0,
            "diag_topk_overlap_mean": 1.0,
        }

    js_vals = []
    overlap_vals = []
    for i in range(len(hists) - 1):
        js_vals.append(_js_divergence(hists[i], hists[i + 1]))
        overlap_vals.append(_topk_overlap(hists[i], hists[i + 1], top_k))

    return {
        "diag_phase_count": float(len(hists)),
        "diag_js_mean": float(np.mean(js_vals)),
        "diag_js_max": float(np.max(js_vals)),
        "diag_topk_overlap_mean": float(np.mean(overlap_vals)),
    }
