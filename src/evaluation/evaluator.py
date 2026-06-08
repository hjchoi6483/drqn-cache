from __future__ import annotations

import hashlib
from typing import Callable, Dict, List, Tuple


BaselineKey = Tuple[str, float, int, str, Tuple[str, ...], int, str]


def _trace_fingerprint(test_stream: List[int]) -> str:
    h = hashlib.sha1()
    for req in test_stream:
        h.update(int(req).to_bytes(8, byteorder="little", signed=True))
    return h.hexdigest()


# Baselines are computed once per scenario/cache/eval_kind/baseline set and cached.
def compute_baselines_once(
    scenario: str,
    alpha: float,
    cache_size: int,
    eval_kind: str,
    test_stream: List[int],
    baseline_names: List[str],
    baseline_cache: Dict[BaselineKey, Dict[str, float]],
    build_baselines_fn: Callable[..., Dict[str, object]],
) -> Dict[str, float]:
    names_key = tuple(sorted(baseline_names))
    trace_fp = _trace_fingerprint(test_stream)
    key: BaselineKey = (
        scenario,
        float(alpha),
        int(cache_size),
        str(eval_kind),
        names_key,
        len(test_stream),
        trace_fp,
    )
    if key in baseline_cache:
        return baseline_cache[key]

    baselines = build_baselines_fn(baseline_names, cache_size, trace=test_stream)
    for req in test_stream:
        for baseline in baselines.values():
            baseline.access(req)

    out = {
        f"baseline_hit_{name}": float(baseline.stats()["hit_rate"])
        for name, baseline in baselines.items()
    }
    baseline_cache[key] = out
    return out


def evaluate_policy_with_baselines(
    model,
    scenario: str,
    alpha: float,
    test_stream: List[int],
    cache_size: int,
    s,
    eval_kind: str,
    baseline_names: List[str],
    baseline_cache: Dict[BaselineKey, Dict[str, float]],
    build_baselines_fn: Callable[..., Dict[str, object]],
    make_env_fn,
    make_obs_fn,
    select_action_fn,
):
    model.eval()
    env = make_env_fn(cache_size, s)
    hidden = model.init_hidden(1)

    # RL hit rate is based on env.step() hit boolean; bypass is a miss by definition.
    rl_hits = 0
    rl_miss = 0
    admissions = 0
    evictions = 0
    for req in test_stream:
        obs = make_obs_fn(env, req)
        a, hidden = select_action_fn(model, obs, hidden, eps=0.0)
        _r, hit = env.step(req, a)
        step_info = getattr(env, "last_step_info", {})
        if hit:
            rl_hits += 1
        else:
            rl_miss += 1
            if step_info.get("admit", False):
                admissions += 1
            if step_info.get("evict", False):
                evictions += 1

    total = rl_hits + rl_miss
    rl_hit = (rl_hits / total) * 100.0 if total else 0.0
    admission_rate = admissions / max(1, rl_miss)

    baseline_res = compute_baselines_once(
        scenario=scenario,
        alpha=alpha,
        cache_size=cache_size,
        eval_kind=eval_kind,
        test_stream=test_stream,
        baseline_names=baseline_names,
        baseline_cache=baseline_cache,
        build_baselines_fn=build_baselines_fn,
    )

    out = {
        "rl_hit": float(rl_hit),
        "read_hit_rate": float(rl_hit),
        "total_lookup_hit_rate": float(rl_hit),
        "admission_rate": float(admission_rate),
        "eviction_count": float(evictions),
        "lookup_count": float(total),
        "read_lookup_count": float(total),
    }
    out.update(baseline_res)
    for name in baseline_names:
        hit = baseline_res.get(f"baseline_hit_{name}", 0.0)
        out[f"rl_minus_baseline_{name}"] = float(rl_hit - hit)
    return out
