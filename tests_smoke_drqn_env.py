import math

import numpy as np

from src.models.drqn import CacheEnv


def run_dynamic_adaptive_smoke():
    cfg = {
        "RECENCY_DENOM": 100.0,
        "FREQ_DENOM": 10.0,
        "HIT_EMA_ALPHA": 0.2,
        "MISS_STREAK_CLIP": 20,
        "FEATURE_SCALING_MODE": "dynamic",
        "SCALER_EMA_ALPHA": 0.2,
        "SCALER_EPS": 1e-6,
        "SCALER_MIN_SCALE": 1.0,
        "SCALER_PERCENTILE": 90.0,
        "REWARD_MODE": "adaptive",
        "REWARD_NORM_EPS": 1e-6,
        "REWARD_CLIP": 2.0,
        "INVALID_ACTION_REWARD": -1.0,
    }
    env = CacheEnv(cache_size=2, use_global=True, invalid_penalty=True, config=cfg)
    reqs = [1, 2, 1, 3, 4, 1, 2, 2]

    saw_negative_miss = False
    hit_count = 0
    miss_count = 0
    for req in reqs:
        obs = env.get_cache_features()
        g = env.get_global_features()
        assert np.isfinite(obs).all()
        assert np.isfinite(g).all()
        assert ((obs >= 0.0) & (obs <= 1.0)).all()

        # choose invalid action once cache is full to verify penalty path
        action = 0
        r, hit = env.step(req, action)
        assert math.isfinite(r)
        if hit:
            hit_count += 1
        else:
            miss_count += 1
            if r < 0:
                saw_negative_miss = True

    assert saw_negative_miss, "adaptive miss reward never became negative"
    assert hit_count + miss_count == len(reqs)


def run_static_binary_compatibility():
    cfg = {
        "RECENCY_DENOM": 100.0,
        "FREQ_DENOM": 10.0,
        "HIT_EMA_ALPHA": 0.2,
        "MISS_STREAK_CLIP": 20,
        "FEATURE_SCALING_MODE": "static",
        "REWARD_MODE": "binary",
        "INVALID_ACTION_REWARD": -1.0,
    }
    env = CacheEnv(cache_size=1, use_global=True, invalid_penalty=True, config=cfg)

    r1, h1 = env.step(1, 0)  # miss + empty
    r2, h2 = env.step(1, 0)  # hit
    r3, h3 = env.step(2, 0)  # miss + full + invalid

    assert (h1, r1) == (False, 0.0)
    assert (h2, r2) == (True, 1.0)
    assert h3 is False and r3 == -1.0


if __name__ == "__main__":
    run_dynamic_adaptive_smoke()
    run_static_binary_compatibility()
    print("smoke tests passed")
