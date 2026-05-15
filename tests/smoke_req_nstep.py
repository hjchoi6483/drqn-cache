import numpy as np
import torch
import torch.optim as optim

from src.models.drqn import AR, CacheEnv, EpisodeReplay, Obs, build_models, make_obs_for_eval, select_action, train_step


class S:
    algo = "drqn_perslot"
    use_global = True
    invalid_penalty = False


def main():
    config = {
        "RECENCY_DENOM": 100.0,
        "FREQ_DENOM": 50.0,
        "REQ_FREQ_DENOM": 20.0,
        "HIT_EMA_ALPHA": 0.01,
        "MISS_STREAK_CLIP": 50,
        "RECENT_WINDOW_SIZE": 16,
        "START_TRAIN_AFTER_EPISODES": 1,
        "BATCH_SIZE": 2,
        "BURN_IN": 2,
        "UNROLL": 4,
        "GAMMA": 0.99,
        "GRAD_CLIP": 1.0,
        "N_STEP": 1,
    }
    env = CacheEnv(cache_size=4, use_global=True, invalid_penalty=False, config=config)
    req = 42
    rf = env.get_request_features(req)
    assert rf.shape == (5,)
    assert np.isfinite(rf).all()
    assert ((rf >= 0.0) & (rf <= 1.0)).all()

    device = torch.device("cpu")
    s = S()
    online, target = build_models(4, s, device)
    obs = make_obs_for_eval(env, req)
    assert isinstance(obs, Obs)
    assert obs.req_feat.shape == (5,)
    h = online.init_hidden(1)
    a, _ = select_action(online, obs, h, eps=0.0, device=device)
    assert isinstance(a, int)

    replay = EpisodeReplay(max_episodes=10)
    for _ in range(3):
        env = CacheEnv(cache_size=4, use_global=True, invalid_penalty=False, config=config)
        obs_list, ar_list = [], []
        reqs = [1, 2, 3, 4, 1, 2, 5, 6, 1, 2, 3, 4]
        hid = online.init_hidden(1)
        for i, r in enumerate(reqs):
            o = make_obs_for_eval(env, r)
            obs_list.append(o)
            act, hid = select_action(online, o, hid, eps=0.5, device=device)
            rew, _ = env.step(r, act)
            ar_list.append(AR(action=act, reward=float(rew), done=(i == len(reqs) - 1)))
        obs_list.append(make_obs_for_eval(env, reqs[-1]))
        replay.add_episode(obs_list, ar_list)

    opt = optim.Adam(online.parameters(), lr=1e-3)
    loss1 = train_step(online, target, opt, replay, 4, config, device)
    assert np.isfinite(loss1)

    config["N_STEP"] = 3
    loss3 = train_step(online, target, opt, replay, 4, config, device)
    assert np.isfinite(loss3)

    print("smoke_req_nstep_ok", loss1, loss3)


if __name__ == "__main__":
    main()
