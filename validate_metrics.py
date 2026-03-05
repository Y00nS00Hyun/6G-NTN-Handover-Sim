# validate_metrics.py
import numpy as np
import torch

from env_6g import Hybrid6GEnv
from model import HybridCNNDQN


def make_policy(mode, model=None, device="cpu"):
    """
    mode:
      - "stay": always a=0
      - "random": random action (0..6)
      - "model": argmax Q from trained model
    """
    if mode == "stay":
        return lambda obs: 0

    if mode == "random":
        return lambda obs: int(np.random.randint(0, 7))

    if mode == "model":
        assert model is not None
        model.eval()

        def _policy(obs):
            # obs: {"image": (3,100,100), "vector": (5,)}
            img = torch.from_numpy(obs["image"]).to(
                device=device, dtype=torch.float32
            ).unsqueeze(0)
            vec = torch.from_numpy(obs["vector"]).to(
                device=device, dtype=torch.float32
            ).unsqueeze(0)

            with torch.no_grad():
                q = model(img, vec)
                a = int(torch.argmax(q, dim=1).item())
            return a

        return _policy

    raise ValueError(f"Unknown mode: {mode}")


def _safe_get(obj, keys, default=None):
    """Try multiple keys/attrs from dict-like or object-like."""
    # dict first
    if isinstance(obj, dict):
        for k in keys:
            if k in obj and obj[k] is not None:
                return obj[k]
        return default

    # object attrs
    for k in keys:
        if hasattr(obj, k):
            v = getattr(obj, k)
            if v is not None:
                return v
    return default


def _fmt(v):
    """Pretty print for debug."""
    if v is None:
        return "None"
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.3f}"
    return str(v)


def evaluate_53(
    env,
    policy_fn,
    episodes=200,
    seed=0,
    max_steps_per_ep=None,
    progress_every=5000,
    debug_first_steps=0,
):
    """
    Paper-like (Section 5.3) metric evaluator.

    Metrics:
      - RLF rate (step): ratio of steps with rlf==True
      - RLF rate (episode, terminated): ratio of episodes ended by done=True
      - HO total: count of ho_exec_complete
      - Ping-pong rate: count(A->B then within T_pp B->A) / HO total
      - HSR: successful handovers / executed handovers
      - Avg Throughput: average Ct (Shannon with Ct-rule) in Mbps
    """
    rng = np.random.default_rng(seed)

    # timers -> steps
    dt = float(getattr(env, "dt", 0.1))
    T_pp = 1.0  # seconds
    N_pp = int(np.ceil(T_pp / dt))

    total_steps = 0
    rlf_steps = 0
    terminated_eps = 0

    ho_exec_total = 0
    ho_success_total = 0
    ho_success_pending = 0

    pingpong_count = 0
    last_ho = None  # (src, tgt, exec_step)

    sum_ct = 0.0

    # diagnostics
    sinr_all = []
    steps_to_end = []
    action_hist = np.zeros(7, dtype=np.int64)

    for ep in range(episodes):
        obs, reset_info = env.reset(seed=int(rng.integers(1, 1_000_000)))
        done = False
        trunc = False
        local_steps = 0

        while (not done) and (not trunc):
            a = int(policy_fn(obs))
            if 0 <= a < 7:
                action_hist[a] += 1

            # ---- DEBUG PRE ----
            if debug_first_steps and total_steps < debug_first_steps:
                serving_pre = _safe_get(
                    env, ["serving_bs", "serving_node", "serving_cell"])
                ue_pre = _safe_get(
                    env, ["ue_pos", "ue_xy", "ue_loc", "ue_idx", "ue_id"])
                sinr_pre = _safe_get(env, ["sinr_db", "sinr", "sinr_mean_db"])
                step_pre = _safe_get(
                    env, ["step", "t", "time_step"], default=total_steps)

                print(
                    f"[PRE ] ep={ep} t={local_steps} gstep={total_steps} "
                    f"a={a} serving={_fmt(serving_pre)} ue={_fmt(ue_pre)} sinr_db={_fmt(sinr_pre)}"
                )

            obs, reward, done, trunc, info = env.step(a)

            total_steps += 1
            local_steps += 1

            # ---- collect ----
            sinr_val = info.get("sinr_db", None)
            if sinr_val is None:
                sinr_val = info.get("sinr", None)
            if sinr_val is not None:
                sinr_all.append(float(sinr_val))

            ct_val = info.get("Ct", None)
            if ct_val is not None:
                sum_ct += float(ct_val)

            if bool(info.get("rlf", False)):
                rlf_steps += 1

            # executed HO count
            if bool(info.get("ho_exec_complete", False)):
                ho_exec_total += 1
                ho_success_pending += 1

                ho_exec = info.get("ho_exec", {}) or {}
                src = ho_exec.get("src", None)
                tgt = ho_exec.get("tgt", None)
                exec_step = info.get("step", total_steps)

                # ping-pong: A->B then within T_pp, B->A
                if last_ho is not None and (src is not None) and (tgt is not None):
                    last_src, last_tgt, last_step = last_ho
                    if (last_src == tgt) and (last_tgt == src) and ((exec_step - last_step) <= N_pp):
                        pingpong_count += 1
                if (src is not None) and (tgt is not None):
                    last_ho = (src, tgt, exec_step)

            # success decision
            if bool(info.get("ho_success_decided", False)):
                if ho_success_pending > 0:
                    ho_success_pending -= 1
                if bool(info.get("ho_success", False)):
                    ho_success_total += 1

            # ---- DEBUG POST ----
            if debug_first_steps and total_steps <= debug_first_steps:
                serving_post = (
                    info.get("serving_node", None)
                    if isinstance(info, dict)
                    else None
                )
                if serving_post is None:
                    serving_post = _safe_get(
                        env, ["serving_bs", "serving_node", "serving_cell"])

                ue_post = info.get("ue_pos", None) if isinstance(
                    info, dict) else None
                if ue_post is None:
                    ue_post = _safe_get(
                        env, ["ue_pos", "ue_xy", "ue_loc", "ue_idx", "ue_id"])

                sinr_post = info.get("sinr_db", None) if isinstance(
                    info, dict) else None
                if sinr_post is None:
                    sinr_post = _safe_get(
                        env, ["sinr_db", "sinr", "sinr_mean_db"])

                print(
                    f"[POST] ep={ep} t={local_steps} gstep={total_steps} "
                    f"a={a} serving={_fmt(serving_post)} ue={_fmt(ue_post)} "
                    f"sinr_db={_fmt(sinr_post)} rlf={info.get('rlf', None)} Ct={_fmt(info.get('Ct', None))}"
                )

            # progress prints
            if progress_every and (total_steps % progress_every == 0):
                print(f"running... total_steps={total_steps}")

            # optional per-episode step cap
            if max_steps_per_ep is not None and local_steps >= int(max_steps_per_ep):
                trunc = True

        steps_to_end.append(local_steps)

        # episode ended by "done"
        if done and (not trunc):
            terminated_eps += 1

        # pending HOs that did not reach decision window are treated as failures
        ho_success_pending = 0

    # final metrics
    rlf_rate_step = (rlf_steps / max(total_steps, 1)) * 100.0
    rlf_rate_ep = (terminated_eps / max(episodes, 1)) * 100.0

    pingpong_rate = (pingpong_count / max(ho_exec_total, 1)) * 100.0
    hsr = (ho_success_total / max(ho_exec_total, 1)) * 100.0

    avg_tp_mbps = (sum_ct / max(total_steps, 1)) / 1e6

    sinr_all = np.array(sinr_all, dtype=np.float32)
    sinr_stats = {
        "sinr_mean_db": float(np.mean(sinr_all)) if sinr_all.size else None,
        "sinr_p10_db": float(np.percentile(sinr_all, 10)) if sinr_all.size else None,
        "sinr_min_db": float(np.min(sinr_all)) if sinr_all.size else None,
        "avg_steps_per_ep": float(np.mean(steps_to_end)) if steps_to_end else None,
        "action_hist": action_hist.tolist(),
        "sinr_samples": int(sinr_all.size),
    }

    return {
        "episodes": episodes,
        "total_steps_run": total_steps,
        "RLF_rate_step_%": rlf_rate_step,
        "RLF_rate_episode_terminated_%": rlf_rate_ep,
        "HO_total_exec_complete": ho_exec_total,
        "Pingpong_rate_%": pingpong_rate,
        "HSR_%": hsr,
        "Avg_Throughput_Mbps": avg_tp_mbps,
        "diag": sinr_stats,
    }


def load_model(ckpt_path, device="cpu"):
    model = HybridCNNDQN(num_actions=7, vector_dim=5).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt)
    return model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device =", device)

    # ---- user knobs ----
    MAX_STEPS_PER_EP = 2000       # cap each episode
    SANITY_EPISODES = 5           # STAY/RANDOM
    MODEL_EPISODES = 10           # quick model run
    PROGRESS_EVERY = 0            # 0 disables
    DEBUG_FIRST_STEPS = 50        # print first global steps
    # --------------------

    # 1) STAY sanity
    env = Hybrid6GEnv(seed=0)
    res_stay = evaluate_53(
        env,
        make_policy("stay"),
        episodes=SANITY_EPISODES,
        seed=1,
        max_steps_per_ep=MAX_STEPS_PER_EP,
        progress_every=PROGRESS_EVERY,
        debug_first_steps=DEBUG_FIRST_STEPS,
    )
    print("\n[Sanity] STAY policy\n", res_stay)

    # 2) RANDOM sanity
    env = Hybrid6GEnv(seed=0)
    res_rand = evaluate_53(
        env,
        make_policy("random"),
        episodes=SANITY_EPISODES,
        seed=2,
        max_steps_per_ep=MAX_STEPS_PER_EP,
        progress_every=PROGRESS_EVERY,
        debug_first_steps=DEBUG_FIRST_STEPS,
    )
    print("\n[Sanity] RANDOM policy\n", res_rand)

    # 3) MODEL
    ckpt = r"C:\Users\soohyun\Desktop\6G-NTN-Handover-Sim\hybrid_cnn_dqn_6g_best.pth"
    model = load_model(ckpt, device=device)

    env = Hybrid6GEnv(seed=0)
    res_model = evaluate_53(
        env,
        make_policy("model", model=model, device=device),
        episodes=MODEL_EPISODES,
        seed=3,
        max_steps_per_ep=MAX_STEPS_PER_EP,
        progress_every=PROGRESS_EVERY,
        debug_first_steps=DEBUG_FIRST_STEPS,
    )
    print("\n[MODEL] Hybrid CNN-DQN\n", res_model)


if __name__ == "__main__":
    main()
