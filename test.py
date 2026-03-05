# test.py
import os
import argparse
import numpy as np
import torch

from env_6g import Hybrid6GEnv
from model import HybridCNNDQN


def select_greedy_action(model, state, device):
    img = torch.as_tensor(
        state["image"], dtype=torch.float32, device=device).unsqueeze(0)
    vec = torch.as_tensor(
        state["vector"], dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        q = model(img, vec)
        return int(torch.argmax(q, dim=1).item())


def run_eval_episode(env, model, device, max_steps=None):
    """
    Returns episode-level rollup and step-level traces.
    Metrics are computed in the caller to avoid double counting.
    """
    state, _ = env.reset()
    done = False

    steps = 0
    rlf_steps = 0
    ct_sum = 0.0

    ho_exec_cnt = 0
    ping_pong_cnt = 0

    hsr_attempt_cnt = 0     # count of HO attempts measured at execution-complete
    hsr_success_cnt = 0     # count of successful HOs (T_succ satisfied)

    terminated_by_rlf = False

    while not done:
        action = select_greedy_action(model, state, device)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        steps += 1

        # ---- RLF step rate ----
        if info.get("rlf", False):
            rlf_steps += 1

        # ---- Ct (already paper-rule: 0 during RLF or HO delay) ----
        ct_sum += float(info.get("Ct", 0.0))

        # ---- HO exec complete count (only count when execution completes) ----
        if info.get("ho_exec_complete", False):
            ho_exec_cnt += 1
            hsr_attempt_cnt += 1

            # ping-pong event (A->B->A within T_pp) is flagged at exec completion
            if info.get("ping_pong_event", False):
                ping_pong_cnt += 1

        # ---- HSR event (True/False/None) ----
        # This is emitted once when success-check finishes.
        # Count only when it's explicitly True.
        hse = info.get("ho_success", None)
        if hse is True:
            hsr_success_cnt += 1
        # If hse is False, it was an attempt that failed (attempt already counted at ho_executed).

        if terminated:
            terminated_by_rlf = True

        state = next_state

        if max_steps is not None and steps >= max_steps:
            break

    return {
        "steps": steps,
        "rlf_steps": rlf_steps,
        "ct_sum": ct_sum,

        "terminated_by_rlf": terminated_by_rlf,

        "ho_exec_cnt": ho_exec_cnt,
        "ping_pong_cnt": ping_pong_cnt,

        "hsr_attempt_cnt": hsr_attempt_cnt,
        "hsr_success_cnt": hsr_success_cnt,
    }


def evaluate(model_path, episodes=200, seed=0, device=None, max_steps_per_ep=None):
    # ---- device ----
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # ---- env ----
    # Use a fixed seed for reproducibility (episode seeds will vary deterministically)
    env = Hybrid6GEnv(seed=seed)

    # ---- model ----
    model = HybridCNNDQN(num_actions=7, vector_dim=5).to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    print(f"[{device}] 체크포인트 로드: {model_path}")

    # ---- aggregates ----
    total_steps = 0
    total_rlf_steps = 0
    total_ct_sum = 0.0

    total_terminated_eps = 0

    total_ho_exec = 0
    total_ping_pong = 0

    total_hsr_attempt = 0
    total_hsr_success = 0

    for ep in range(episodes):
        # make each episode deterministic but different
        env.reset(seed=seed + ep)

        out = run_eval_episode(env, model, device, max_steps=max_steps_per_ep)

        total_steps += out["steps"]
        total_rlf_steps += out["rlf_steps"]
        total_ct_sum += out["ct_sum"]

        if out["terminated_by_rlf"]:
            total_terminated_eps += 1

        total_ho_exec += out["ho_exec_cnt"]
        total_ping_pong += out["ping_pong_cnt"]

        total_hsr_attempt += out["hsr_attempt_cnt"]
        total_hsr_success += out["hsr_success_cnt"]

    # ---- metrics (paper 5.3) ----
    rlf_rate_step = (total_rlf_steps / max(total_steps, 1)) * 100.0
    rlf_rate_episode = (total_terminated_eps / max(episodes, 1)) * 100.0

    ping_pong_rate = (total_ping_pong / total_ho_exec *
                      100.0) if total_ho_exec > 0 else 0.0
    hsr = (total_hsr_success / total_hsr_attempt *
           100.0) if total_hsr_attempt > 0 else 0.0

    avg_throughput_mbps = (total_ct_sum / max(total_steps, 1)) / 1e6

    # ---- print ----
    print("\n" + "=" * 55)
    print("[논문 5.3 메트릭 결과]")
    print("=" * 55)
    print(f"- episodes                         : {episodes}")
    print(f"- total_steps_run                  : {total_steps}")
    print(f"- RLF rate (step)                  : {rlf_rate_step:.2f} %")
    print(f"- RLF rate (episode, terminated)   : {rlf_rate_episode:.2f} %")
    print(f"- HO total (exec-complete count)   : {total_ho_exec}")
    print(f"- Ping-pong rate (T_pp=1s)         : {ping_pong_rate:.2f} %")
    print(f"- HSR (T_succ=300ms, gamma_th)     : {hsr:.2f} %")
    print(
        f"- Avg Throughput (Shannon, Ct rule): {avg_throughput_mbps:.2f} Mbps")
    print("=" * 55)

    return {
        "episodes": episodes,
        "total_steps": total_steps,
        "rlf_rate_step": rlf_rate_step,
        "rlf_rate_episode": rlf_rate_episode,
        "ho_exec_total": total_ho_exec,
        "ping_pong_rate": ping_pong_rate,
        "hsr": hsr,
        "avg_throughput_mbps": avg_throughput_mbps,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str,
                   default="hybrid_cnn_dqn_6g_best.pth")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None,
                   help="cuda / cpu (default: auto)")
    p.add_argument("--max_steps_per_ep", type=int, default=None,
                   help="optional cap per episode")
    args = p.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {args.model_path}")

    evaluate(
        model_path=args.model_path,
        episodes=args.episodes,
        seed=args.seed,
        device=args.device,
        max_steps_per_ep=args.max_steps_per_ep
    )


if __name__ == "__main__":
    main()
