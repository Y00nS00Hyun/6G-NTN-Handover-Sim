# validate_metrics.py
import numpy as np
import torch

from env_6g import Hybrid6GEnv
from model import HybridCNNDQN


def make_policy(mode, model=None, device="cpu"):
    """
    mode:
      - "stay": 항상 a=0
      - "random": 랜덤 (0~6)
      - "model": 학습된 모델 argmax
    """
    if mode == "stay":
        return lambda obs: 0

    if mode == "random":
        return lambda obs: np.random.randint(0, 7)

    if mode == "model":
        assert model is not None
        model.eval()

        def _policy(obs):
            # obs: {"image": (3,100,100), "vector": (5,)}
            img = torch.tensor(
                obs["image"], dtype=torch.float32, device=device).unsqueeze(0)
            vec = torch.tensor(
                obs["vector"], dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q = model(img, vec)
                a = int(torch.argmax(q, dim=1).item())
            return a

        return _policy

    raise ValueError(f"Unknown mode: {mode}")


def evaluate_53(env, policy_fn, episodes=200, seed=0):
    """
    논문 5.3 기반 메트릭을 env info/파라미터로 "재현 가능한 방식"으로 집계.
    - RLF rate (step): 전체 step 중 rlf True 비율
    - RLF rate (episode): terminated(=RLF)로 끝난 에피소드 비율
    - HO total: ho_exec_complete 카운트
    - Ping-pong: A->B 후 T_pp 안에 B->A가 발생한 횟수 / HO total
    - HSR: exec-complete 된 HO 중, T_succ 동안 gamma_th 이상 유지 성공 비율
           (결정이 나오기 전에 에피소드가 종료되면 "실패"로 처리)
    - Avg Throughput: Ct rule (RLF or HO pending이면 0) 평균
    """
    rng = np.random.default_rng(seed)

    # 논문 타이머(초) -> step 수
    dt = env.dt
    T_pp = 1.0
    N_pp = int(np.ceil(T_pp / dt))

    total_steps = 0
    rlf_steps = 0
    terminated_eps = 0

    ho_exec_total = 0
    ho_success_total = 0
    ho_success_pending = 0  # exec 완료 후 아직 성공판정 안 난 HO 수

    pingpong_count = 0
    last_ho = None  # (src, tgt, exec_step)

    sum_ct = 0.0

    # 디버그용(원인 추적)
    sinr_all = []
    steps_to_end = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=int(rng.integers(1, 1_000_000)))
        done = False
        trunc = False

        while not done and not trunc:
            a = policy_fn(obs)
            obs, reward, done, trunc, info = env.step(a)

            total_steps += 1
            sinr_all.append(info["sinr_db"])
            sum_ct += float(info["Ct"])

            if info["rlf"]:
                rlf_steps += 1

            # HO exec-complete 카운트
            if info["ho_exec_complete"]:
                ho_exec_total += 1
                ho_success_pending += 1  # 성공 판정이 나올 때까지 pending 처리

                src = info["ho_exec"]["src"]
                tgt = info["ho_exec"]["tgt"]
                exec_step = info["step"]

                # Ping-pong: 직전 HO가 A->B였고, 이번이 B->A이며, 시간차 <= T_pp이면 1회
                if last_ho is not None:
                    last_src, last_tgt, last_step = last_ho
                    if (last_src == tgt) and (last_tgt == src) and ((exec_step - last_step) <= N_pp):
                        pingpong_count += 1

                last_ho = (src, tgt, exec_step)

            # HSR 판정: env가 N_succ 끝나는 시점에 ho_success_decided=True로 알려줌
            if info["ho_success_decided"]:
                # 결정이 나온 HO 1건을 pending에서 정리
                if ho_success_pending > 0:
                    ho_success_pending -= 1
                # 성공 집계
                if bool(info["ho_success"]):
                    ho_success_total += 1

        steps_to_end.append(env.current_step)

        # 에피소드 종료 원인
        if done and (not trunc):
            terminated_eps += 1

        # (중요) 에피소드가 끝났는데 ho_success_pending이 남아있으면
        # 아직 T_succ 판단 전 종료된 HO들이므로 논문 정의상 "성공 못함" = 실패로 처리
        # => 성공 카운트는 안 올리고, 분모(HO total)는 이미 올린 상태 그대로 둠
        ho_success_pending = 0

    # 최종 메트릭
    rlf_rate_step = (rlf_steps / max(total_steps, 1)) * 100.0
    rlf_rate_ep = (terminated_eps / max(episodes, 1)) * 100.0

    pingpong_rate = (pingpong_count / max(ho_exec_total, 1)) * 100.0
    hsr = (ho_success_total / max(ho_exec_total, 1)) * 100.0

    avg_tp_mbps = (sum_ct / max(total_steps, 1)) / 1e6

    # 진단 출력용 SINR 통계
    sinr_all = np.array(sinr_all, dtype=np.float32)
    sinr_stats = {
        "sinr_mean_db": float(np.mean(sinr_all)) if sinr_all.size else None,
        "sinr_p10_db": float(np.percentile(sinr_all, 10)) if sinr_all.size else None,
        "sinr_min_db": float(np.min(sinr_all)) if sinr_all.size else None,
        "avg_steps_per_ep": float(np.mean(steps_to_end)) if steps_to_end else None,
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
    # 반드시 num_actions=7로 맞춰야 함 (환경 action 0..6)
    model = HybridCNNDQN(num_actions=7, vector_dim=5).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt)
    return model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = Hybrid6GEnv(seed=0)

    # 1) sanity check: stay 정책
    res_stay = evaluate_53(env, make_policy("stay"), episodes=50, seed=1)
    print("\n[Sanity] STAY policy\n", res_stay)

    # 2) sanity check: random 정책
    env = Hybrid6GEnv(seed=0)
    res_rand = evaluate_53(env, make_policy("random"), episodes=50, seed=2)
    print("\n[Sanity] RANDOM policy\n", res_rand)

    # 3) 모델 정책 (원하면 주석 해제)
    ckpt = r"C:\Users\soohyun\Desktop\6G-NTN-Handover-Sim\hybrid_cnn_dqn_6g_best.pth"
    model = load_model(ckpt, device=device)
    env = Hybrid6GEnv(seed=0)
    res_model = evaluate_53(env, make_policy(
        "model", model=model, device=device), episodes=200, seed=3)
    print("\n[MODEL] Hybrid CNN-DQN\n", res_model)


if __name__ == "__main__":
    main()
