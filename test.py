import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from env_6g import Hybrid6GEnv
from model import HybridCNNDQN


def safe_torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def pick_checkpoint():
    for p in ["hybrid_cnn_dqn_6g_best.pth", "hybrid_cnn_dqn_6g_final.pth"]:
        if os.path.exists(p):
            return p
    return None


def greedy_action(model, state, device):
    img_tensor = torch.FloatTensor(state["image"]).unsqueeze(0).to(device)
    vec_tensor = torch.FloatTensor(state["vector"]).unsqueeze(0).to(device)
    q = model(img_tensor, vec_tensor)
    return torch.argmax(q, dim=1).item()


def run_test(
    episodes=200,
    total_steps=1000,
    B_hz=100e6,
    T_pp_steps=10,        # 1.0s / 0.1s
    N_succ=3,             # 300ms / 100ms
    gamma_th=-6.0,        # 성공 판정 임계(필요하면 논문 값으로)
    plot_episode_idx=0
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Hybrid6GEnv()

    ckpt = pick_checkpoint()
    if ckpt is None:
        print("❌ 체크포인트 없음: hybrid_cnn_dqn_6g_best.pth 또는 final.pth 필요")
        return

    print(f"[{device}] 체크포인트 로드: {ckpt}")
    model = HybridCNNDQN(num_actions=7, vector_dim=5).to(device)
    model.load_state_dict(safe_torch_load(ckpt, device))
    model.eval()

    # 누적 메트릭
    total_steps_run = 0
    rlf_steps = 0
    rlf_episodes = 0

    ho_total = 0
    pp_count = 0

    ho_succ = 0

    throughput_sum = 0.0

    saved_sinr = None

    with torch.no_grad():
        for ep in range(episodes):
            state, _ = env.reset()

            # Ping-pong 판정용 HO 이력: (step, from_node, to_node)
            ho_history = []

            # HSR 판정용: t_exec 이후 N_succ 스텝 검사
            # pending_success_check = { "remain": k, "ok": True }
            success_checks = []

            steps_this_ep = 0
            terminated_flag = False

            for t in range(total_steps):
                action = greedy_action(model, state, device)

                next_state, reward, terminated, truncated, info = env.step(
                    action)

                steps_this_ep += 1
                total_steps_run += 1

                is_rlf = bool(info.get("rlf", False))
                sinr_db = float(info.get("sinr", 0.0))
                in_ho_delay = bool(info.get("in_ho_delay", False))

                # RLF step 카운트
                if is_rlf:
                    rlf_steps += 1

                # Throughput: RLF 또는 HO 지연이면 0
                if (is_rlf or in_ho_delay):
                    Ct = 0.0
                else:
                    sinr_lin = 10 ** (sinr_db / 10.0)
                    Ct = B_hz * np.log2(1.0 + sinr_lin)

                throughput_sum += Ct

                # HO 트리거/실행 정보
                # env는 ho_triggered(트리거)와 pending_timer를 제공하고,
                # 실행 완료 시점은 pending_ho가 False로 돌아가며 hold가 시작됨
                # 여기서는 ho_triggered를 이벤트로 잡고,
                # t_exec는 "pending_timer가 0이 된 직후"를 잡기 위해,
                # current_node 변화 시점을 감지해서 처리(가장 명확)
                current_node = int(info.get("current_node", 1))

                # HO 감지(노드 변화)
                # 방법: action이 바뀌어도 바로 바뀌지 않기 때문에 current_node 변화가 실행 완료(t_exec)
                # 그래서 직전 node를 저장해서 변화 시점에 HSR 체크 시작
                if t == 0:
                    prev_node = current_node
                else:
                    # 실행 완료 감지
                    if current_node != prev_node:
                        # 총 HO 시도 수(실행 완료 기준으로 카운트)
                        ho_total += 1

                        # Ping-pong 판정: A->B 후 T_pp 내 B->A이면 1회
                        # 현재 변화는 prev_node -> current_node
                        src = prev_node
                        tgt = current_node
                        for p_step, p_src, p_tgt in reversed(ho_history):
                            if t - p_step <= T_pp_steps:
                                if p_src == tgt and p_tgt == src:
                                    pp_count += 1
                                    break
                            else:
                                break
                        ho_history.append((t, src, tgt))

                        # HSR 판정 체크 시작: t_exec 이후 N_succ 스텝 동안 sinr >= gamma_th
                        success_checks.append({"remain": N_succ, "ok": True})

                    prev_node = current_node

                # HSR 체크 업데이트
                if len(success_checks) > 0:
                    for chk in success_checks:
                        if chk["remain"] > 0:
                            if sinr_db < gamma_th:
                                chk["ok"] = False
                            chk["remain"] -= 1

                    # 완료된 체크는 확정 처리
                    still = []
                    for chk in success_checks:
                        if chk["remain"] == 0:
                            if chk["ok"]:
                                ho_succ += 1
                        else:
                            still.append(chk)
                    success_checks = still

                if ep == plot_episode_idx:
                    if saved_sinr is None:
                        saved_sinr = []
                    saved_sinr.append(sinr_db)

                state = next_state

                if terminated or truncated:
                    terminated_flag = bool(terminated)
                    break

            if terminated_flag:
                rlf_episodes += 1

    # 최종 산출
    step_rlf_rate = (rlf_steps / max(total_steps_run, 1)) * 100.0
    episode_rlf_rate = (rlf_episodes / episodes) * 100.0
    ping_pong_rate = (pp_count / ho_total * 100.0) if ho_total > 0 else 0.0
    hsr = (ho_succ / ho_total * 100.0) if ho_total > 0 else 0.0
    avg_tp_mbps = (throughput_sum / max(total_steps_run, 1)) / 1e6

    print("=" * 55)
    print("[논문 5.3 메트릭 결과]")
    print("=" * 55)
    print(f"- episodes                         : {episodes}")
    print(f"- total_steps_run                  : {total_steps_run}")
    print(f"- RLF rate (step)                  : {step_rlf_rate:.2f} %")
    print(f"- RLF rate (episode, terminated)   : {episode_rlf_rate:.2f} %")
    print(f"- HO total (exec-complete count)   : {ho_total}")
    print(f"- Ping-pong rate (T_pp=1s)         : {ping_pong_rate:.2f} %")
    print(
        f"- HSR (T_succ=300ms, gamma_th)     : {hsr:.2f} %  (gamma_th={gamma_th}dB)")
    print(f"- Avg Throughput (Shannon, Ct rule): {avg_tp_mbps:.2f} Mbps")
    print("=" * 55)

    if saved_sinr is not None and len(saved_sinr) > 0:
        plt.figure(figsize=(10, 4))
        plt.plot(saved_sinr, label="SINR (dB)", linewidth=1.5)
        plt.axhline(y=-6.0, linestyle="--", label="RLF Threshold (-6 dB)")
        plt.title(
            "SINR Trajectory (Hybrid CNN-DQN, paper-consistent env)", fontsize=12)
        plt.xlabel("Time step (100ms)")
        plt.ylabel("SINR (dB)")
        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig("sinr_result.png", dpi=300)
        print("✅ sinr_result.png 저장 완료")


if __name__ == "__main__":
    run_test(episodes=200, total_steps=1000, plot_episode_idx=0)
