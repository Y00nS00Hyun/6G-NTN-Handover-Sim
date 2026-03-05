import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

from env_6g import Hybrid6GEnv
from model import HybridCNNDQN


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=64):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, num_actions=7, vector_dim=5):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.main_net = HybridCNNDQN(num_actions, vector_dim).to(self.device)
        self.target_net = HybridCNNDQN(num_actions, vector_dim).to(self.device)
        self.target_net.load_state_dict(self.main_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.main_net.parameters(), lr=1e-4)
        self.gamma = 0.95
        self.batch_size = 64
        self.memory = ReplayBuffer(capacity=10000)

    def select_action(self, state, epsilon):
        # 안전 우선 guard(원하면 유지, 아니면 제거 가능)
        # 현재 vector[2]=sinr_norm, vector[3]=rlf_risk
        sinr_norm = float(state["vector"][2])
        rlf_risk = float(state["vector"][3])
        if rlf_risk >= 0.5 or sinr_norm <= 0.25:
            return 1  # Macro(GBS1) 강제

        if random.random() < epsilon:
            return random.randint(0, 6)  # 0~6

        self.main_net.eval()
        with torch.no_grad():
            img_tensor = torch.FloatTensor(
                state["image"]).unsqueeze(0).to(self.device)
            vec_tensor = torch.FloatTensor(
                state["vector"]).unsqueeze(0).to(self.device)
            q_values = self.main_net(img_tensor, vec_tensor)
            action = torch.argmax(q_values, dim=1).item()
        self.main_net.train()
        return action

    def learn(self):
        if len(self.memory) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size)

        img_batch = torch.FloatTensor(
            np.array([s["image"] for s in states])).to(self.device)
        vec_batch = torch.FloatTensor(
            np.array([s["vector"] for s in states])).to(self.device)
        next_img_batch = torch.FloatTensor(
            np.array([s["image"] for s in next_states])).to(self.device)
        next_vec_batch = torch.FloatTensor(
            np.array([s["vector"] for s in next_states])).to(self.device)

        action_batch = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        done_batch = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        current_q = self.main_net(img_batch, vec_batch).gather(1, action_batch)

        with torch.no_grad():
            next_actions = self.main_net(
                next_img_batch, next_vec_batch).argmax(1, keepdim=True)
            next_q = self.target_net(
                next_img_batch, next_vec_batch).gather(1, next_actions)
            target_q = reward_batch + (self.gamma * next_q * (1 - done_batch))

        loss = nn.SmoothL1Loss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.main_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        return float(loss.item())

    def update_target_network(self):
        self.target_net.load_state_dict(self.main_net.state_dict())

# 👽


def evaluate(agent, env, episodes=10):
    rlf_episodes = 0
    total_steps = 0
    rlf_steps = 0

    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.select_action(state, epsilon=0.0)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_steps += 1
            if info.get("rlf", False):
                rlf_steps += 1

            state = next_state

        if terminated:
            rlf_episodes += 1

    return {
        "episode_rlf_rate": rlf_episodes / episodes,
        "step_rlf_rate": (rlf_steps / max(total_steps, 1)),
    }


if __name__ == "__main__":
    env = Hybrid6GEnv()
    eval_env = Hybrid6GEnv()
    agent = DQNAgent(num_actions=7)

    # --- 에러 검증: 100, 본학습: 10000 ---
    num_episodes = 10000  # <- 여기만 100으로 바꾸면 검사용

    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 7000
    target_update_freq = 10

    recent_rewards = deque(maxlen=50)
    best_avg_reward = -float("inf")

    warmup_min_transitions = 10000  # 안정 학습용(검사용이면 500~2000으로 낮춰도 됨)

    print(
        f"\n🚀 [{agent.device}] 학습 시작 | episodes={num_episodes} | actions=7 (0=stay,6=LEO)\n")

    for ep in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        step_count = 0

        epsilon = max(epsilon_end, epsilon_start - ep / epsilon_decay)

        while not done:
            action = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.memory.push(state, action, reward, next_state, done)

            if len(agent.memory) > warmup_min_transitions and step_count % 4 == 0:
                _ = agent.learn()

            state = next_state
            total_reward += float(reward)
            step_count += 1

            if step_count >= 1000:
                done = True

        if ep % target_update_freq == 0:
            agent.update_target_network()

        recent_rewards.append(total_reward)
        avg_reward = float(np.mean(recent_rewards))

        print(
            f"Episode: {ep:5d} | Steps: {step_count:4d} | Total Reward: {total_reward:8.2f} | Epsilon: {epsilon:.3f}")

        if ep >= 50 and avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            torch.save(agent.main_net.state_dict(),
                       "hybrid_cnn_dqn_6g_best.pth")
            print(
                f"   🌟 Best 갱신: avg_reward={best_avg_reward:.2f} -> hybrid_cnn_dqn_6g_best.pth 저장")

        if ep > 0 and ep % 100 == 0:
            torch.save(agent.main_net.state_dict(),
                       f"hybrid_cnn_dqn_6g_ep{ep}.pth")

        if ep > 0 and ep % 200 == 0:
            stats = evaluate(agent, eval_env, episodes=200)
            print(
                f"[EVAL] ep={ep} episode_rlf_rate={stats['episode_rlf_rate']:.4f} step_rlf_rate={stats['step_rlf_rate']:.4f}")

    torch.save(agent.main_net.state_dict(), "hybrid_cnn_dqn_6g_final.pth")
    print("\n✅ 학습 완료 및 최종 가중치 저장: hybrid_cnn_dqn_6g_final.pth")
