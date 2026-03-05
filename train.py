import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

from env_6g import Hybrid6GEnv
from model import HybridCNNDQN

# --- [1] 경험 재생 버퍼 ---


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=64):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

# --- [2] DQN 에이전트 ---


class DQNAgent:
    def __init__(self, num_actions=6, vector_dim=5):
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
        if random.random() < epsilon:
            return random.randint(0, 5)

        self.main_net.eval()
        with torch.no_grad():
            img_tensor = torch.FloatTensor(
                state['image']).unsqueeze(0).to(self.device)
            vec_tensor = torch.FloatTensor(
                state['vector']).unsqueeze(0).to(self.device)
            q_values = self.main_net(img_tensor, vec_tensor)
            action = torch.argmax(q_values).item()
        self.main_net.train()
        return action

    def learn(self):
        if len(self.memory) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size)

        img_batch = torch.FloatTensor(
            np.array([s['image'] for s in states])).to(self.device)
        vec_batch = torch.FloatTensor(
            np.array([s['vector'] for s in states])).to(self.device)
        next_img_batch = torch.FloatTensor(
            np.array([s['image'] for s in next_states])).to(self.device)
        next_vec_batch = torch.FloatTensor(
            np.array([s['vector'] for s in next_states])).to(self.device)

        action_batch = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        done_batch = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        current_q_values = self.main_net(
            img_batch, vec_batch).gather(1, action_batch)
        with torch.no_grad():
            next_q_values = self.target_net(next_img_batch, next_vec_batch).max(1)[
                0].unsqueeze(1)
            target_q_values = reward_batch + \
                (self.gamma * next_q_values * (1 - done_batch))

        loss = nn.MSELoss()(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.main_net.state_dict())


# --- [3] 메인 학습 루프 (엔진 시작!) ---
if __name__ == "__main__":
    env = Hybrid6GEnv()
    agent = DQNAgent()

    num_episodes = 1000
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 600
    target_update_freq = 10

    print(f"\n🚀 [{agent.device}] 장치에서 6G 하이브리드 핸드오버 학습을 시작합니다...\n")

    for ep in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        epsilon = max(epsilon_end, epsilon_start - ep / epsilon_decay)

        while not done:
            action = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.memory.push(state, action, reward, next_state, done)
            loss = agent.learn()

            state = next_state
            total_reward += reward
            step_count += 1

            # 테스트를 위해 1에피소드당 최대 스텝을 1000으로 제한 (너무 오래 도는 것 방지)
            if step_count >= 1000:
                break

        if ep % target_update_freq == 0:
            agent.update_target_network()

        print(
            f"Episode: {ep:4d} | Steps: {step_count:4d} | Total Reward: {total_reward:8.2f} | Epsilon: {epsilon:.3f}")

    torch.save(agent.main_net.state_dict(), "hybrid_cnn_dqn_6g.pth")
    print("\n✅ 학습 완료 및 모델 가중치 저장 성공!")
