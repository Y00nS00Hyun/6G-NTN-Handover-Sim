import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# 앞서 작성한 파일에서 환경과 모델을 불러온다고 가정합니다.
# from env_6g import Hybrid6GEnv
# from model import HybridCNNDQN

# --- [1] 경험 재생 버퍼 (Experience Replay Buffer) ---


class ReplayBuffer:
    def __init__(self, capacity=10000):  # 논문 파라미터: 10,000 트랜지션
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """환경과 상호작용하여 얻은 튜플 e_t = (s_t, a_t, r_t, s_{t+1}) 저장"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=64):
        """버퍼에서 무작위로 미니배치를 추출하여 시간적 상관관계(Correlation) 제거"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

# --- [2] DQN 에이전트 클래스 ---


class DQNAgent:
    def __init__(self, num_actions=6, vector_dim=5):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # 주 네트워크 (Main Network)와 타겟 네트워크 (Target Network) 분리 (학습 안정화)
        self.main_net = HybridCNNDQN(num_actions, vector_dim).to(self.device)
        self.target_net = HybridCNNDQN(num_actions, vector_dim).to(self.device)
        self.target_net.load_state_dict(self.main_net.state_dict())
        self.target_net.eval()  # 타겟 네트워크는 가중치 업데이트를 하지 않음

        # 논문 파라미터 적용
        self.optimizer = optim.Adam(self.main_net.parameters(), lr=1e-4)
        self.gamma = 0.95  # 할인율 (Discount Factor)
        self.batch_size = 64
        self.memory = ReplayBuffer(capacity=10000)

    def select_action(self, state, epsilon):
        """엡실론-그리디(epsilon-greedy) 탐색 정책"""
        if random.random() < epsilon:
            return random.randint(0, 5)  # 탐색(Exploration): 무작위 행동 (0~5)

        # 활용(Exploitation): 현재 네트워크가 생각하는 최적의 행동 선택
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
        """미니배치를 뽑아 손실함수 L(theta)를 계산하고 역전파하여 가중치 업데이트"""
        if len(self.memory) < self.batch_size:
            return 0.0  # 버퍼에 데이터가 충분히 찰 때까지는 학습 대기

        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size)

        # 딕셔너리 형태의 State들을 PyTorch 텐서 미니배치로 변환
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

        # 1. 현재 상태에서의 Q(s, a; theta) 계산
        current_q_values = self.main_net(
            img_batch, vec_batch).gather(1, action_batch)

        # 2. 다음 상태에서의 최대 Q-value 예측: max Q(s', a'; theta^-)
        with torch.no_grad():
            next_q_values = self.target_net(next_img_batch, next_vec_batch).max(1)[
                0].unsqueeze(1)
            # TD Target 계산 (done이면 미래 보상이 없으므로 reward만)
            target_q_values = reward_batch + \
                (self.gamma * next_q_values * (1 - done_batch))

        # 3. 손실 함수 (MSE Loss) 계산 및 역전파
        loss = nn.MSELoss()(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """일정 주기마다 타겟 네트워크의 가중치를 주 네트워크와 동기화"""
        self.target_net.load_state_dict(self.main_net.state_dict())


# --- [3] 메인 학습 루프 (Training Loop) ---
if __name__ == "__main__":
    # env = Hybrid6GEnv() # 실제 환경 인스턴스화
    agent = DQNAgent()

    num_episodes = 5000  # 논문 6.1절 학습 에포크 수
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 3000  # 약 3000 에피소드 부근에서 탐색 종료

    target_update_freq = 10  # 10 에피소드마다 타겟 네트워크 업데이트

    # for ep in range(num_episodes):
    #     state, _ = env.reset()
    #     done = False
    #     total_reward = 0
    #
    #     # 선형 엡실론 감소 (1.0 -> 0.01)
    #     epsilon = max(epsilon_end, epsilon_start - ep / epsilon_decay)
    #
    #     while not done:
    #         action = agent.select_action(state, epsilon)
    #         next_state, reward, terminated, truncated, info = env.step(action)
    #         done = terminated or truncated
    #
    #         agent.memory.push(state, action, reward, next_state, done)
    #         loss = agent.learn()
    #
    #         state = next_state
    #         total_reward += reward
    #
    #     if ep % target_update_freq == 0:
    #         agent.update_target_network()
    #
    #     print(f"Episode: {ep}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

    # torch.save(agent.main_net.state_dict(), "hybrid_cnn_dqn_6g.pth")
    # print("학습 완료 및 모델 가중치 저장 성공!")
