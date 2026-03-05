import numpy as np
import gymnasium as gym
from gymnasium import spaces


class Hybrid6GEnv(gym.Env):
    """
    6G 하이브리드 네트워크(NTN+GBS) 핸드오버 최적화를 위한 Custom RL 환경
    """
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self):
        super(Hybrid6GEnv, self).__init__()

        # --- [1] 물리적 환경 파라미터 ---
        self.map_size = 5000.0        # 맵 크기 (5km)
        self.grid_size = 100          # 맵 해상도 (100x100 픽셀)
        self.pixel_res = self.map_size / self.grid_size  # 1픽셀 = 50m
        self.dt = 0.1                 # 의사결정 주기 (100ms)

        # 통신 노드 수 및 위치 설정
        self.num_nodes = 6
        self.node_positions = np.zeros((self.num_nodes, 2))
        self.node_positions[1] = [2500, 2500]  # Macro Cell (중앙)
        self.node_positions[2] = [1250, 1250]  # Small Cell 1
        self.node_positions[3] = [3750, 1250]  # Small Cell 2
        self.node_positions[4] = [1250, 3750]  # Small Cell 3
        self.node_positions[5] = [3750, 3750]  # Small Cell 4

        # 송신 전력 (dBm) 및 통신 파라미터
        # 0:LEO, 1:Macro, 2~5:Small
        self.ptx = np.array([38.0, 46.0, 30.0, 30.0, 30.0, 30.0])
        self.noise_floor = -90.0  # dBm
        self.gamma_out = -6.0    # RLF 판정 임계값 (dB)
        self.c_hor = 10.0        # 수평 핸드오버 페널티
        self.c_ver = 50.0        # 수직 핸드오버 페널티 (지연 300ms 반영)
        self.p_fail = 100.0      # RLF 발생 페널티

        # 정적 장애물(건물) 맵 생성 (0.0 ~ 1.0 감쇠값)
        self.obstacle_map = np.zeros(
            (self.grid_size, self.grid_size), dtype=np.float32)
        self._generate_obstacles(num_obstacles=20)

        # --- [2] 행동 및 상태 공간 ---
        self.action_space = spaces.Discrete(self.num_nodes)
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=1, shape=(3, self.grid_size, self.grid_size), dtype=np.float32),
            "vector": spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        })

        # --- [3] 내부 변수 ---
        self.current_step = 0
        self.max_steps = 10000
        self.ue_pos = np.zeros(2)
        self.ue_velocity = 0.0
        self.ue_direction = 0.0
        self.leo_pos = np.zeros(2)
        self.current_node = 1
        self.rlf_counter = 0
        self.trajectory_history = []
        self.current_sinr_db = 10.0

    def _generate_obstacles(self, num_obstacles):
        """맵 상에 임의의 사각형 건물(장애물) 배치"""
        for _ in range(num_obstacles):
            x = np.random.randint(5, self.grid_size - 15)
            y = np.random.randint(5, self.grid_size - 15)
            w = np.random.randint(3, 10)
            h = np.random.randint(3, 10)
            attenuation = np.random.uniform(0.5, 1.0)  # 투과 손실 (0.5~1.0)
            self.obstacle_map[x:x+w, y:y+h] = attenuation

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.rlf_counter = 0

        # 단말 초기화
        self.ue_pos = self.np_random.uniform(0, self.map_size, size=(2,))
        self.ue_velocity = self.np_random.uniform(
            30, 120) * (1000 / 3600)  # m/s
        self.ue_direction = self.np_random.uniform(0, 2 * np.pi)
        self.current_node = 1

        # LEO 위성 초기화 (X축 방향으로 7.5km/s 비행 가정)
        self.leo_pos = np.array([0.0, 2500.0])

        # 궤적 초기화 (5스텝 윈도우)
        self.trajectory_history = [self.ue_pos.copy() for _ in range(5)]

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        self.current_step += 1

        # 1. 모빌리티 업데이트 (단말 및 위성 이동)
        self._update_mobility()

        # 2. 핸드오버 페널티 처리
        cost_ho = self._calculate_ho_cost(action)
        if action != self.current_node:
            self.current_node = action

        # 3. 채널 품질(SINR) 및 RLF 판정
        self.current_sinr_db, is_rlf = self._calculate_channel_quality()

        # 4. 보상 계산
        reward = self._calculate_reward(self.current_sinr_db, is_rlf, cost_ho)

        # 5. 종료 조건
        terminated = False
        truncated = self.current_step >= self.max_steps

        obs = self._get_obs()
        info = {"sinr": self.current_sinr_db, "rlf": is_rlf, "action": action}

        return obs, reward, terminated, truncated, info

    # ==========================================================
    # 내부 수학/통신 로직 구현부
    # ==========================================================

    def _update_mobility(self):
        """단말(UE)과 LEO 위성을 dt 동안 이동시킴"""
        # 단말 이동 (맵 경계에 닿으면 튕겨나오도록 반사)
        dx = self.ue_velocity * np.cos(self.ue_direction) * self.dt
        dy = self.ue_velocity * np.sin(self.ue_direction) * self.dt
        self.ue_pos += np.array([dx, dy])

        if self.ue_pos[0] < 0 or self.ue_pos[0] >= self.map_size:
            self.ue_direction = np.pi - self.ue_direction
            self.ue_pos[0] = np.clip(self.ue_pos[0], 0, self.map_size - 1)
        if self.ue_pos[1] < 0 or self.ue_pos[1] >= self.map_size:
            self.ue_direction = -self.ue_direction
            self.ue_pos[1] = np.clip(self.ue_pos[1], 0, self.map_size - 1)

        # LEO 위성 이동 (7.5 km/s)
        self.leo_pos[0] = (self.leo_pos[0] + 7500.0 * self.dt) % self.map_size
        self.node_positions[0] = self.leo_pos

        # 과거 궤적 업데이트
        self.trajectory_history.pop(0)
        self.trajectory_history.append(self.ue_pos.copy())

    def _calculate_ho_cost(self, action):
        """핸드오버 시그널링 오버헤드 비용 (논문 4.1절)"""
        if action == self.current_node:
            return 0.0
        elif action == 0:
            return self.c_ver  # 위성으로 핸드오버
        else:
            return self.c_hor  # 지상 기지국 간 핸드오버

    def _calculate_channel_quality(self):
        """경로손실 및 간섭을 고려한 SINR 계산"""
        px = int(np.clip(self.ue_pos[0] /
                 self.pixel_res, 0, self.grid_size - 1))
        py = int(np.clip(self.ue_pos[1] /
                 self.pixel_res, 0, self.grid_size - 1))

        # 장애물 감쇠 (0.0~1.0을 최대 20dB 감쇠로 스케일링)
        blockage_attenuation = self.obstacle_map[px, py] * 20.0

        rx_power = np.zeros(self.num_nodes)

        for i in range(self.num_nodes):
            dist = np.linalg.norm(self.ue_pos - self.node_positions[i])
            dist = max(dist, 1.0)

            if i == 0:
                # LEO: FSPL + 고도 600km
                slant_dist = np.sqrt(dist**2 + 600000**2)
                pl = 32.4 + 20 * np.log10(2.0) + 20 * np.log10(slant_dist)
                rx_power[i] = self.ptx[i] - pl
            else:
                # GBS: UMa 모델 기반 + 장애물
                pl = 32.4 + 20 * np.log10(2.0) + 31.9 * np.log10(dist)
                rx_power[i] = self.ptx[i] - pl - blockage_attenuation

        # 선형 스케일(mW) 변환 후 SINR 계산
        rx_linear = 10 ** (rx_power / 10.0)
        noise_linear = 10 ** (self.noise_floor / 10.0)

        target_signal = rx_linear[self.current_node]
        interference = np.sum(rx_linear) - target_signal

        sinr_linear = target_signal / (interference + noise_linear)
        current_sinr_db = 10 * np.log10(sinr_linear)

        # RLF 판정 (연속 2스텝 임계값 미만)
        is_rlf = False
        if current_sinr_db < self.gamma_out:
            self.rlf_counter += 1
            if self.rlf_counter >= 2:
                is_rlf = True
        else:
            self.rlf_counter = 0

        return current_sinr_db, is_rlf

    def _calculate_reward(self, sinr_db, is_rlf, cost_ho):
        """최종 보상 함수 도출"""
        w1, w2, w3 = 1.0, 1.0, 1.0
        if is_rlf:
            return - (w3 * self.p_fail)
        else:
            sinr_linear = 10 ** (sinr_db / 10.0)
            r_cap = np.log2(1 + sinr_linear)
            return (w1 * r_cap) - (w2 * cost_ho)

    def _get_obs(self):
        """CNN 모듈에 입력될 3채널 2D 이미지 및 수치 벡터 생성 (논문 3.2절)"""
        image_tensor = np.zeros(
            (3, self.grid_size, self.grid_size), dtype=np.float32)

        # 채널 1: 장애물 및 인프라 맵
        image_tensor[0] = self.obstacle_map.copy()
        for i in range(1, self.num_nodes):  # 기지국 픽셀은 1.0으로 표시
            px = int(
                np.clip(self.node_positions[i][0] / self.pixel_res, 0, self.grid_size - 1))
            py = int(
                np.clip(self.node_positions[i][1] / self.pixel_res, 0, self.grid_size - 1))
            image_tensor[0, px, py] = 1.0

        # 채널 2: LEO 풋프린트 (위성 위치 기반 가시성 맵)
        l_px = int(np.clip(self.leo_pos[0] /
                   self.pixel_res, 0, self.grid_size - 1))
        l_py = int(np.clip(self.leo_pos[1] /
                   self.pixel_res, 0, self.grid_size - 1))
        # 간단한 원형 풋프린트: 반경 20픽셀(1km) 내에만 1~0 그라데이션 적용
        y_idx, x_idx = np.ogrid[:self.grid_size, :self.grid_size]
        dist_from_leo = np.sqrt((x_idx - l_px)**2 + (y_idx - l_py)**2)
        footprint = np.clip(1.0 - (dist_from_leo / 20.0), 0, 1)
        image_tensor[1] = footprint

        # 채널 3: 단말 궤적 (지수적 감쇠)
        decay_factor = 1.0
        for pos in reversed(self.trajectory_history):
            px = int(np.clip(pos[0] / self.pixel_res, 0, self.grid_size - 1))
            py = int(np.clip(pos[1] / self.pixel_res, 0, self.grid_size - 1))
            image_tensor[2, px, py] = max(
                image_tensor[2, px, py], decay_factor)
            decay_factor *= 0.8  # 과거 궤적일수록 흐릿하게

        # 1D 벡터 (딥러닝 학습을 돕기 위해 값 정규화 적용)
        vector_tensor = np.array([
            self.current_node / 5.0,                  # 0~1 스케일링
            self.ue_velocity / 33.3,                  # 최고 120km/h(33.3m/s) 기준
            np.clip((self.current_sinr_db + 20) / 50.0, 0, 1),  # SINR 정규화
            self.rlf_counter / 2.0,                   # 위험도 지표
            1.0 if self.current_node == 0 else 0.0    # 위성 접속 여부
        ], dtype=np.float32)

        return {"image": image_tensor, "vector": vector_tensor}
