import numpy as np
import gymnasium as gym
from gymnasium import spaces


class Hybrid6GEnv(gym.Env):
    """
    6G 하이브리드 네트워크(NTN+GBS) 핸드오버 최적화를 위한 Custom RL 환경
    논문 정합 포인트:
    - Action: 0=유지, 1~5=GBS, 6=LEO
    - HO 실행 지연: 수평 50ms(1 step), 수직 300ms(3 steps)
    - HO 지연 구간 및 RLF 구간 처리량 0(C_t=0) 반영 (reward에서 구현)
    - RLF: SINR < gamma_out 연속 2스텝이면 발생(T_fail=200ms)
    - Ping-pong 억제를 위한 최소 유지시간(hold) 1초(10 steps) 제공(환경 레벨)
    """
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self):
        super().__init__()

        # --- [1] 물리적 환경 파라미터 ---
        self.map_size = 5000.0
        self.grid_size = 100
        self.pixel_res = self.map_size / self.grid_size
        self.dt = 0.1  # 100ms

        # 물리 노드(링크 계산용): 0=LEO, 1=Macro, 2~5=Small
        self.num_nodes = 6

        # 에이전트 행동: 0=유지, 1~5=GBS, 6=LEO
        self.num_actions = 7

        self.node_positions = np.zeros((self.num_nodes, 2))
        self.node_positions[1] = [2500, 2500]  # Macro
        self.node_positions[2] = [1250, 1250]
        self.node_positions[3] = [3750, 1250]
        self.node_positions[4] = [1250, 3750]
        self.node_positions[5] = [3750, 3750]

        self.ptx = np.array([38.0, 46.0, 30.0, 30.0, 30.0, 30.0])
        self.noise_floor = -90.0

        # RLF 임계
        self.gamma_out = -6.0
        self.T_fail = 0.2
        self.N_fail = int(np.ceil(self.T_fail / self.dt))  # 2 steps

        # HO 비용
        self.c_hor = 10.0
        self.c_ver = 50.0
        self.p_fail = 100.0

        # HO 실행 지연 (step 단위)
        self.D_HO_hor_steps = int(np.ceil(0.05 / self.dt))  # 50ms -> 1
        self.D_HO_ver_steps = int(np.ceil(0.30 / self.dt))  # 300ms -> 3

        # Ping-pong 억제용 최소 유지시간(hold) 1초
        self.ho_hold_steps = int(np.ceil(1.0 / self.dt))  # 10 steps

        # 장애물 맵
        self.obstacle_map = np.zeros(
            (self.grid_size, self.grid_size), dtype=np.float32)
        self._generate_obstacles(num_obstacles=20)

        # --- [2] 행동 및 상태 공간 ---
        self.action_space = spaces.Discrete(self.num_actions)
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

        # 현재 접속 노드(물리 노드 0~5)
        self.current_node = 1

        # RLF용 카운터
        self.rlf_counter = 0

        # HO 실행 지연 상태
        self.pending_ho = False
        self.pending_target_node = None  # 물리 노드(0~5)
        self.pending_timer = 0           # 남은 지연 step
        self.in_ho_delay = False         # 해당 스텝이 HO 지연 구간인지

        # Ping-pong 억제용 hold 타이머
        self.ho_hold_timer = 0

        self.trajectory_history = []
        self.current_sinr_db = 10.0

        # 메트릭 계산에 도움되는 로그(선택)
        self.ho_triggered_this_step = False

    def _generate_obstacles(self, num_obstacles):
        for _ in range(num_obstacles):
            x = np.random.randint(5, self.grid_size - 15)
            y = np.random.randint(5, self.grid_size - 15)
            w = np.random.randint(3, 10)
            h = np.random.randint(3, 10)
            attenuation = np.random.uniform(0.5, 1.0)
            self.obstacle_map[x:x+w, y:y+h] = attenuation

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.rlf_counter = 0

        self.pending_ho = False
        self.pending_target_node = None
        self.pending_timer = 0
        self.in_ho_delay = False
        self.ho_hold_timer = 0
        self.ho_triggered_this_step = False

        self.ue_pos = self.np_random.uniform(0, self.map_size, size=(2,))
        self.ue_velocity = self.np_random.uniform(
            30, 120) * (1000 / 3600)  # m/s
        self.ue_direction = self.np_random.uniform(0, 2 * np.pi)

        self.current_node = 1  # Macro로 시작
        self.leo_pos = np.array([0.0, 2500.0])

        self.trajectory_history = [self.ue_pos.copy() for _ in range(5)]
        self.current_sinr_db = 10.0

        obs = self._get_obs()
        info = {}
        return obs, info

    # -------------------------
    # 논문 정합: action 해석
    # -------------------------
    def _action_to_target_node(self, action: int):
        # 0=유지
        if action == 0:
            return None
        # 1~5=GBS
        if 1 <= action <= 5:
            return action
        # 6=LEO
        if action == 6:
            return 0
        raise ValueError(f"Invalid action: {action}")

    def _calculate_ho_cost(self, target_node):
        if target_node is None or target_node == self.current_node:
            return 0.0
        if target_node == 0:
            return self.c_ver
        return self.c_hor

    def step(self, action):
        self.current_step += 1
        self.ho_triggered_this_step = False

        # 1) 모빌리티 업데이트
        self._update_mobility()

        # 2) 홀드 타이머 감소
        if self.ho_hold_timer > 0:
            self.ho_hold_timer -= 1

        # 3) 액션 해석
        requested_target = self._action_to_target_node(int(action))

        cost_ho = 0.0
        self.in_ho_delay = False

        # (A) 이미 HO 진행 중: 타이머 감소, 완료되면 전환
        if self.pending_ho:
            self.in_ho_delay = True
            self.pending_timer -= 1

            if self.pending_timer <= 0:
                # 실행 완료 시점(t_exec): 물리 노드 전환
                self.current_node = int(self.pending_target_node)
                self.pending_ho = False
                self.pending_target_node = None
                self.pending_timer = 0

                # 전환 직후 최소 유지시간(핑퐁 억제)
                self.ho_hold_timer = self.ho_hold_steps

        # (B) HO 진행 중 아니면: 트리거 가능
        else:
            # 홀드 중이면 HO 트리거 금지(유지로 강제)
            if self.ho_hold_timer > 0:
                requested_target = None

            if requested_target is not None and requested_target != self.current_node:
                # 트리거 수락
                cost_ho = self._calculate_ho_cost(requested_target)

                delay_steps = self.D_HO_ver_steps if requested_target == 0 else self.D_HO_hor_steps
                self.pending_ho = True
                self.pending_target_node = int(requested_target)
                self.pending_timer = int(delay_steps)
                self.in_ho_delay = True
                self.ho_triggered_this_step = True

        # 4) 채널 품질 및 RLF 판정(현재 붙어있는 링크 기준)
        self.current_sinr_db, is_rlf = self._calculate_channel_quality()

        # 5) 보상(논문: HO 지연 구간 및 RLF 구간 처리량 0 반영)
        reward = self._calculate_reward(
            self.current_sinr_db, is_rlf, cost_ho, self.in_ho_delay)

        terminated = bool(is_rlf)
        truncated = self.current_step >= self.max_steps

        obs = self._get_obs()
        info = {
            "sinr": float(self.current_sinr_db),
            "rlf": bool(is_rlf),
            "action": int(action),
            "current_node": int(self.current_node),
            "in_ho_delay": bool(self.in_ho_delay),
            "ho_triggered": bool(self.ho_triggered_this_step),
            "pending_ho": bool(self.pending_ho),
            "pending_timer": int(self.pending_timer),
            "pending_target_node": -1 if self.pending_target_node is None else int(self.pending_target_node),
        }
        return obs, reward, terminated, truncated, info

    # ==========================================================
    # 내부 수학/통신 로직
    # ==========================================================
    def _update_mobility(self):
        dx = self.ue_velocity * np.cos(self.ue_direction) * self.dt
        dy = self.ue_velocity * np.sin(self.ue_direction) * self.dt
        self.ue_pos += np.array([dx, dy])

        if self.ue_pos[0] < 0 or self.ue_pos[0] >= self.map_size:
            self.ue_direction = np.pi - self.ue_direction
            self.ue_pos[0] = np.clip(self.ue_pos[0], 0, self.map_size - 1)
        if self.ue_pos[1] < 0 or self.ue_pos[1] >= self.map_size:
            self.ue_direction = -self.ue_direction
            self.ue_pos[1] = np.clip(self.ue_pos[1], 0, self.map_size - 1)

        # LEO 이동(7.5 km/s)
        self.leo_pos[0] = (self.leo_pos[0] + 7500.0 * self.dt) % self.map_size
        self.node_positions[0] = self.leo_pos

        # 궤적 업데이트
        self.trajectory_history.pop(0)
        self.trajectory_history.append(self.ue_pos.copy())

    def _calculate_channel_quality(self):
        px = int(np.clip(self.ue_pos[0] /
                 self.pixel_res, 0, self.grid_size - 1))
        py = int(np.clip(self.ue_pos[1] /
                 self.pixel_res, 0, self.grid_size - 1))

        blockage_attenuation = self.obstacle_map[px, py] * 20.0

        rx_power = np.zeros(self.num_nodes)
        for i in range(self.num_nodes):
            dist = np.linalg.norm(self.ue_pos - self.node_positions[i])
            dist = max(dist, 1.0)

            if i == 0:
                slant_dist = np.sqrt(dist**2 + 600000**2)
                pl = 32.4 + 20 * np.log10(2.0) + 20 * np.log10(slant_dist)
                rx_power[i] = self.ptx[i] - pl
            else:
                pl = 32.4 + 20 * np.log10(2.0) + 31.9 * np.log10(dist)
                rx_power[i] = self.ptx[i] - pl - blockage_attenuation

        rx_linear = 10 ** (rx_power / 10.0)
        noise_linear = 10 ** (self.noise_floor / 10.0)

        target_signal = rx_linear[self.current_node]
        interference = np.sum(rx_linear) - target_signal

        sinr_linear = target_signal / (interference + noise_linear)
        current_sinr_db = 10 * np.log10(sinr_linear)

        # RLF 판정: SINR < gamma_out 연속 N_fail 스텝
        is_rlf = False
        if current_sinr_db < self.gamma_out:
            self.rlf_counter += 1
            if self.rlf_counter >= self.N_fail:
                is_rlf = True
        else:
            self.rlf_counter = 0

        return float(current_sinr_db), bool(is_rlf)

    def _calculate_reward(self, sinr_db, is_rlf, cost_ho, in_ho_delay=False):
        w1, w2, w3 = 1.0, 1.0, 1.0

        if is_rlf:
            return - (w3 * self.p_fail)

        # 논문: HO 지연 구간은 처리량 0
        if in_ho_delay:
            r_cap = 0.0
        else:
            sinr_linear = 10 ** (sinr_db / 10.0)
            r_cap = np.log2(1 + sinr_linear)

        return (w1 * r_cap) - (w2 * cost_ho)

    def _get_obs(self):
        image_tensor = np.zeros(
            (3, self.grid_size, self.grid_size), dtype=np.float32)

        # 채널 1: 장애물+인프라
        image_tensor[0] = self.obstacle_map.copy()
        for i in range(1, self.num_nodes):
            px = int(
                np.clip(self.node_positions[i][0] / self.pixel_res, 0, self.grid_size - 1))
            py = int(
                np.clip(self.node_positions[i][1] / self.pixel_res, 0, self.grid_size - 1))
            image_tensor[0, px, py] = 1.0

        # 채널 2: LEO footprint
        l_px = int(np.clip(self.leo_pos[0] /
                   self.pixel_res, 0, self.grid_size - 1))
        l_py = int(np.clip(self.leo_pos[1] /
                   self.pixel_res, 0, self.grid_size - 1))
        y_idx, x_idx = np.ogrid[:self.grid_size, :self.grid_size]
        dist_from_leo = np.sqrt((x_idx - l_px)**2 + (y_idx - l_py)**2)
        footprint = np.clip(1.0 - (dist_from_leo / 20.0), 0, 1)
        image_tensor[1] = footprint

        # 채널 3: UE trajectory
        decay_factor = 1.0
        for pos in reversed(self.trajectory_history):
            px = int(np.clip(pos[0] / self.pixel_res, 0, self.grid_size - 1))
            py = int(np.clip(pos[1] / self.pixel_res, 0, self.grid_size - 1))
            image_tensor[2, px, py] = max(
                image_tensor[2, px, py], decay_factor)
            decay_factor *= 0.8

        vector_tensor = np.array([
            self.current_node / 5.0,
            self.ue_velocity / 33.3,
            np.clip((self.current_sinr_db + 20) / 50.0, 0, 1),
            self.rlf_counter / float(self.N_fail),
            1.0 if self.current_node == 0 else 0.0
        ], dtype=np.float32)

        return {"image": image_tensor, "vector": vector_tensor}
