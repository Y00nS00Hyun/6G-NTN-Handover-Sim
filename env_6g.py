import numpy as np
import gymnasium as gym
from gymnasium import spaces


class Hybrid6GEnv(gym.Env):
    """
    6G Hybrid (GBS + LEO) handover RL environment with:
      - pending handover execution delay (D_HO)
      - Ct = 0 during HO execution window and RLF window
      - HSR measurement with success timer (T_succ)
      - Action space: 7 actions (0=stay, 1~5=GBS nodes, 6=LEO)
    Internal node indexing:
      - node 0: LEO
      - node 1: Macro
      - node 2~5: Small cells
    """
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, seed=None):
        super().__init__()

        # -----------------------------
        # [1] Physical / simulation params
        # -----------------------------
        self.map_size = 5000.0
        self.grid_size = 100
        self.pixel_res = self.map_size / self.grid_size
        self.dt = 0.1  # 100 ms

        # Nodes
        self.num_nodes = 6  # (0=LEO, 1=Macro, 2..5=Small)
        self.node_positions = np.zeros((self.num_nodes, 2), dtype=np.float32)
        self.node_positions[1] = [2500, 2500]  # Macro
        self.node_positions[2] = [1250, 1250]
        self.node_positions[3] = [3750, 1250]
        self.node_positions[4] = [1250, 3750]
        self.node_positions[5] = [3750, 3750]

        # Tx power (dBm): [LEO, Macro, Small...]
        self.ptx = np.array(
            [38.0, 46.0, 30.0, 30.0, 30.0, 30.0], dtype=np.float32)
        self.noise_floor_dbm = -90.0

        # RLF rule params
        self.gamma_out_db = -6.0              # RLF threshold
        self.T_fail = 0.2                     # 200ms
        self.N_fail = int(np.ceil(self.T_fail / self.dt))  # 2 steps

        # HO execution delay params (논문 가정)
        self.D_HO_hor = 0.05   # 50ms
        self.D_HO_ver = 0.30   # 300ms
        self.N_HO_hor = int(np.ceil(self.D_HO_hor / self.dt))  # typically 1
        self.N_HO_ver = int(np.ceil(self.D_HO_ver / self.dt))  # typically 3

        # HSR timer params
        # success threshold (same as gamma_out by default)
        self.gamma_th_db = -6.0
        self.T_succ = 0.30        # 300ms
        self.N_succ = int(np.ceil(self.T_succ / self.dt))  # 3 steps

        # Reward weights / costs
        self.c_hor = 10.0
        self.c_ver = 50.0
        self.p_fail = 100.0
        self.w_cap = 1.0
        self.w_cost = 1.0
        self.w_fail = 1.0

        # Shannon capacity bandwidth for Ct (논문 5.3)
        self.B_hz = 100e6

        # Obstacles map (0..1), scaled to max 20 dB attenuation
        self.obstacle_map = np.zeros(
            (self.grid_size, self.grid_size), dtype=np.float32)
        self._generate_obstacles(num_obstacles=20)

        # -----------------------------
        # [2] Action / observation spaces
        # -----------------------------
        # 7 actions: 0=stay, 1..5=GBS, 6=LEO
        self.action_space = spaces.Discrete(7)

        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=1, shape=(3, self.grid_size, self.grid_size), dtype=np.float32),
            "vector": spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32),
        })

        # -----------------------------
        # [3] Internal states
        # -----------------------------
        self.np_random = None
        self.seed(seed)

        self.current_step = 0
        self.max_steps = 10000

        self.ue_pos = np.zeros(2, dtype=np.float32)
        self.ue_velocity = 0.0
        self.ue_direction = 0.0

        self.leo_pos = np.zeros(2, dtype=np.float32)

        # Serving node (internal index 0..5)
        self.serving_node = 1  # start at Macro

        # SINR tracking
        self.current_sinr_db = 10.0
        self.rlf_counter = 0
        self.is_rlf = False

        # Trajectory history (for channel 3)
        self.trajectory_history = []

        # HO pending execution
        self.ho_pending = False
        self.ho_pending_target = None      # internal index 0..5
        self.ho_pending_steps_left = 0
        self.ho_pending_type = None        # "hor" or "ver"
        self.ho_pending_src = None

        # HSR checking window after exec complete
        self.hsr_check_active = False
        self.hsr_check_steps_left = 0
        self.hsr_current_is_success = True
        self.last_ho_exec = None  # dict for last exec-complete event

        # For info/debug
        self.last_ct = 0.0

    def seed(self, seed=None):
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        return [seed]

    def _generate_obstacles(self, num_obstacles=20):
        for _ in range(num_obstacles):
            x = np.random.randint(5, self.grid_size - 15)
            y = np.random.randint(5, self.grid_size - 15)
            w = np.random.randint(3, 10)
            h = np.random.randint(3, 10)
            attenuation = np.random.uniform(0.5, 1.0)
            self.obstacle_map[x:x+w, y:y+h] = attenuation

    # -------------------------------------------------
    # Gym APIs
    # -------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)

        self.current_step = 0
        self.rlf_counter = 0
        self.is_rlf = False

        self.ue_pos = self.np_random.uniform(
            0, self.map_size, size=(2,)).astype(np.float32)
        self.ue_velocity = float(self.np_random.uniform(
            30, 120) * (1000 / 3600))  # m/s
        self.ue_direction = float(self.np_random.uniform(0, 2 * np.pi))

        self.leo_pos = np.array([0.0, 2500.0], dtype=np.float32)
        self.node_positions[0] = self.leo_pos

        self.serving_node = 1
        self.current_sinr_db = 10.0

        self.trajectory_history = [self.ue_pos.copy() for _ in range(5)]

        # HO/HSR states
        self.ho_pending = False
        self.ho_pending_target = None
        self.ho_pending_steps_left = 0
        self.ho_pending_type = None
        self.ho_pending_src = None

        self.hsr_check_active = False
        self.hsr_check_steps_left = 0
        self.hsr_current_is_success = True
        self.last_ho_exec = None
        self.last_ct = 0.0

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action: int):
        self.current_step += 1

        # -----------------------------
        # 0) Basic action validation
        # -----------------------------
        if not isinstance(action, (int, np.integer)):
            action = int(action)

        if action < 0 or action > 6:
            raise ValueError(
                f"Invalid action {action}, action must be in [0..6]")

        # -----------------------------
        # 1) Mobility update
        # -----------------------------
        self._update_mobility()

        # -----------------------------
        # 2) Handle HO pending countdown & possible exec-complete
        # -----------------------------
        ho_exec_complete = False
        ho_exec_event = None

        if self.ho_pending:
            self.ho_pending_steps_left -= 1
            if self.ho_pending_steps_left <= 0:
                # exec complete: switch serving
                src = int(self.ho_pending_src)
                tgt = int(self.ho_pending_target)

                self.serving_node = tgt

                self.ho_pending = False
                self.ho_pending_target = None
                self.ho_pending_type = None
                self.ho_pending_src = None
                self.ho_pending_steps_left = 0

                # start HSR check window
                self.hsr_check_active = True
                self.hsr_check_steps_left = int(self.N_succ)
                self.hsr_current_is_success = True

                ho_exec_complete = True
                ho_exec_event = {"src": src, "tgt": tgt}

                # store for info
                self.last_ho_exec = ho_exec_event

        # -----------------------------
        # 3) Channel quality (SINR) & RLF判定 (serving link 기준)
        # -----------------------------
        self.current_sinr_db = self._calculate_sinr_db(
            serving_node=self.serving_node)
        self.is_rlf = self._update_rlf(self.current_sinr_db)

        # -----------------------------
        # 4) HSR check update (exec complete 이후 N_succ 동안)
        # -----------------------------
        ho_success_flag = None  # None if not decided this step
        if self.hsr_check_active:
            # check threshold
            if self.current_sinr_db < self.gamma_th_db:
                self.hsr_current_is_success = False

            self.hsr_check_steps_left -= 1
            if self.hsr_check_steps_left <= 0:
                # decision point
                ho_success_flag = bool(self.hsr_current_is_success)
                self.hsr_check_active = False
                self.hsr_check_steps_left = 0
                self.hsr_current_is_success = True

        # -----------------------------
        # 5) Decide if a new HO is triggered by this action
        #    - action==0: stay
        #    - action==6: target = LEO (internal 0)
        #    - action==1..5: target = same internal index
        # Rule: if HO pending already, ignore new HO triggers (논문 일관성)
        # -----------------------------
        ho_triggered = False
        ho_trigger_event = None
        ho_cost = 0.0

        if (not self.ho_pending) and (action != 0):
            target_node = 0 if action == 6 else int(action)  # internal index
            if target_node != self.serving_node:
                ho_triggered = True
                src = int(self.serving_node)
                tgt = int(target_node)
                ho_type = "ver" if tgt == 0 or src == 0 else "hor"
                delay_steps = int(
                    self.N_HO_ver) if ho_type == "ver" else int(self.N_HO_hor)

                # start pending execution
                self.ho_pending = True
                self.ho_pending_src = src
                self.ho_pending_target = tgt
                self.ho_pending_type = ho_type
                self.ho_pending_steps_left = max(delay_steps, 1)

                # costs (trigger time에만 1회 부과)
                ho_cost = self.c_ver if ho_type == "ver" else self.c_hor

                ho_trigger_event = {
                    "src": src, "tgt": tgt, "type": ho_type, "delay_steps": self.ho_pending_steps_left}

        # -----------------------------
        # 6) Ct (논문 5.3): RLF or HO execution window => Ct=0
        # HO execution window = pending 상태인 동안
        # -----------------------------
        if self.is_rlf or self.ho_pending:
            ct = 0.0
        else:
            sinr_linear = 10 ** (self.current_sinr_db / 10.0)
            ct = self.B_hz * np.log2(1.0 + sinr_linear)

        self.last_ct = float(ct)

        # -----------------------------
        # 7) Reward
        # - “논문 메트릭”과 “학습 reward”는 엄밀히 동일일 필요는 없지만,
        #   여기서는 정석적으로 Ct(정규화) 기반으로 구성
        # -----------------------------
        if self.is_rlf:
            reward = -(self.w_fail * self.p_fail)
        else:
            # capacity reward: use log2(1+SINR) but 0 during HO pending (Ct rule과 일치)
            if self.ho_pending:
                cap = 0.0
            else:
                sinr_linear = 10 ** (self.current_sinr_db / 10.0)
                cap = float(np.log2(1.0 + sinr_linear))

            reward = (self.w_cap * cap) - (self.w_cost * ho_cost)

        # -----------------------------
        # 8) Termination
        # -----------------------------
        terminated = bool(self.is_rlf)
        truncated = bool(self.current_step >= self.max_steps)

        obs = self._get_obs()
        info = {
            # states
            "step": self.current_step,
            "serving_node": int(self.serving_node),
            "sinr_db": float(self.current_sinr_db),
            "rlf": bool(self.is_rlf),
            "rlf_counter": int(self.rlf_counter),

            # HO pending
            "ho_pending": bool(self.ho_pending),
            "ho_pending_steps_left": int(self.ho_pending_steps_left) if self.ho_pending else 0,
            "ho_pending_type": self.ho_pending_type if self.ho_pending else None,
            "ho_pending_target": int(self.ho_pending_target) if self.ho_pending else None,

            # HO events
            "ho_triggered": bool(ho_triggered),
            "ho_trigger": ho_trigger_event,
            "ho_exec_complete": bool(ho_exec_complete),
            "ho_exec": ho_exec_event,

            # HSR decision (only set when window ends)
            "ho_success_decided": (ho_success_flag is not None),
            "ho_success": ho_success_flag,

            # Ct
            "Ct": float(self.last_ct),
        }

        return obs, reward, terminated, truncated, info

    # -------------------------------------------------
    # Internal logic
    # -------------------------------------------------
    def _update_mobility(self):
        dx = self.ue_velocity * np.cos(self.ue_direction) * self.dt
        dy = self.ue_velocity * np.sin(self.ue_direction) * self.dt
        self.ue_pos += np.array([dx, dy], dtype=np.float32)

        # reflect boundaries
        if self.ue_pos[0] < 0 or self.ue_pos[0] >= self.map_size:
            self.ue_direction = np.pi - self.ue_direction
            self.ue_pos[0] = np.clip(self.ue_pos[0], 0, self.map_size - 1)
        if self.ue_pos[1] < 0 or self.ue_pos[1] >= self.map_size:
            self.ue_direction = -self.ue_direction
            self.ue_pos[1] = np.clip(self.ue_pos[1], 0, self.map_size - 1)

        # LEO movement (7.5 km/s along x)
        self.leo_pos[0] = (self.leo_pos[0] + 7500.0 * self.dt) % self.map_size
        self.node_positions[0] = self.leo_pos

        # trajectory history
        if len(self.trajectory_history) == 0:
            self.trajectory_history = [self.ue_pos.copy() for _ in range(5)]
        else:
            self.trajectory_history.pop(0)
            self.trajectory_history.append(self.ue_pos.copy())

    def _update_rlf(self, sinr_db: float) -> bool:
        # RLF timer: N_fail consecutive below gamma_out
        if sinr_db < self.gamma_out_db:
            self.rlf_counter += 1
            if self.rlf_counter >= self.N_fail:
                return True
        else:
            self.rlf_counter = 0
        return False

    def _calculate_sinr_db(self, serving_node: int) -> float:
        # UE pixel
        px = int(np.clip(self.ue_pos[0] /
                 self.pixel_res, 0, self.grid_size - 1))
        py = int(np.clip(self.ue_pos[1] /
                 self.pixel_res, 0, self.grid_size - 1))

        # obstacle attenuation (0..1 -> up to 20 dB)
        blockage_atten = float(self.obstacle_map[px, py]) * 20.0

        rx_power_dbm = np.zeros(self.num_nodes, dtype=np.float32)

        for i in range(self.num_nodes):
            dist = float(np.linalg.norm(self.ue_pos - self.node_positions[i]))
            dist = max(dist, 1.0)

        f_hz = 2.0e9
        c = 3.0e8

        if i == 0:
            # LEO: FSPL + altitude 600 km
            slant_dist_m = np.sqrt(dist**2 + 600000.0**2)  # meters
            fspl_db = 20.0 * np.log10(4.0 * np.pi * slant_dist_m * f_hz / c)
            rx_power_dbm[i] = self.ptx[i] - fspl_db
        else:
            # GBS: FSPL + 추가 감쇠(도심) + blockage
            # (여기서 15dB는 도심 NLoS 성향을 약하게 반영하는 상수. 필요하면 0~30에서 튜닝)
            fspl_db = 20.0 * np.log10(4.0 * np.pi * dist * f_hz / c)
            rx_power_dbm[i] = self.ptx[i] - (fspl_db + 15.0 + blockage_atten)
            rx_linear = 10 ** (rx_power_dbm / 10.0)  # mW
            noise_linear = 10 ** (self.noise_floor_dbm / 10.0)

        s = rx_linear[int(serving_node)]
        if int(serving_node) == 0:
            # LEO serving: 현재 구현은 위성 1개뿐이라 간섭 0으로 둠
            i = 0.0
        else:
            # GBS serving: 다른 GBS만 간섭으로 합산
            i = float(np.sum(rx_linear[1:]) - s)

        sinr_linear = float(s / (i + noise_linear + 1e-12))
        sinr_db = 10.0 * np.log10(sinr_linear + 1e-12)
        return float(sinr_db)

    def _get_obs(self):
        image = np.zeros((3, self.grid_size, self.grid_size), dtype=np.float32)

        # ch1: obstacles + GBS infra marks
        image[0] = self.obstacle_map.copy()
        for i in range(1, self.num_nodes):
            px = int(
                np.clip(self.node_positions[i][0] / self.pixel_res, 0, self.grid_size - 1))
            py = int(
                np.clip(self.node_positions[i][1] / self.pixel_res, 0, self.grid_size - 1))
            image[0, px, py] = 1.0

        # ch2: LEO footprint
        lpx = int(np.clip(self.leo_pos[0] /
                  self.pixel_res, 0, self.grid_size - 1))
        lpy = int(np.clip(self.leo_pos[1] /
                  self.pixel_res, 0, self.grid_size - 1))
        y_idx, x_idx = np.ogrid[:self.grid_size, :self.grid_size]
        dist_from_leo = np.sqrt((x_idx - lpx) ** 2 + (y_idx - lpy) ** 2)
        footprint = np.clip(1.0 - (dist_from_leo / 20.0), 0, 1)
        image[1] = footprint.astype(np.float32)

        # ch3: trajectory with decay
        decay = 1.0
        for pos in reversed(self.trajectory_history[-5:]):
            px = int(np.clip(pos[0] / self.pixel_res, 0, self.grid_size - 1))
            py = int(np.clip(pos[1] / self.pixel_res, 0, self.grid_size - 1))
            image[2, px, py] = max(image[2, px, py], decay)
            decay *= 0.8

        # vector
        # [0] serving node norm
        # [1] velocity norm (33.3 m/s ~ 120 km/h)
        # [2] sinr norm
        # [3] rlf risk (counter/N_fail)
        # [4] is satellite serving (serving_node==0)
        sinr_norm = np.clip((self.current_sinr_db + 20.0) / 50.0, 0.0, 1.0)
        rlf_risk = np.clip(self.rlf_counter / max(self.N_fail, 1), 0.0, 1.0)

        vec = np.array([
            float(self.serving_node) / 5.0,
            float(self.ue_velocity) / 33.3,
            float(sinr_norm),
            float(rlf_risk),
            1.0 if self.serving_node == 0 else 0.0
        ], dtype=np.float32)

        return {"image": image, "vector": vec}
