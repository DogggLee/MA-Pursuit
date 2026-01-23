import numpy as np
import matplotlib.pyplot as plt
import torch as T
import utils
from Lidar import Lidar

"""
多目标围捕强化学习环境核心模块
核心功能：
1. 构建包含猎人、逃逸者、障碍物的2D有界仿真环境
2. 实现智能体运动、感知（激光雷达）、奖励计算、状态观测逻辑
3. 支持环境重置、步更新、可视化渲染等强化学习标准接口

核心类说明：
- MultiTarEnv: 主环境类，管理所有智能体、障碍物和环境交互逻辑
- Obstacle: 障碍物类，定义障碍物位置、尺寸等属性
- Hunter/Target: 智能体基类（文件中未完整展示，逻辑内联在MultiTarEnv中）
"""

# 设置全局随机种子，保证实验可复现
def set_global_seeds(seed):
    """
    设置全局随机种子，统一numpy、random、torch的随机数生成器

    Args:
        seed (int): 随机种子值
    """
    import random
    np.random.seed(seed)
    random.seed(seed)
    T.manual_seed(seed)

class MultiTarEnv:
    """
    多目标围捕环境类
    核心参数（建议外部传入）：
    - length: 环境边界尺寸（km）
    - num_obstacle: 障碍物数量
    - num_hunters: 猎人智能体数量
    - num_targets: 逃逸者智能体数量
    - h_actor_dim: 猎人观测维度
    - t_actor_dim: 逃逸者观测维度
    - action_dim: 智能体动作维度
    - visualize_lasers: 是否可视化激光雷达扫描结果
    - env_config: 环境配置字典，包含所有magic number的外部配置
    """
    def __init__(self, length, num_obstacle, num_hunters, num_targets,
                 h_actor_dim, t_actor_dim, action_dim, visualize_lasers=False, env_config=None):
        # 基础环境参数（必传）
        self.length = length  # 环境边界尺寸（km）
        self.num_obstacle = num_obstacle  # 障碍物数量
        self.num_hunters = num_hunters  # 猎人数量
        self.num_targets = num_targets  # 逃逸者数量

        self.h_actor_dim = h_actor_dim  # 猎人观测维度
        self.t_actor_dim = t_actor_dim  # 逃逸者观测维度
        self.action_dim = action_dim  # 智能体动作维度

        self.visualize_lasers = visualize_lasers  # 是否可视化激光雷达

        # 加载外部配置（优先级：外部配置 > 默认值）
        self._load_env_config(env_config)

        # 初始化障碍物、猎人、逃逸者
        self.obstacles = [Obstacle(length=self.length) for _ in range(self.num_obstacle)]
        self.hunters = [
            Hunter(
                length=self.length, 
                L_sensor=self.L_sensor,
                num_lasers=self.num_lasers, 
                time_step=self.time_step, 
                obstacles=self.obstacles
            ) 
            for _ in range(self.num_hunters)
        ]
        self.targets = [
            Target(
                length=self.length, 
                L_sensor=self.L_sensor,
                num_lasers=self.num_lasers, 
                time_step=self.time_step, 
                obstacles=self.obstacles
            ) 
            for _ in range(self.num_targets)
        ]

        # 可视化相关初始化
        self.fig = plt.figure(figsize=(8,8))
        self.ax = self.fig.add_subplot(111,projection='3d')

    def _load_env_config(self, env_config):
        """
        加载环境配置，统一管理magic number，支持外部传入覆盖默认值

        Args:
            env_config (dict): 环境配置字典，格式示例：
                {
                    "time_step": 0.5,          # 时间步长
                    "v_max": 0.05,             # 最大速度
                    "a_max": 0.01,             # 最大加速度
                    "num_lasers": 16,          # 激光雷达波束数
                    "L_sensor": 0.2,           # 传感器最大探测距离
                    "escape_distance": 0.05,   # 逃逸距离阈值
                    "distance_threshold": 0.01,# 碰撞距离阈值
                    "max_escape_angle": 30,    # 最大逃逸角度
                    # 奖励系数
                    "capture_reward": 2.0,
                    "chase_reward_coeff": 0.8,
                    "escape_reward_coeff": 0.4,
                    "safe_penalty_coeff": 0.7
                }
        """
        # 默认配置（所有magic number集中管理）
        default_config = {
            # 时间/运动相关
            "time_step": 0.5,               # 状态更新时间步（s）
            "v_max": 0.05,                  # 智能体最大速度（km/s）
            "a_max": 0.01,                  # 智能体最大加速度（km/s²）
            # 传感器相关
            "num_lasers": 16,               # 激光雷达波束数量
            "L_sensor": 0.2,                # 激光雷达最大探测距离（km）
            # 围捕/逃逸相关阈值
            "escape_distance": 0.05,        # 逃逸者成功逃逸的距离阈值（km）
            "distance_threshold": 0.01,     # 碰撞/围捕成功的距离阈值（km）
            "max_escape_angle": 30,         # 逃逸者最大逃逸角度（度）
            # 奖励系数
            "capture_reward": 2.0,          # 猎人围捕成功奖励
            "chase_reward_coeff": 0.8,      # 猎人追击奖励系数
            "escape_reward_coeff": 0.4,     # 逃逸者逃逸奖励系数
            "safe_penalty_coeff": 0.7,      # 碰撞惩罚系数
            # 初始化位置范围
            "hunter_init_low": 0.50,        # 猎人初始位置下限（km）
            "hunter_init_high": 0.75,       # 猎人初始位置上限（km）
            "target_init_low": 1.50,        # 逃逸者初始位置下限（km）
            "target_init_high": 1.75,       # 逃逸者初始位置上限（km）
            "init_height": 0.10             # 智能体初始高度（km）
        }

        # 合并外部配置与默认配置
        self.config = default_config.copy()
        if env_config is not None and isinstance(env_config, dict):
            self.config.update(env_config)

        # 将配置赋值为实例属性（方便调用）
        for key, value in self.config.items():
            setattr(self, key, value)

    def _collect_obs_info(self):
        """收集所有障碍物的观测信息（预留接口，暂未实现）"""
        multi_obs_info = []
        for obstacle in self.obstacles:
            multi_obs_info.append(obstacle._return_obs_info())

    def reset(self):
        """
        重置环境到初始状态，生成初始观测

        Returns:
            h_obs (list of np.array): 所有猎人的初始观测
            t_obs (list of np.array): 所有逃逸者的初始观测
        """
        # 重置猎人初始位置和状态
        for hunter in self.hunters:
            hunter.position = np.random.uniform(
                low=self.hunter_init_low, 
                high=self.hunter_init_high, 
                size=(3,)
            )
            hunter.position[-1] = self.init_height  # 初始高度
            hunter.velocity = np.zeros(3)
            hunter.history_pos = []
            hunter.lasers = hunter.lidar.scan(hunter.position, self.length)

        # 重置逃逸者初始位置和状态
        for target in self.targets:
            target.position = np.random.uniform(
                low=self.target_init_low, 
                high=self.target_init_high, 
                size=(3,)
            )
            target.position[-1] = self.init_height  # 初始高度
            target.velocity = np.zeros(3)
            target.history_pos = []
            target.lasers = target.lidar.scan(target.position, self.length)

        # 为猎人分配初始围捕目标
        self._assign_targets_to_hunters()

        # 获取初始观测
        h_obs, t_obs = self._get_observations()

        return h_obs, t_obs

    def step(self, actions):
        """
        执行智能体动作，更新环境状态，计算奖励和终止标志

        Args:
            actions (list of np.array): 所有智能体的动作（先猎人，后逃逸者）

        Returns:
            h_obs_next (list of np.array): 猎人下一时刻观测
            t_obs_next (list of np.array): 逃逸者下一时刻观测
            rewards (list of float): 所有智能体的奖励（先猎人，后逃逸者）
            dones (list of bool): 所有智能体的终止标志（先猎人，后逃逸者）
        """
        # 执行猎人动作
        for i, hunter in enumerate(self.hunters):
            hunter.move(actions[i], self.v_max)
        
        # 执行逃逸者动作
        for i, target in enumerate(self.targets):
            target.move(actions[self.num_hunters + i], self.v_max)

        # 更新激光雷达扫描数据
        for hunter in self.hunters:
            hunter.lasers = hunter.lidar.scan(hunter.position, self.length)
        for target in self.targets:
            target.lasers = target.lidar.scan(target.position, self.length)

        # 边界裁剪：限制智能体在环境范围内
        for agent in self.hunters + self.targets:
            agent.position[:2] = np.clip(agent.position[:2], 0, self.length)

        # 计算奖励和终止标志
        rewards, dones = self._compute_rewards()

        # 获取下一时刻观测
        h_obs_next, t_obs_next = self._get_observations()

        return h_obs_next, t_obs_next, rewards, dones

    def _assign_targets_to_hunters(self):
        """
        为猎人分配围捕目标（TODO：实现基于密度的分配算法）
        核心逻辑：按距离分配，保证每个逃逸者分配到数量均衡的猎人
        """
        num_hunters = self.num_hunters
        num_targets = self.num_targets
        targets = self.targets
        
        # 计算每个猎人到每个逃逸者的距离
        hunter_target_distances = []
        for hunter in self.hunters:
            distances = [np.linalg.norm(hunter.position[:2] - target.position[:2]) for target in targets]
            hunter_target_distances.append(distances)
        
        # 计算每个逃逸者分配的猎人数量（均衡分配）
        ## 均分后，剩余的记作 extra_hunters, 靠前的目标优先多分配一个hunter（实际上应该根据目标的价值来配置）
        base_hunters_per_target = num_hunters // num_targets
        extra_hunters = num_hunters % num_targets
        hunters_per_target = [
            base_hunters_per_target + 1 if i < extra_hunters else base_hunters_per_target 
            for i in range(num_targets)
        ]
        
        hunter_indices = list(range(num_hunters))

        # 为每个逃逸者分配猎人
        for target_idx in range(num_targets):
            num_to_assign = hunters_per_target[target_idx]
            # 按距离排序，分配最近的猎人
            sorted_hunters = sorted(
                hunter_indices, 
                key=lambda x: hunter_target_distances[x][target_idx]
            )
            assigned_hunters = sorted_hunters[:num_to_assign]
            # 记录分配结果
            for hunter_idx in assigned_hunters:
                self.hunters[hunter_idx].assigned_target = targets[target_idx]
                hunter_indices.remove(hunter_idx)

    def _get_nearest_target(self, hunter):
        """
        找到离指定猎人最近的逃逸者

        Args:
            hunter (Hunter): 目标猎人

        Returns:
            Target: 最近的逃逸者
        """
        min_dist = float('inf')
        nearest = None
        for target in self.targets:
            # 计算2D平面距离（TODO：扩展为3D距离）
            dist = np.linalg.norm(hunter.position[:2] - target.position[:2])
            if dist < min_dist:
                min_dist = dist
                nearest = target
        return nearest
    
    def _get_observations(self):
        """
        计算所有智能体的观测向量

        Returns:
            h_obs (list of np.array): 猎人观测列表
            t_obs (list of np.array): 逃逸者观测列表
        """
        h_obs = []
        t_obs = []

        # 预计算所有猎人位置（加速最近邻计算）
        hunter_positions = np.array([hunter.position for hunter in self.hunters])

        # 计算猎人观测
        for i, hunter in enumerate(self.hunters):
            # 找到最近的2个其他猎人（不足则补0）
            other_hunters = np.delete(hunter_positions, i, axis=0)
            if len(other_hunters) >= 2:
                distances = np.linalg.norm(other_hunters[:, :2] - hunter.position[:2], axis=1)
                nearest_indices = distances.argsort()[:2]
                nearest_hunters = other_hunters[nearest_indices]
            else:
                nearest_hunters = np.zeros((2, 3))
                if len(other_hunters) == 1:
                    nearest_hunters[0] = other_hunters[0]

            # 提取观测分量（归一化）
            velocity = hunter.velocity / self.v_max  # 速度归一化
            # 分配目标的位置和距离（归一化）
            if hunter.assigned_target is not None:
                target_pos = hunter.assigned_target.position / self.length
                distance_to_target = np.linalg.norm(hunter.position[:2] - hunter.assigned_target.position[:2]) / (np.sqrt(2)*self.length)
            else:
                target_pos = np.zeros(3)
                distance_to_target = 0.0
            laser_data = hunter.lasers / self.L_sensor  # 激光数据归一化

            # 拼接观测向量
            obs = np.concatenate([
                nearest_hunters.flatten()/self.length,  # 最近2个猎人位置（6维）
                hunter.position/self.length,            # 自身位置（3维）
                velocity,                                # 自身速度（3维）
                target_pos,                              # 分配目标位置（3维）
                np.array([distance_to_target]),          # 到目标距离（1维）
                laser_data                               # 激光雷达数据（num_lasers维）
            ]).astype(np.float32)
            h_obs.append(obs)

        # 计算逃逸者观测
        for target in self.targets:
            # 提取观测分量（归一化）
            own_pos = target.position / self.length  # 自身位置（3维）
            own_vel = target.velocity / self.v_max   # 自身速度（3维）
            # 最近3个猎人位置（不足则补0，归一化）
            distances = np.linalg.norm(hunter_positions[:, :2] - target.position[:2], axis=1)
            nearest_indices = distances.argsort()[:3]
            nearest_hunters = hunter_positions[nearest_indices]
            if len(nearest_hunters) < 3:
                pad_size = 3 - len(nearest_hunters)
                nearest_hunters = np.vstack([nearest_hunters, np.zeros((pad_size, 3))])
            nearest_hunters = nearest_hunters.flatten() / self.length  # 9维
            laser_data = target.lasers / self.L_sensor  # 激光数据（num_lasers维）

            # 拼接观测向量
            obs = np.concatenate([
                own_pos,
                own_vel,
                nearest_hunters,
                laser_data
            ]).astype(np.float32)
            t_obs.append(obs)

        return h_obs, t_obs

    def _compute_rewards(self):
        """
        计算所有智能体的奖励和终止标志

        Returns:
            rewards (list of float): 奖励列表（先猎人，后逃逸者）
            dones (list of bool): 终止标志列表（先猎人，后逃逸者）
        """
        rewards = [0.0] * (self.num_hunters + self.num_targets)
        dones = [False] * (self.num_hunters + self.num_targets)

        # 构建「逃逸者-分配猎人」映射
        target_hunter_groups = {}
        for target in self.targets:
            target_hunter_groups[target] = [hunter for hunter in self.hunters if hunter.assigned_target == target]

        # 1. 猎人奖励：追击奖励 + 围捕成功奖励
        for target, hunters in target_hunter_groups.items():
            for hunter in hunters:
                # 追击奖励：基于猎人朝向与目标方向的余弦相似度
                hunter_dir = hunter.velocity[:2]
                if np.linalg.norm(hunter_dir) == 0:
                    hunter_dir_unit = np.zeros(2)
                else:
                    hunter_dir_unit = hunter_dir / np.linalg.norm(hunter_dir)
                target_dir = target.position[:2] - hunter.position[:2]
                if np.linalg.norm(target_dir) == 0:
                    target_dir_unit = np.zeros(2)
                else:
                    target_dir_unit = target_dir / np.linalg.norm(target_dir)
                chase_reward = np.dot(hunter_dir_unit, target_dir_unit)
                hunter_index = self.hunters.index(hunter)
                rewards[hunter_index] += self.chase_reward_coeff * chase_reward

            # 围捕成功奖励：判断是否满足围捕条件
            multi_hunters_pos = [h.position for h in hunters]
            if utils.isRounded(
                tuple(target.position[:2]), 
                [tuple(row[:2]) for row in multi_hunters_pos], 
                self.L_sensor, 
                self.max_escape_angle
            ):
                for hunter in hunters:
                    hunter_index = self.hunters.index(hunter)
                    rewards[hunter_index] += self.capture_reward
                target_index = self.targets.index(target)
                dones[self.num_hunters + target_index] = True  # 逃逸者被围捕，终止

        # 2. 逃逸者奖励：逃逸奖励
        for target in self.targets:
            target_index = self.targets.index(target)
            if dones[self.num_hunters + target_index]:
                continue  # 被围捕则无额外奖励

            hunters = target_hunter_groups[target]
            if len(hunters) != 0:
                distances = [np.linalg.norm(target.position[:2] - hunter.position[:2]) for hunter in hunters]
                nearest_distance = min(distances)
            else:
                nearest_distance = np.inf

            # 逃逸奖励：距离超过阈值则正向奖励，否则负向
            escape_reward = 0.1 if nearest_distance > self.escape_distance else -0.1
            rewards[self.num_hunters + target_index] += self.escape_reward_coeff * escape_reward

        # 3. 碰撞惩罚：智能体之间碰撞
        # 猎人之间碰撞
        for i in range(self.num_hunters):
            for j in range(i+1, self.num_hunters):
                distance = np.linalg.norm(self.hunters[i].position[:2] - self.hunters[j].position[:2])
                if distance < self.distance_threshold:
                    penalty = self.safe_penalty_coeff * (self.distance_threshold - distance)
                    rewards[i] -= penalty
                    rewards[j] -= penalty

        # 猎人-逃逸者碰撞
        for hunter in self.hunters:
            for target in self.targets:
                distance = np.linalg.norm(hunter.position[:2] - target.position[:2])
                if distance < self.distance_threshold:
                    rewards[self.hunters.index(hunter)] -= self.safe_penalty_coeff * (self.distance_threshold - distance)
                    rewards[self.num_hunters + self.targets.index(target)] -= self.safe_penalty_coeff * (self.distance_threshold - distance)
        
        # 逃逸者之间碰撞
        for i in range(self.num_targets):
            for j in range(i+1, self.num_targets):
                distance = np.linalg.norm(self.targets[i].position[:2] - self.targets[j].position[:2])
                if distance < self.distance_threshold:
                    penalty = self.safe_penalty_coeff * (self.distance_threshold - distance)
                    rewards[self.num_hunters + i] -= penalty
                    rewards[self.num_hunters + j] -= penalty

        # 4. 障碍物碰撞惩罚：基于激光雷达最小探测距离
        for agent in self.hunters + self.targets:
            min_laser_length = min(agent.lasers)
            collision_penalty = -self.safe_penalty_coeff * (self.L_sensor - min_laser_length) / self.L_sensor
            if agent in self.hunters:
                agent_index = self.hunters.index(agent)
            else:
                agent_index = self.targets.index(agent) + self.num_hunters
            rewards[agent_index] += collision_penalty

        return rewards, dones
    
    def rewardNorm(self, rewards):
        """
        对奖励进行归一化（分猎人和逃逸者分别归一化）

        Args:
            rewards (list of float): 原始奖励列表

        Returns:
            normalized_rewards (list of float): 归一化后的奖励列表
        """
        h_rewards = rewards[:self.num_hunters]
        t_rewards = rewards[self.num_hunters:]
        
        # 猎人奖励归一化
        h_mean = np.mean(h_rewards)
        h_std = np.std(h_rewards)
        h_normalized = h_rewards if h_std == 0 else (h_rewards - h_mean) / h_std
        
        # 逃逸者奖励归一化
        t_mean = np.mean(t_rewards)
        t_std = np.std(t_rewards)
        t_normalized = t_rewards if t_std == 0 else (t_rewards - t_mean) / t_std
        
        return list(h_normalized) + list(t_normalized)

    def render(self):
        """
        可视化环境状态（3D渲染）
        - 绘制环境边界、障碍物、猎人（红）、逃逸者（绿）
        - 可选可视化激光雷达扫描结果
        """
        self.ax.clear()
        self.ax.set_xlim(0, self.length)
        self.ax.set_ylim(0, self.length)
        self.ax.set_zlim(0, self.length/4)
        self.ax.set_title("Multi-Target Pursuit Environment")

        # 绘制环境边界
        self.ax.plot(
            [0, self.length, self.length, 0, 0], 
            [0, 0, self.length, self.length, 0], 
            [0, 0, 0, 0, 0], 
            color='black', linewidth=2
        )

        # 绘制障碍物（圆柱体）
        for obstacle in self.obstacles:
            cx, cy, cz, r, h = obstacle._return_obs_info()
            self._create_cylinders(self.ax, cx, cy, cz, r, h)

        # 绘制猎人
        for hunter in self.hunters:
            x, y, z = hunter.position
            self.ax.scatter(
                x, y, z, 
                color='red', 
                label='Hunter' if hunter == self.hunters[0] else ""
            )
            if self.visualize_lasers:
                hunter.lidar.visualize_lasers(hunter.position, self.ax)

        # 绘制逃逸者
        for target in self.targets:
            x, y, z = target.position
            self.ax.scatter(
                x, y, z, 
                color='green', 
                label='Target' if target == self.targets[0] else ""
            )

        self.ax.legend(loc='upper right')
        plt.pause(0.001)

    def close(self):
        """关闭可视化窗口"""
        plt.close(self.fig)

    def _create_cylinders(self, ax, x, y, z, r, h):
        """
        在3D轴上绘制圆柱体（表示障碍物）

        Args:
            ax (matplotlib.axes.Axes3D): 3D绘图轴
            x (float): 圆柱体中心x坐标
            y (float): 圆柱体中心y坐标
            z (float): 圆柱体底部z坐标
            r (float): 圆柱体半径
            h (float): 圆柱体高度
        """
        # 采样生成圆柱体表面点（减少采样数提升性能）
        theta = np.linspace(0, 2 * np.pi, 16) 
        z_vals = np.linspace(z, z + h, 8)   
        theta, z_vals = np.meshgrid(theta, z_vals)
        x_vals = x + r * np.cos(theta)
        y_vals = y + r * np.sin(theta)

        ax.plot_surface(x_vals, y_vals, z_vals, color='black', alpha=0.5)

class Obstacle:
    """
    障碍物类，定义障碍物的位置、尺寸等属性
    """
    def __init__(self, length=2, speed=0):
        # 随机初始化障碍物位置（避开边界）
        self.position = np.random.uniform(
            low=0.45, 
            high=length-0.55, 
            size=(3,)
        )
        self.position[-1] = 0  # 初始z坐标为0
        self.speed = speed     # 障碍物速度（默认静止）
        self.angle = np.random.uniform(0, 2 * np.pi)  # 随机角度（预留运动方向）

    def _return_obs_info(self):
        """
        返回障碍物的观测信息（中心坐标、半径、高度）
        （注：原始代码未完整实现，需补充具体逻辑）
        """
        # 示例返回（需根据实际需求调整）
        return (
            self.position[0], 
            self.position[1], 
            self.position[2], 
            0.1,  # 障碍物半径
            0.2   # 障碍物高度
        )

# 补充Hunter/Target类的基础定义（原始代码中未完整展示，保证代码可运行）
class Hunter:
    def __init__(self, length, L_sensor, num_lasers, time_step, obstacles):
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.history_pos = []
        self.lidar = Lidar(num_lasers, L_sensor, obstacles)
        self.lasers = np.zeros(num_lasers)
        self.assigned_target = None
        self.time_step = time_step

    def move(self, action, v_max):
        """执行动作更新位置和速度"""
        self.velocity[:2] += action * self.time_step
        self.velocity = np.clip(self.velocity, -v_max, v_max)
        self.position[:2] += self.velocity[:2] * self.time_step
        self.history_pos.append(self.position.copy())

class Target:
    def __init__(self, length, L_sensor, num_lasers, time_step, obstacles):
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.history_pos = []
        self.lidar = Lidar(num_lasers, L_sensor, obstacles)
        self.lasers = np.zeros(num_lasers)
        self.time_step = time_step

    def move(self, action, v_max):
        """执行动作更新位置和速度"""
        self.velocity[:2] += action * self.time_step
        self.velocity = np.clip(self.velocity, -v_max, v_max)
        self.position[:2] += self.velocity[:2] * self.time_step
        self.history_pos.append(self.position.copy())
