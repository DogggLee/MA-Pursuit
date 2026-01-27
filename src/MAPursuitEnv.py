import numpy as np
import matplotlib.pyplot as plt
import torch as T
from . import utils
from .Lidar import Lidar

'''
    class MultiTarEnv: is a world created with UAVs (multi-target&hunter) and obstacles in a limited square area.
                       param : 
                            h_actor_dim: a list to store hunters' output dimensions
                            t_actor_dim: a list to store targets' output dimensions
                       funcs :

    class Obstacle: defined obstacles and their contributions
    class Hunter & Target: they'll all be equipped with a sensor to detect obstacles nearby and other UAVs, 
                           and an AI brain(agent network) to make decision, with some elemental contributions.
'''

# add a global random seed to control randomness
def set_global_seeds(seed):
        import random
        np.random.seed(seed)
        random.seed(seed)
        T.manual_seed(seed)

class MAPursuitEnv:
    def __init__(self,
                 config,
                 h_actor_dim,
                 t_actor_dim,
                 visualize_lasers=False):
        self.map_size = config.Env.map_size # length of boundary

        self.num_obstacle = config.Env.num_obstacle # number of obstacles
        self.num_hunters = config.Env.num_hunters # number of hunters
        self.num_targets = config.Env.num_targets # number of targets

        self.h_actor_dim = h_actor_dim # each hunter's observation dimension
        self.t_actor_dim = t_actor_dim # each target's observation dimension

        self.time_step = config.Env.time_step # update time step
        
        self.escape_distance = config.Env.escape_dist # escape distance threshold for target
        self.max_escape_angle = config.Env.escape_angle # max escape angle for target (degree)

        self.distance_threshold = config.Env.collision_dist # distance threshold for collision
        
        ## Instancing hunters and targets, obstacles.
        self.config = config
        self.gen_env(config)

        # Control whether to visualize laser scans & plot relevant
        self.visualize_lasers = visualize_lasers
        self.fig = plt.figure(figsize=(8,8))
        self.ax = self.fig.add_subplot(111,projection='3d')

        # reward relevant
        self.capture_reward = config.Reward.capture_reward_w        # roundup success reward
        self.chase_reward_coeff = config.Reward.chase_reward_w    # chase reward coeff
        self.escape_reward_coeff = config.Reward.escape_reward_w   # escape reward coeff
        self.safe_penalty_coeff = config.Reward.safe_penalty_w    # safe penalty coeff
    
    def _collect_obs_info(self):
        multi_obs_info = []
        for obstacle in self.obstacles:
            multi_obs_info.append(obstacle._return_obs_info())

    def gen_env(self, config):
        self.num_obs = np.random.randint(self.num_obstacle[0], self.num_obstacle[1]+1)
        self.num_hunter = np.random.randint(self.num_hunters[0], self.num_hunters[1]+1)
        self.num_target = np.random.randint(self.num_targets[0], self.num_targets[1]+1)

        # set of obstacles
        self.obstacles = []
        for _ in range(self.num_obs):
            obs = Obstacle(config.Env.map_size, config.Env.obstacle_margin,
                           config.Env.obstacle_radius_range, config.Env.obstacle_height_range,
                           speed=config.Env.obstacle_speed)
            self.obstacles.append(obs)

        # set of hunters
        self.hunters = []
        for _ in range(self.num_hunter):   
            target_lidar = Lidar(config.Hunter.max_lidar_dist, config.Hunter.lidar_beams, self.obstacles)
            target = Hunter(target_lidar, config.Env.map_size, 
                            config.Env.hunter_margin, config.Env.hunter_init_height,
                            config.Hunter.max_vel, config.Hunter.max_acc, config.Env.time_step)
            self.hunters.append(target)

        # set of targets
        self.targets = []
        for _ in range(self.num_target):   
            target_lidar = Lidar(config.Target.max_lidar_dist, config.Target.lidar_beams, self.obstacles)
            target = Hunter(target_lidar, config.Env.map_size, 
                            config.Env.target_margin, config.Env.target_init_height,
                            config.Target.max_vel, config.Target.max_acc, config.Env.time_step)
            self.targets.append(target)

    def re_gen(self):
        self.gen_env(self.config)


    def reset(self):
        '''
        Reset the environment to an initial state and returns the initial observations.
        Returns:
            h_obs (list of np.array): Observations for all hunters.
            t_obs (list of np.array): Observations for all targets.
        '''
        for hunter in self.hunters:
            hunter.reset()
        for target in self.targets:
            target.reset()

        # Assign initial targets to hunters
        self._assign_targets_to_hunters()

        # Get initial observations
        h_obs, t_obs = self._get_observations()

        return h_obs,t_obs

    # TODO!!!
    def step(self,actions):
        """
        Apply actions to agents, update the environment state, compute rewards, and return observations.
        Args:
            actions (list of np.array): Actions for all agents (hunters followed by targets).
        Returns:
            h_obs_next (list of np.array): Next observations for all hunters.
            t_obs_next (list of np.array): Next observations for all targets.
            rewards (list of float): Rewards for all agents.
            dones (list of bool): Done flags for all agents.
        """
        # Apply actions to hunters
        for i, hunter in enumerate(self.hunters):
            hunter.move(actions[i])
        
        # Apply actions to targets
        for i, target in enumerate(self.targets):
            target.move(actions[self.num_hunter + i]) # Assume "action" set is combination of hunetrs' & targets'

        # Update lasers after movement
        for hunter in self.hunters:
            # hunter.lasers = hunter.lidar.scan(hunter.position, self.map_size)
            hunter.update_lidar()
        for target in self.targets:
            # target.lasers = target.lidar.scan(target.position, self.map_size)
            target.update_lidar()

        # Optionally: Boundary blocking (when train agents, 
        # to allow agents traverse the boundary may get worse performance)
        for agent in self.hunters + self.targets:
            agent.position[:2] = np.clip(agent.position[:2], 0, self.map_size)

        # # Assign targets to hunters
        # self._assign_targets_to_hunters()

        # Compute rewards and check for captures
        rewards, dones = self._compute_rewards()

        # Get next observations
        h_obs_next, t_obs_next = self._get_observations()

        return h_obs_next, t_obs_next, rewards, dones

    # TODO: introduce a density-based assignment method
    def _assign_targets_to_hunters(self):
        num_hunters = self.num_hunter
        num_targets = self.num_target
        targets = self.targets
        
        hunter_target_distances = []
        for hunter in self.hunters:
            distances = [np.linalg.norm(hunter.position[:2] - target.position[:2]) for target in targets]
            hunter_target_distances.append(distances)
        
        base_hunters_per_target = num_hunters // num_targets
        extra_hunters = num_hunters % num_targets
        hunters_per_target = [base_hunters_per_target + 1 if i < extra_hunters else base_hunters_per_target for i in range(num_targets)]
        
        hunter_indices = list(range(num_hunters))

        for target_idx in range(num_targets):

            num_to_assign = hunters_per_target[target_idx]
            sorted_hunters = sorted(hunter_indices, key=lambda x: hunter_target_distances[x][target_idx])
            assigned_hunters = sorted_hunters[:num_to_assign]
            for hunter_idx in assigned_hunters:
                self.hunters[hunter_idx].assigned_target = targets[target_idx]
                hunter_indices.remove(hunter_idx)

    def _get_nearest_target(self, hunter):
        """
        Find the nearest target to the given hunter.
        Args:
            hunter (Hunter): The hunter to find the nearest target for.
        Returns:
            Target: The nearest target.
        """
        min_dist = float('inf')
        nearest = None
        for target in self.targets:
            dist = np.linalg.norm(hunter.position[:2] - target.position[:2])  # TODO: 2D distance -> 3D
            if dist < min_dist:
                min_dist = dist
                nearest = target
        return nearest
    
    def _get_observations(self):
        """
        Compute observations for all hunters and targets.
        Returns:
            h_obs (list of np.array): Observations for all hunters.
            t_obs (list of np.array): Observations for all targets.
        """
        h_obs = []
        t_obs = []

        # Precompute hunter positions for nearest hunters
        hunter_positions = np.array([hunter.position for hunter in self.hunters])

        # Compute observations for hunters
        for i, hunter in enumerate(self.hunters):
            # Get other hunters' positions excluding itself
            other_hunters = np.delete(hunter_positions, i, axis=0)
            # Find two nearest hunters
            if len(other_hunters) >= 2:
                distances = np.linalg.norm(other_hunters[:, :2] - hunter.position[:2], axis=1)
                nearest_indices = distances.argsort()[:2]
                nearest_hunters = other_hunters[nearest_indices]
            else:
                # If less than two other hunters, pad with zeros
                nearest_hunters = np.zeros((2, 3))
                if len(other_hunters) == 1:
                    nearest_hunters[0] = other_hunters[0]
                    nearest_hunters[1] = np.zeros(3)
                else:
                    nearest_hunters = np.zeros((2, 3))

            # Get velocity
            velocity = hunter.velocity

            # Get assigned target's position and distance
            if hunter.assigned_target is not None:
                target_pos = hunter.assigned_target.position
                distance_to_target = np.linalg.norm(hunter.position[:2] - target_pos[:2])
            else:
                target_pos = np.zeros(3)
                distance_to_target = 0.0

            # Get laser data
            laser_data = hunter.lasers  # Assuming it's a 1D array of size num_lasers

            # Concatenate all observation components
            obs = np.concatenate([
                nearest_hunters.flatten()/self.map_size,                   # 2 * 3 = 6
                hunter.position/self.map_size,                             # 3
                velocity/hunter.max_vel,                                     # 3
                target_pos/self.map_size,                                  # 3
                np.array([distance_to_target])/(np.sqrt(2)*self.map_size), # 1
                laser_data/hunter.lidar.max_detect_d                                # num_lasers
            ]).astype(np.float32)

            h_obs.append(obs)

        # Precompute hunter positions for targets' observations
        for target in self.targets:
            # Get target's own position and velocity
            own_pos = target.position
            own_vel = target.velocity

            # Find three nearest hunters
            distances = np.linalg.norm(hunter_positions[:, :2] - own_pos[:2], axis=1)
            nearest_indices = distances.argsort()[:3]
            nearest_hunters = hunter_positions[nearest_indices]
            if len(nearest_hunters) < 3:
                # Pad with zeros if less than 3 hunters
                pad_size = 3 - len(nearest_hunters)
                nearest_hunters = np.vstack([nearest_hunters, np.zeros((pad_size, 3))])

            # Get laser data
            laser_data = target.lasers  # Assuming it's a 1D array of size num_lasers

            # Concatenate all observation components
            obs = np.concatenate([
                own_pos/self.map_size,                    # 3
                own_vel/target.max_vel,                     # 3
                nearest_hunters.flatten()/self.map_size,  # 3 * 3 = 9
                laser_data/target.lidar.max_detect_d               # num_lasers
            ]).astype(np.float32)

            t_obs.append(obs)

        return h_obs, t_obs

    def _compute_rewards(self):
        """
        Compute rewards for all agents and check for done conditions.
        Returns:
            rewards (list of float): Rewards for all agents (hunters followed by targets).
            dones (list of bool): Done flags for all agents.
        """
        rewards = [0.0] * (self.num_hunter + self.num_target)
        dones = [False] * (self.num_hunter + self.num_target)

        # Map each target to its assigned hunters
        target_hunter_groups = {}
        for target in self.targets:
            target_hunter_groups[target] = [hunter for hunter in self.hunters if hunter.assigned_target == target]

        # Reward for hunters chasing and capturing targets
        for target, hunters in target_hunter_groups.items():
            # calculate chasing reward and ifrounded reward
            for hunter in hunters:
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

            multi_hunters_pos = [h.position for h in hunters]
            if utils.isRounded(tuple(target.position[:2]), [tuple(row[:2]) for row in multi_hunters_pos], hunter.lidar.max_detect_d, self.max_escape_angle):
                for hunter in hunters:
                    hunter_index = self.hunters.index(hunter)
                    rewards[hunter_index] += self.capture_reward
                target_index = self.targets.index(target)
                dones[self.num_hunter + target_index] = True  # to mark target as done

        # Reward for targets
        for target in self.targets:
            target_index = self.targets.index(target)
            if dones[self.num_hunter + target_index]:
                rewards[self.num_hunter + target_index] += 0  # No additional reward if captured
                continue

            hunters = target_hunter_groups[target]
            if len(hunters) != 0:
                distances = [np.linalg.norm(target.position[:2] - hunter.position[:2]) for hunter in hunters]
                nearest_distance = min(distances)
            else:
                nearest_distance = np.inf

            if nearest_distance > self.escape_distance:
                escape_reward = 0.1
            else:
                escape_reward = -0.1
            rewards[self.num_hunter + target_index] += self.escape_reward_coeff * escape_reward

        # reward for safety (u-u)
        # between hunters
        for i in range(self.num_hunter):
            for j in range(i+1, self.num_hunter):
                distance = np.linalg.norm(self.hunters[i].position[:2] - self.hunters[j].position[:2])
                if distance < self.distance_threshold:
                    penalty = self.safe_penalty_coeff * (self.distance_threshold - distance)
                    rewards[i] -= penalty
                    rewards[j] -= penalty

        # between hunters and targets
        for hunter in self.hunters:
            for target in self.targets:
                distance = np.linalg.norm(hunter.position[:2] - target.position[:2])
                if distance < self.distance_threshold:
                    rewards[self.hunters.index(hunter)] -= self.safe_penalty_coeff * (self.distance_threshold - distance)
                    rewards[self.num_hunter + self.targets.index(target)] -= self.safe_penalty_coeff * (self.distance_threshold - distance)
        
        # between targets
        for i in range(self.num_target):
            for j in range(i+1, self.num_target):
                distance = np.linalg.norm(self.targets[i].position[:2] - self.targets[j].position[:2])
                if distance < self.distance_threshold:
                    penalty = self.safe_penalty_coeff * (self.distance_threshold - distance)
                    rewards[self.num_hunter + i] -= penalty
                    rewards[self.num_hunter + j] -= penalty

        # reward for safety (uav-obstacles)
        for agent in self.hunters + self.targets:
            min_laser_length = min(agent.lasers)
            collision_penalty = -self.safe_penalty_coeff * (agent.lidar.max_detect_d - min_laser_length) / agent.lidar.max_detect_d
            if agent in self.hunters:
                agent_index = self.hunters.index(agent)
            else:
                agent_index = self.targets.index(agent) + self.num_hunter
            rewards[agent_index] += collision_penalty

        return rewards, dones
    
    def rewardNorm(self, rewards):
        """
        normalize rewards for hunters and targets
        """
        h_rewards = rewards[:self.num_hunter]
        t_rewards = rewards[self.num_hunter:]
        
        h_mean = np.mean(h_rewards)
        h_std = np.std(h_rewards)
        if h_std == 0:
            h_normalized = h_rewards
        else:
            h_normalized = (h_rewards - h_mean) / h_std
        
        t_mean = np.mean(t_rewards)
        t_std = np.std(t_rewards)
        if t_std == 0:
            t_normalized = t_rewards
        else:
            t_normalized = (t_rewards - t_mean) / t_std
        
        normalized_rewards = list(h_normalized) + list(t_normalized)
        return normalized_rewards

    # TODO: use simulator like airsim to render
    def render(self):
        """
        Visualize the environment.
        """
        self.ax.clear()
        self.ax.set_xlim(0, self.map_size)
        self.ax.set_ylim(0, self.map_size)
        self.ax.set_zlim(0, self.map_size/4)
        self.ax.set_title("Environment Visualization")

        # Draw boundaries
        self.ax.plot([0, self.map_size, self.map_size, 0, 0], [0, 0, self.map_size, self.map_size, 0], [0, 0, 0, 0, 0], color='black', linewidth=2)
        # Draw obstacles (cylinders)
        for obstacle in self.obstacles:
            cx, cy, cz, r, h = obstacle._return_obs_info()
            self._create_cylinders(self.ax, cx, cy, cz, r, h)

        # Draw hunters
        for hunter in self.hunters:
            x, y, z = hunter.position
            self.ax.scatter(x, y, z, color='red', label='Hunter' if hunter == self.hunters[0] else "")
            if self.visualize_lasers:
                # Draw lasers
                hunter.lidar.visualize_lasers(hunter.position,self.ax)

        # Draw targets
        for target in self.targets:
            x, y, z = target.position
            self.ax.scatter(x, y, z, color='green', label='Target' if target == self.targets[0] else "")

        self.ax.legend(loc='upper right')
        plt.pause(0.001)

    def close(self):
        plt.close(self.fig)


    def _create_cylinders(self, ax, x, y, z, r, h):
        """
        Create a 3D cylinder to represent obstacles in the environment.
        """
        # Create obstacle as a cylinder in 3D
        # reduce the number of sampling points in the circumferential and height directions 
        # to improve rendering performance
        theta = np.linspace(0, 2 * np.pi, 16) 
        z_vals = np.linspace(z, z + h, 8)   
        theta, z_vals = np.meshgrid(theta, z_vals)
        x_vals = x + r * np.cos(theta)
        y_vals = y + r * np.sin(theta)

        ax.plot_surface(x_vals, y_vals, z_vals, color='black', alpha=0.5)

class Obstacle:
    def __init__(self, map_size, margin, 
                 radius_range, height_range, 
                 speed=0):
        self.map_size = map_size
        self.valid_area = [margin * map_size, (1 - margin) * map_size]
        
        self.speed = speed
        self.radius_range = radius_range
        self.height_range = height_range

        self.reset()
    
    def reset(self):
        self.position = np.random.uniform(low=self.valid_area[0], 
                                          high=self.valid_area[1], 
                                          size=(3,))  # TODO: initial spawn scope
        
        self.position[-1] = 0 # firstly let z = 0
        angle = np.random.uniform(0, 2 * np.pi)
        self.velocity = np.array([self.speed * np.cos(angle), self.speed * np.sin(angle)])
        self.radius = np.random.uniform(*self.radius_range)
        self.height = np.random.uniform(*self.height_range)
    
    def move(self):
        #TODO
        pass

    def _return_obs_info(self):
        x,y,z = self.position
        r = self.radius
        h = self.height
        return (x,y,z,r,h)

class AgentBase:
    def __init__(self, lidar: Lidar, 
                 map_size, margin, init_height,
                 max_vel, max_acc,
                 time_step=0.5):
        self.map_size = map_size
        self.valid_area = [margin * map_size, (1 - margin) * map_size]
        self.init_height = init_height

        self.max_vel = max_vel
        self.max_acc = max_acc
        
        self.time_step = time_step  # update time step

        self.lidar = lidar
        self.reset()
        
    def reset(self):
        self.position = np.random.uniform(low=self.valid_area[0], 
                                          high=self.valid_area[1], 
                                          size=(3,))  # TODO: initial spawn scope
        
        self.position[-1] = self.init_height  # initial height
        self.velocity = np.zeros(3)  # initial velocity

        self.update_lidar()
        self.lidar.distances = self.lasers
        self.history_pos = []  # to store trajectory

    # update state
    def move(self, action, v_max=None):
        v_max = self.max_vel if v_max is None else v_max

        ax, ay = action
        if self.lidar.isInObs:
            self.velocity = np.zeros(3)
        else:
            self.velocity[0] += self.time_step * ax
            self.velocity[1] += self.time_step * ay
            velocity_norm = (self.velocity[0]**2 + self.velocity[1]**2)**0.5
            if velocity_norm > self.max_vel:
                scale_factor = self.max_vel / velocity_norm
                self.velocity[0] *= scale_factor
                self.velocity[1] *= scale_factor

        self.position += self.time_step * self.velocity 
        self.update_lidar()
        self.history_pos.append(self.position.copy())
    
    def update_lidar(self):
        self.lasers = self.lidar.scan(self.position, self.map_size)

class Hunter(AgentBase):
    def __init__(self, lidar: Lidar, 
                 map_size, margin, init_height,
                 max_vel, max_acc,
                 time_step=0.5):
        super().__init__(lidar, map_size, margin, init_height, max_vel, max_acc, time_step)

        self.role = {'0':'chaser', '1':'predator'}

class Target(AgentBase):
    def __init__(self, lidar: Lidar, 
                 map_size, margin, init_height,
                 max_vel, max_acc,
                 time_step=0.5):
        super().__init__(lidar, map_size, margin, init_height, max_vel, max_acc, time_step)

