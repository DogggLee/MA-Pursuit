import numpy as np

class Lidar:
    def __init__(self, max_detect_d, num_rays, obstacles):
        """
        :param max_detect_d: max laser length
        :param num_rays: number of rays
        :param obstacles: list of obstacles in form of (x, y, z, r, h) 
        """
        self.max_detect_d = max_detect_d
        self.num_rays = num_rays
        self.obstacles = obstacles # global obstacle information
        self.angles = np.linspace(0, 2 * np.pi, num_rays)
        self.distances = np.full(num_rays, max_detect_d) # initial lasers' lengths
        self.isInObs = False # whether the agent is inside the obstacle

    def _calculate_intersections(self,position,map_size):
        """
        calculate intersections and update laser's length
        """
        self.isInObs = False
        x0, y0, z0 = position

        # 在地图范围外，视作处于障碍物内部
        if x0 < 0 or x0 > map_size or y0 < 0 or y0 > map_size:
            self.distances = np.zeros(self.num_rays)
            self.isInObs = True
            return
        
        # local obstacle information
        self.obstacles_near = self._detect_obs_near(self.obstacles,position,self.max_detect_d)

        # 判断是否处于障碍物内部
        for cylinder in self.obstacles_near:
            cx, cy, cz, r, h = cylinder._return_obs_info()
            distance_to_center = np.sqrt((x0 - cx)**2 + (y0 - cy)**2)
            # if inside cylinder
            if distance_to_center < r and cz <= z0 <= cz + h:
                self.distances = np.zeros(self.num_rays)
                self.isInObs = True
                return

        # 每条激光射线计算与障碍物的交点
        for i, angle in enumerate(self.angles):
            for cylinder in self.obstacles:
                cx, cy, cz, r, h = cylinder._return_obs_info()
                dx, dy = np.cos(angle), np.sin(angle)
                a = dx**2 + dy**2
                b = 2 * (dx * (x0 - cx) + dy * (y0 - cy))
                c = (x0 - cx)**2 + (y0 - cy)**2 - r**2
                discriminant = b**2 - 4 * a * c
                if discriminant >= 0:
                    t1 = (-b - np.sqrt(discriminant)) / (2 * a)
                    t2 = (-b + np.sqrt(discriminant)) / (2 * a)
                    valid_t = [t for t in [t1, t2] if t > 0]
                    if valid_t:
                        t = min(valid_t)
                        intersection_z = z0 + t * 0  
                        if cz <= intersection_z <= cz + h: 
                            distance = t
                            if distance < self.distances[i]:
                                self.distances[i] = distance
                
                t_boundary = self._calculate_boundary_intersection(x0, y0, dx, dy, map_size)
                if t_boundary is not None and t_boundary < self.distances[i]:
                    self.distances[i] = t_boundary

    def _calculate_boundary_intersection(self, x0, y0, dx, dy, length):
        t_values = []

        if dx < 0:
            t = -x0 / dx
            y_intersect = y0 + t * dy
            if 0 <= y_intersect <= length:
                t_values.append(t)
        if dx > 0:
            t = (length - x0) / dx
            y_intersect = y0 + t * dy
            if 0 <= y_intersect <= length:
                t_values.append(t)
        if dy < 0:
            t = -y0 / dy
            x_intersect = x0 + t * dx
            if 0 <= x_intersect <= length:
                t_values.append(t)
        if dy > 0:
            t = (length - y0) / dy
            x_intersect = x0 + t * dx
            if 0 <= x_intersect <= length:
                t_values.append(t)

        if t_values:
            return min(t_values)
        else:
            return None

    def scan(self, position, length):
        """
        scan function
        """
        self.distances = np.full(self.num_rays, self.max_detect_d)
        self._calculate_intersections(position, length)
        return self.distances

    # detect obstacles nearby, return a local set
    def _detect_obs_near(self, obstacles, position, max_distance):
        if obstacles is None:
            return []

        nearby_obstacles = []
        for obstacle in obstacles:
            cx, cy, cz, r, h = obstacle._return_obs_info()
            x0, y0, z0 = position
            distance = np.sqrt((x0 - cx)**2 + (y0 - cy)**2)
            if distance - r <= max_distance and cz <= z0 <= cz + h: 
                nearby_obstacles.append(obstacle)
        return nearby_obstacles

    def visualize_lasers_ori(self,position,ax):
        """
        Visualize the laser rays.
        """
        x0, y0, z0 = position
        x_end = x0 + self.distances * np.cos(self.angles)
        y_end = y0 + self.distances * np.sin(self.angles)

        # Plot laser rays
        for x, y, d in zip(x_end, y_end, self.distances):
            ax.plot([x0, x], [y0, y], [z0, z0], color='green', alpha=0.6)

    def visualize_lasers(self, position, ax, label=False, fill=False):
        """
        Visualize the laser rays and scan range contour.
        :param position: (x0, y0, z0) 智能体位置
        :param ax: 3D绘图轴对象
        """
        x0, y0, z0 = position
        # 计算每条激光的终点坐标
        x_end = x0 + self.distances * np.cos(self.angles)
        y_end = y0 + self.distances * np.sin(self.angles)

        # 1. 绘制原始激光射线（保留原有逻辑）
        # for i, (x, y, d) in enumerate(zip(x_end, y_end, self.distances)):
        #     ax.plot([x0, x], [y0, y], [z0, z0], color='blue', alpha=0.1, label='Laser Ray' if label and (i == 0) else "")

        # 2. 新增：绘制扫描范围轮廓（闭合多边形）
        # 拼接起点和终点，形成闭合轮廓
        contour_x = np.concatenate([x_end, [x_end[0]]])
        contour_y = np.concatenate([y_end, [y_end[0]]])
        contour_z = np.full_like(contour_x, z0)
        # 绘制轮廓线（红色加粗）
        ax.plot(contour_x, contour_y, contour_z, color='blue', linewidth=2, alpha=0.3, label='Scan Range Contour' if label else "")

        if fill:
            # TODO: 填充扫描范围（半透明）
            pass
            # ax.fill_between(contour_x[:-1], contour_y[:-1], zs=z0, alpha=0.2, color='red', label='Scan Area' if label else "")

        # 设置图例（避免重复）
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')


    def _create_cylinders(self, ax):
        """
        plot cylinder in env.
        """
        for cylinder in self.obstacles:
            x, y, z, r, h = cylinder
            theta = np.linspace(0, 2 * np.pi, 10)
            z_vals = np.linspace(z, z + h, 10)
            theta, z_vals = np.meshgrid(theta, z_vals)
            x_vals = x + r * np.cos(theta)
            y_vals = y + r * np.sin(theta)
            ax.plot_surface(x_vals, y_vals, z_vals, color='blue', alpha=0.5)
