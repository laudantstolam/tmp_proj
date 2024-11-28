#!/usr/bin/env python3
import rospy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from geometry_msgs.msg import Twist, PointStamped
from sensor_msgs.msg import PointCloud2, Imu
from gazebo_msgs.srv import SetModelState, GetModelState
from gazebo_msgs.msg import ModelState, ContactsState
import sensor_msgs.point_cloud2 as pc2
from scipy.special import comb
from collections import namedtuple
import cv2
import open3d as o3d
import tf
from tf.transformations import quaternion_from_euler
import time
from torch.amp import GradScaler
import yaml
from PIL import Image
import matplotlib.pyplot as plt

# 超參數
REFERENCE_DISTANCE_TOLERANCE = 0.65
MEMORY_SIZE = 10000
BATCH_SIZE = 256
GAMMA = 0.99
LEARNING_RATE = 0.0003
PPO_EPOCHS = 5
CLIP_PARAM = 0.2
PREDICTION_HORIZON = 400
CONTROL_HORIZON = 10

device = torch.device("cpu")
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class PrioritizedMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = [None] * capacity
        self.position = 0
        self.priorities = torch.zeros((capacity,), dtype=torch.float32).cuda()
        self.alpha = 0.6
        self.epsilon = 1e-5

    def add(self, state, action, reward, done, next_state):
        if state is None or action is None or reward is None or done is None or next_state is None:
            rospy.logwarn("Warning: Attempted to add None to memory, skipping entry.")
            return

        max_priority = self.priorities.max() if self.memory[self.position] is not None else torch.tensor(1.0, device=device)
        self.memory[self.position] = (
            torch.tensor(state, dtype=torch.float32, device=device),
            torch.tensor(action, dtype=torch.float32, device=device),
            torch.tensor(reward, dtype=torch.float32, device=device),
            torch.tensor(done, dtype=torch.float32, device=device),
            torch.tensor(next_state, dtype=torch.float32, device=device)
        )
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if self.position == 0:
            raise ValueError("No samples available in memory.")

        # Ensure all priorities are valid for sampling
        if len(self.memory) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]

        # Handle NaN in priorities
        if torch.isnan(priorities).any():
            priorities = torch.nan_to_num(priorities, nan=0.0)

        probabilities = priorities ** self.alpha
        total = probabilities.sum()

        if total > 0:
            probabilities /= total
        else:
            probabilities = torch.ones_like(probabilities) / len(probabilities)

        indices = torch.multinomial(probabilities, batch_size, replacement=False).cuda()
        samples = [self.memory[idx] for idx in indices if self.memory[idx] is not None]

        if len(samples) == 0 or any(sample is None for sample in samples):
            raise ValueError("Sampled None from memory.")

        weights = (len(self.memory) * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        batch = list(zip(*samples))
        states, actions, rewards, dones, next_states = batch

        # Ensure all states are 4D tensors
        states = [s if s.dim() == 4 else s.view(1, 3, 64, 64) for s in states]
        next_states = [ns if ns.dim() == 4 else ns.view(1, 3, 64, 64) for ns in next_states]

        return (
            torch.stack(states).to(device),
            torch.stack(actions).to(device),
            torch.stack(rewards).to(device),
            torch.stack(dones).to(device),
            torch.stack(next_states).to(device),
            indices,
            weights.to(device)
        )

    def update_priorities(self, batch_indices, batch_priorities):
        # 確保每個 priority 是單一標量
        for idx, priority in zip(batch_indices, batch_priorities):
            # 如果 priority 是 numpy 陣列，檢查其 size
            if priority.size > 1:
                priority = priority[0]
            self.priorities[idx] = priority.item() + self.epsilon

    def clear(self):
        self.position = 0
        self.memory = [None] * self.capacity
        self.priorities = torch.zeros((self.capacity,), dtype=torch.float32).cuda()

class GazeboEnv:
    def __init__(self, model):
        rospy.init_node('gazebo_rl_agent', anonymous=True)
        self.model = model
        self.pub_cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.pub_imu = rospy.Publisher('/imu/data', Imu, queue_size=10)
        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.listener = tf.TransformListener()
        self.action_space = 2
        self.observation_space = (3, 64, 64)
        self.state = np.zeros(self.observation_space)
        self.done = False
        self.target_x = -7.2213
        self.target_y = -1.7003  
        # self.initialize_path_planning()
        # self.waypoints = self.generate_random_waypoints()
        # self.waypoint_distances = self.calculate_waypoint_distances()   # 計算一整圈機器任要奏的大致距離
        self.current_waypoint_index = 0
        self.last_twist = Twist()
        self.epsilon = 0.05
        self.collision_detected = False
        self.previous_robot_position = None  # 初始化 previous_robot_position 為 None
        self.previous_distance_to_goal = None  # 初始化 previous_distance_to_goal 為 None

        self.max_no_progress_steps = 10
        self.no_progress_steps = 0
        
        self.optimized_segments = []
        self.base_path = None
        self.path_segments = []

        # map相關
        # self.waypoint_failures = {i: 0 for i in range(len(self.waypoints))}
        self.robot_radius = 0.1
        self.safety_margin = 0.2

        # self.initialize_path_planning()
    
    def generate_costmap(self):
        """生成基于颜色的代价地图"""
        if self.slam_map is None:
            rospy.logerr("SLAM map not loaded. Cannot generate costmap.")
            return False

        # 假设墙壁颜色是白色（你可以根据实际地图修改这个颜色）
        wall_color = np.array([100, 100, 100])  # 例如，墙壁是白色的
        wall_color2 = np.array([120, 120, 120])  # 例如，墙壁是白色的

        # 将地图转换为 RGB 色彩模式（如果它是灰度图）
        img = cv2.cvtColor(self.slam_map, cv2.COLOR_GRAY2BGR) 

        # 创建掩码：找出指定颜色的墙壁（使用 cv2.inRange 对墙壁颜色进行提取）
        wall_mask = cv2.inRange(img, wall_color, wall_color2)

        # 初始化代价地图
        self.cost_map = np.zeros_like(self.slam_map)

        # 设定膨胀的内外层大小
        inner_dilation = 7  # 内层膨胀的大小
        outer_dilation = 15  # 外层膨胀的大小

        # 内层膨胀 (膨胀操作将墙壁区域扩展)
        inner_kernel = np.ones((inner_dilation * 2 + 1, inner_dilation * 2 + 1), np.uint8)
        inner_dilated = cv2.dilate(wall_mask, inner_kernel, iterations=1)

        # 外层膨胀
        outer_kernel = np.ones((outer_dilation * 2 + 1, outer_dilation * 2 + 1), np.uint8)
        outer_dilated = cv2.dilate(wall_mask, outer_kernel, iterations=1)

        # 计算外层区域（去掉内层部分）
        outer_only = cv2.subtract(outer_dilated, inner_dilated)

        # 内层区域设置为高代价
        self.cost_map[inner_dilated > 0] = 255  # 高代价区域

        # 外层区域设置为低代价
        self.cost_map[outer_only > 0] = 100  # 低代价区域

        # 墙壁区域为障碍物
        self.cost_map[wall_mask > 0] = 50  # 障碍物

        rospy.loginfo("Costmap generated successfully.")
        return True

    
    def load_slam_map(self, yaml_path):
        """读取并加载SLAM地图"""
        # 读取YAML文件
        with open(yaml_path, 'r') as file:
            map_metadata = yaml.safe_load(file)
            
        self.map_origin = map_metadata['origin']  # 地图原点
        self.map_resolution = map_metadata['resolution']  # 地图解析度
        png_path = map_metadata['image'].replace(".pgm", ".png")  # 修改为png文件路径
        
        # 使用PIL读取PNG文件
        png_image = Image.open(png_path).convert('L')
        self.slam_map = np.array(png_image)  # 转为NumPy数组

    def generate_random_waypoints(self):
        # 預定義的 waypoints
        all_waypoints = [
            (-6.4981, -1.0627),
            (-5.4541, -1.0117),
            (-4.4041, -0.862),
            (-3.3692, -1.0294),
            (-2.295, -1.114),
            (-1.2472, -1.0318),
            (-0.1614, -0.6948),
            (0.8931, -0.8804),
            (1.9412, -0.8604),
            (2.9804, -0.7229),
            (3.874, -0.2681),
            (4.9283, -0.1644),
            (5.9876, -0.345),
            (7.019, -0.5218),
            (7.9967, -0.2338),
            (9.0833, -0.1096),
            (10.1187, -0.3335),
            (11.1745, -0.6322),
            (12.1693, -0.8619),
            (13.1291, -0.4148),
            (14.1217, -0.0282),
            (15.1261, 0.123),
            (16.1313, 0.4439),
            (17.1389, 0.696),
            (18.1388, 0.6685),
            (19.2632, 0.5127),
            (20.2774, 0.2655),
            (21.2968, 0.0303),
            (22.3133, -0.0192),
            (23.2468, 0.446),
            (24.1412, 0.9065),
            (25.1178, 0.5027),
            (26.1279, 0.4794),
            (27.0867, 0.8266),
            (28.0713, 1.4229),
            (29.1537, 1.3866),
            (30.2492, 1.1549),
            (31.385, 1.0995),
            (32.4137, 1.243),
            (33.4134, 1.5432),
            (34.4137, 1.5904),
            (35.4936, 1.5904),
            (36.5067, 1.5607),
            (37.5432, 1.5505),
            (38.584, 1.7008),
            (39.6134, 1.9053),
            (40.5979, 2.0912),
            (41.6557, 2.3779),
            (42.5711, 2.8643),
            (43.5911, 2.9725),
            (44.5929, 3.0637),
            (45.5919, 2.9841),
            (46.6219, 2.9569),
            (47.6314, 3.0027),
            (48.7359, 2.832),
            (49.5462, 2.1761),
            (50.5982, 2.1709),
            (51.616, 2.3573),
            (52.6663, 2.5593),
            (53.7532, 2.5325),
            (54.7851, 2.5474),
            (55.8182, 2.5174),
            (56.8358, 2.6713),
            (57.8557, 2.8815),
            (58.8912, 3.0949),
            (59.7436, 3.6285),
            (60.5865, 4.2367),
            (60.6504, 5.2876),
            (60.7991, 6.3874),
            (60.322, 7.3094),
            (59.8004, 8.1976),
            (59.4093, 9.195),
            (59.1417, 10.1994),
            (59.1449, 11.2274),
            (59.5323, 12.2182),
            (59.8637, 13.2405),
            (60.5688, 14.0568),
            (60.6266, 15.1571),
            (60.007, 15.9558),
            (59.0539, 16.5128),
            (57.9671, 16.526),
            (56.9161, 16.7399),
            (55.9553, 17.0346),
            (54.9404, 17.0596),
            (53.9559, 16.8278),
            (52.9408, 16.8697),
            (51.9147, 16.7642),
            (50.9449, 16.4902),
            (49.9175, 16.3029),
            (48.8903, 16.1165),
            (47.7762, 16.0994),
            (46.7442, 16.0733),
            (45.7566, 15.8195),
            (44.756, 15.7218),
            (43.7254, 15.9309),
            (42.6292, 15.8439),
            (41.6163, 15.8177),
            (40.5832, 15.7881),
            (39.5617, 15.773),
            (38.5099, 15.5648),
            (37.692, 14.9481),
            (36.8538, 14.3078),
            (35.8906, 13.8384),
            (34.8551, 13.6316),
            (33.8205, 13.5495),
            (32.7391, 13.4423),
            (31.7035, 13.1056),
            (30.6971, 12.7802),
            (29.6914, 12.5216),
            (28.7072, 12.3238),
            (27.6442, 12.0953),
            (26.5991, 11.9873),
            (25.5713, 11.9867),
            (24.488, 12.0679),
            (23.4441, 12.0246),
            (22.3169, 11.7745),
            (21.3221, 11.538),
            (20.3265, 11.4243),
            (19.2855, 11.5028),
            (18.2164, 11.5491),
            (17.1238, 11.6235),
            (16.0574, 11.4029),
            (14.982, 11.2479),
            (13.9491, 11.0487),
            (12.9017, 11.1455),
            (11.8915, 11.4186),
            (10.8461, 11.6079),
            (9.9029, 12.0097),
            (9.0549, 12.5765),
            (8.4289, 13.4238),
            (7.4035, 13.6627),
            (6.3785, 13.5659),
            (5.3735, 13.4815),
            (4.3971, 13.1044),
            (3.3853, 13.2918),
            (2.3331, 13.0208),
            (1.2304, 12.9829),
            (0.2242, 13.094),
            (-0.807, 12.9358),
            (-1.8081, 12.8495),
            (-2.7738, 13.3168),
            (-3.4822, 14.0699),
            (-4.5285, 14.2483),
            (-5.5965, 13.9753),
            (-6.5324, 13.6016),
            (-7.3092, 12.8632),
            (-8.3255, 12.9916),
            (-9.1914, 13.7593),
            (-10.2374, 14.069),
            (-11.2162, 13.7566),
            (-11.653, 12.8061),
            (-11.6989, 11.7238),
            (-11.8899, 10.7353),
            (-12.6174, 10.0373),
            (-12.7701, 8.9551),
            (-12.4859, 7.9523),
            (-12.153, 6.8903),
            (-12.4712, 5.819),
            (-13.0498, 4.8729),
            (-13.1676, 3.8605),
            (-12.4328, 3.1822),
            (-12.1159, 2.1018),
            (-12.8436, 1.2659),
            (-13.3701, 0.2175),
            (-13.0514, -0.8866),
            (-12.3046, -1.619),
            (-11.2799, -1.472),
            (-10.1229, -1.3051),
            (-9.1283, -1.4767),
            (-8.1332, -1.2563),
            (self.target_x, self.target_y)
        ]
        
        # 設定要採樣的點數（可以調整）
        num_samples = 12
        
        # 確保包含起點和終點
        sampled_waypoints = [all_waypoints[0]]  # 起點
        
        # 隨機採樣中間點
        indices = np.linspace(1, len(all_waypoints)-2, num_samples-2, dtype=int)
        indices = indices + np.random.randint(-3, 4, size=len(indices))  # 添加隨機偏移
        indices = np.clip(indices, 1, len(all_waypoints)-2)  # 確保索引在有效範圍內
        
        for idx in indices:
            sampled_waypoints.append(all_waypoints[idx])
        
        sampled_waypoints.append(all_waypoints[-1])  # 終點
        
        return sampled_waypoints
    
    def calculate_waypoint_distances(self):
        """
        計算每對相鄰 waypoint 之間的距離，並返回一個距離列表。
        """
        distances = []
        for i in range(len(self.waypoints) - 1):
            start_wp = self.waypoints[i]
            next_wp = self.waypoints[i + 1]
            distance = np.linalg.norm([next_wp[0] - start_wp[0], next_wp[1] - start_wp[1]])
            distances.append(distance)
        return distances

    def gazebo_to_image_coords(self, gazebo_x, gazebo_y):
        img_x = 2000 + gazebo_x * 20
        img_y = 2000 - gazebo_y * 20
        return int(img_x), int(img_y)

    def image_to_gazebo_coords(self, img_x, img_y):
        gazebo_x = (img_x - 2000) / 20
        gazebo_y = (2000 - img_y) / 20
        return gazebo_x, gazebo_y

    def bezier_curve(self, waypoints, n_points=100):
        waypoints = np.array(waypoints)
        n = len(waypoints) - 1

        def bernstein_poly(i, n, t):
            return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))

        t = np.linspace(0.0, 1.0, n_points)
        curve = np.zeros((n_points, 2))

        for i in range(n + 1):
            curve += np.outer(bernstein_poly(i, n, t), waypoints[i])

        return curve

    def collision_callback(self, data):
        if len(data.states) > 0:
            self.collision_detected = True
            rospy.loginfo("Collision detected!")
        else:
            self.collision_detected = False

    def generate_imu_data(self):
        imu_data = Imu()
        imu_data.header.stamp = rospy.Time.now()
        imu_data.header.frame_id = 'chassis'

        imu_data.linear_acceleration.x = np.random.normal(0, 0.1)
        imu_data.linear_acceleration.y = np.random.normal(0, 0.1)
        imu_data.linear_acceleration.z = np.random.normal(9.81, 0.1)
        
        imu_data.angular_velocity.x = np.random.normal(0, 0.01)
        imu_data.angular_velocity.y = np.random.normal(0, 0.01)
        imu_data.angular_velocity.z = np.random.normal(0, 0.01)

        robot_x, robot_y, robot_yaw = self.get_robot_position()
        quaternion = quaternion_from_euler(0.0, 0.0, robot_yaw)
        imu_data.orientation.x = quaternion[0]
        imu_data.orientation.y = quaternion[1]
        imu_data.orientation.z = quaternion[2]
        imu_data.orientation.w = quaternion[3]

        return imu_data

    def is_valid_data(self, data):
        for point in pc2.read_points(data, field_names=("x", "y", "z"), skip_nans=True):
            if point[0] != 0.0 or point[1] != 0.0 or point[2] != 0.0:
                return True
        return False

    def transform_point(self, point, from_frame, to_frame):
        try:
            now = rospy.Time.now()
            self.listener.waitForTransform(to_frame, from_frame, now, rospy.Duration(1.0))
            
            point_stamped = PointStamped()
            point_stamped.header.frame_id = from_frame
            point_stamped.header.stamp = now
            point_stamped.point.x = point[0]
            point_stamped.point.y = point[1]
            point_stamped.point.z = point[2]
            
            point_transformed = self.listener.transformPoint(to_frame, point_stamped)
            return [point_transformed.point.x, point_transformed.point.y, point_transformed.point.z]
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logerr(f"Transform failed: {e}")
            return [point[0], point[1], point[2]]

    def convert_open3d_to_ros(self, cloud):
        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'velodyne'
        points = np.asarray(cloud.points)
        return pc2.create_cloud_xyz32(header, points)

    def generate_occupancy_grid(self, robot_x, robot_y, grid_size=0.05, map_size=100):
        # 將機器人的座標轉換為地圖上的像素座標
        img_x, img_y = self.gazebo_to_image_coords(robot_x, robot_y)
        
        # 計算64x64網格在圖片上的起始和結束索引
        half_grid = 32  # 因為需要64x64的矩陣，所以邊長的一半是32
        start_x = max(0, img_x - half_grid)
        start_y = max(0, img_y - half_grid)
        end_x = min(self.slam_map.shape[1], img_x + half_grid)
        end_y = min(self.slam_map.shape[0], img_y + half_grid)

        # 擷取圖片中的64x64區域
        grid = np.zeros((64, 64), dtype=np.float32)
        grid_slice = self.slam_map[start_y:end_y, start_x:end_x]
        
        # 填充grid，將超出地圖範圍的部分填充為0
        grid[:grid_slice.shape[0], :grid_slice.shape[1]] = grid_slice

        # 將當前機器人位置資訊添加到occupancy grid
        occupancy_grid = np.zeros((3, 64, 64), dtype=np.float32)
        occupancy_grid[0, :, :] = grid
        occupancy_grid[1, :, :] = robot_x  # 機器人的x位置
        occupancy_grid[2, :, :] = robot_y  # 機器人的y位置

        return occupancy_grid

    def step(self, action):
        reward = 0
        robot_x, robot_y, robot_yaw = self.get_robot_position()
        self.state = self.generate_occupancy_grid(robot_x, robot_y)

        # 添加可视化占用网格
        if self.state is not None:
            # 如果是 NumPy 数组，直接处理
            if isinstance(self.state, np.ndarray):
                grid = self.state[0, :, :]  # 提取第一通道为灰度图
            elif isinstance(self.state, torch.Tensor):
                grid = self.state[0].cpu().numpy()[0, :, :]  # 转换为 NumPy 格式并提取第一通道
            
            # 将灰度图数据归一化到 0-255 范围
            grid_normalized = (grid / grid.max() * 255).astype(np.uint8)
            grid_color = cv2.cvtColor(grid_normalized, cv2.COLOR_GRAY2BGR)
            # 在中心绘制机器人位置
            cv2.circle(grid_color, (32, 32), 3, (0, 0, 255), -1)
            # 显示网格
            cv2.imshow("Occupancy Grid", grid_color)
            cv2.waitKey(1)  # 不阻塞程序

        # 计算当前机器人位置与所有 waypoints 的距离，并找到距离最近的 waypoint 的索引
        distances = [np.linalg.norm([robot_x - wp_x, robot_y - wp_y]) for wp_x, wp_y in self.waypoints]
        closest_index = np.argmin(distances)
        if closest_index > self.current_waypoint_index:
            distance_reward = sum(self.waypoint_distances[self.current_waypoint_index:closest_index])
            reward += distance_reward * 100
            self.current_waypoint_index = closest_index

        # 确认最近的 waypoint 是否与目标位置相同
        current_waypoint_x, current_waypoint_y = self.waypoints[self.current_waypoint_index]

        distance_moved = np.linalg.norm([robot_x - self.previous_robot_position[0], robot_y - self.previous_robot_position[1]]) if self.previous_robot_position else 0
        self.previous_robot_position = (robot_x, robot_y)

        use_deep_rl_control = any(
            self.waypoint_failures.get(i, 0) > 1
            for i in range(max(0, self.current_waypoint_index - 3), min(len(self.waypoints), self.current_waypoint_index + 4))
        )

        if use_deep_rl_control:
            print('operate by RL')
            if isinstance(self.state, np.ndarray):
                self.state = torch.tensor(self.state, dtype=torch.float32).view(1, 3, 64, 64).to(device)
            elif self.state.dim() != 4:
                self.state = self.state.view(1, 3, 64, 64)

            action = self.model.act(self.state)
            print("RL Action output:", action)
            linear_speed = np.clip(action[0, 0].item(), -2.0, 2.0)
            steer_angle = np.clip(action[0, 1].item(), -0.6, 0.6)

            if distance_moved < 0.05:
                self.no_progress_steps += 1
                if self.no_progress_steps >= self.max_no_progress_steps:
                    print('failure at point', self.current_waypoint_index)
                    rospy.loginfo("No progress detected, resetting environment.")
                    reward = -2000.0
                    self.reset()
                    return self.state, reward, True, {}
            else:
                self.no_progress_steps = 0

        else:
            # 计算当前位置与 current_waypoint_index 的距离
            robot_x, robot_y, _ = self.get_robot_position()
            current_waypoint_x, current_waypoint_y = self.waypoints[self.current_waypoint_index]
            distance_to_waypoint = np.linalg.norm([robot_x - current_waypoint_x, robot_y - current_waypoint_y])

            if distance_moved < 0.03:
                self.no_progress_steps += 1
                if self.no_progress_steps >= self.max_no_progress_steps:
                    self.waypoint_failures[self.current_waypoint_index] += 1
                    print('failure at point', self.current_waypoint_index)
                    rospy.loginfo("No progress detected, resetting environment.")
                    reward = -2000.0
                    self.reset()
                    return self.state, reward, True, {}
            else:
                self.no_progress_steps = 0

            action = self.calculate_action_pure_pursuit()
            linear_speed = np.clip(action[0], -2.0, 3.0)
            steer_angle = np.clip(action[1], -0.6, 0.6)

        twist = Twist()
        twist.linear.x = linear_speed
        twist.angular.z = steer_angle
        self.pub_cmd_vel.publish(twist)

        imu_data = self.generate_imu_data()
        self.pub_imu.publish(imu_data)

        rospy.sleep(0.1)

        reward, _ = self.calculate_reward(robot_x, robot_y, reward, self.state)

        return self.state, reward, self.done, {}

    def reset(self):

        robot_x, robot_y,_ = self.get_robot_position()
        self.state = self.generate_occupancy_grid(robot_x, robot_y)

        # 設置初始機器人位置和姿態
        yaw = -0.0053
        quaternion = quaternion_from_euler(0.0, 0.0, yaw)
        state_msg = ModelState()
        state_msg.model_name = 'my_robot'
        state_msg.pose.position.x = 0.2206
        state_msg.pose.position.y = 0.1208
        state_msg.pose.position.z = 2.2
        state_msg.pose.orientation.x = quaternion[0]
        state_msg.pose.orientation.y = quaternion[1]
        state_msg.pose.orientation.z = quaternion[2]
        state_msg.pose.orientation.w = quaternion[3]

        state_msg.twist.linear.x = 0.0
        state_msg.twist.linear.y = 0.0
        state_msg.twist.linear.z = 0.0
        state_msg.twist.angular.x = 0.0
        state_msg.twist.angular.y = 0.0
        state_msg.twist.angular.z = 0.0

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            self.set_model_state(state_msg)
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

        rospy.sleep(0.5)

        # 確保使用優化過的路徑點
        if hasattr(self, 'waypoints') and self.waypoints:
            rospy.loginfo("Using optimized waypoints for reset.")
        else:
            rospy.loginfo("No optimized waypoints found, generating new waypoints.")
            self.waypoints = self.generate_waypoints()

        self.current_waypoint_index = 0
        self.done = False

        self.last_twist = Twist()
        self.pub_cmd_vel.publish(self.last_twist)

        imu_data = self.generate_imu_data()
        self.pub_imu.publish(imu_data)

        self.previous_yaw_error = 0
        self.no_progress_steps = 0
        self.previous_distance_to_goal = None
        self.collision_detected = False

        # Ensure the state is 4D tensor
        if isinstance(self.state, np.ndarray):
            self.state = torch.tensor(self.state, dtype=torch.float32).view(1, 3, 64, 64).to(device)
        elif self.state.dim() != 4:
            self.state = self.state.view(1, 3, 64, 64)

        return self.state

    def calculate_reward(self, robot_x, robot_y, reward, state):
        done = False
        # 將機器人的座標轉換為地圖上的坐標

        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()

        if state.ndim == 4:
            # 对于 4 维情况，一个批次数据中的第一层
            occupancy_grid = state[0, 0]
        elif state.ndim == 3:
            # 对于 3 维情况，直接取第一层
            occupancy_grid = state[0]

        img_x, img_y = self.gazebo_to_image_coords(robot_x, robot_y)
        obstacle_count = np.sum(occupancy_grid <= 190)  # 假設state[0]為佔據網格通道
        # print('obstacle_count',obstacle_count)
        reward += 300 - obstacle_count*3

        return reward, done

    def get_robot_position(self):
        try:
            rospy.wait_for_service('/gazebo/get_model_state')
            model_state = self.get_model_state('my_robot', '')
            robot_x = model_state.pose.position.x
            robot_y = model_state.pose.position.y

            orientation_q = model_state.pose.orientation
            yaw = self.quaternion_to_yaw(orientation_q)
            return robot_x, robot_y, yaw
        except rospy.ServiceException as e:
            rospy.logerr(f"Get model state service call failed: %s", e)
            return 0, 0, 0

    def quaternion_to_yaw(self, orientation_q):
        siny_cosp = 2 * (orientation_q.w * orientation_q.z + orientation_q.x * orientation_q.y)
        cosy_cosp = 1 - 2 * (orientation_q.y * orientation_q.y + orientation_q.z * orientation_q.z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return yaw

    def calculate_action_pure_pursuit(self):
        robot_x, robot_y, robot_yaw = self.get_robot_position()

        # 動態調整前視距離（lookahead distance）
        linear_speed = np.linalg.norm([self.last_twist.linear.x, self.last_twist.linear.y])
        lookahead_distance = 2.0 + 0.5 * linear_speed  # 根據速度調整前視距離

        # 定義角度範圍，以當前車輛的yaw為中心
        angle_range = np.deg2rad(40)  # ±40度的範圍
        closest_index = None
        min_distance = float('inf')

        # 尋找該範圍內的最近路徑點
        for i in range(self.current_waypoint_index, len(self.waypoints)):
            wp_x, wp_y = self.waypoints[i]
            dist_to_wp = np.linalg.norm([wp_x - robot_x, wp_y - robot_y])
            direction_to_wp = np.arctan2(wp_y - robot_y, wp_x - robot_x)

            # 計算該點相對於當前車輛朝向的角度
            yaw_diff = direction_to_wp - robot_yaw
            yaw_diff = np.arctan2(np.sin(yaw_diff), np.cos(yaw_diff))  # 確保角度在[-pi, pi]範圍內

            # 如果點位於yaw ± 35度範圍內，並且距離更近
            if np.abs(yaw_diff) < angle_range and dist_to_wp < min_distance:
                min_distance = dist_to_wp
                closest_index = i

        # 如果有找到符合條件的點，則繼續使用原始最近點
        if closest_index is None:
            closest_index = self.find_closest_waypoint(robot_x, robot_y)

        target_index = closest_index

        # 根據前視距離選擇參考的路徑點
        cumulative_distance = 0.0
        for i in range(closest_index, len(self.waypoints)):
            wp_x, wp_y = self.waypoints[i]
            dist_to_wp = np.linalg.norm([wp_x - robot_x, wp_y - robot_y])
            cumulative_distance += dist_to_wp
            if cumulative_distance >= lookahead_distance:
                target_index = i
                break
        # 獲取前視點座標
        target_x, target_y = self.waypoints[target_index]

        # 計算前視點的方向
        direction_to_target = np.arctan2(target_y - robot_y, target_x - robot_x)
        yaw_error = direction_to_target - robot_yaw
        yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))  # 確保角度在[-pi, pi]範圍內

        # 根據角度誤差調整速度
        if np.abs(yaw_error) > 0.3:
            linear_speed = 0.5
        elif np.abs(yaw_error) > 0.1:
            linear_speed = 1.0
        else:
            linear_speed = 2

        # 使用PD控制器調整轉向角度
        kp, kd = self.adjust_control_params(linear_speed)
        previous_yaw_error = getattr(self, 'previous_yaw_error', 0)
        current_yaw_error_rate = yaw_error - previous_yaw_error
        steer_angle = kp * yaw_error + kd * current_yaw_error_rate
        steer_angle = np.clip(steer_angle, -0.6, 0.6)

        self.previous_yaw_error = yaw_error

        return np.array([linear_speed, steer_angle])

    def find_closest_waypoint(self, x, y):
        # 找到與當前位置最接近的路徑點
        min_distance = float('inf')
        closest_index = 0
        for i, (wp_x, wp_y) in enumerate(self.waypoints):
            dist = np.linalg.norm([wp_x - x, wp_y - y])
            if dist < min_distance:
                min_distance = dist
                closest_index = i
        return closest_index
    
    def adjust_control_params(self, linear_speed):
        if linear_speed <= 0.5:
            kp = 0.5
            kd = 0.2
        elif linear_speed <= 1.0:
            kp = 0.4
            kd = 0.3
        else:
            kp = 0.3
            kd = 0.4
        return kp, kd

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()  # 因为路径是从终点回溯到起点，所以需要反转
        print("PATH=",path)
        return path
    def calculate_movement_cost(self, current, neighbor):
        """计算移动成本"""
        # 将 Gazebo 坐标转换为图像坐标
        current_img = self.gazebo_to_image_coords(*current)
        neighbor_img = self.gazebo_to_image_coords(*neighbor)
        
        # 获取当前点和邻居点之间的区域
        x_min, x_max = min(current_img[0], neighbor_img[0]), max(current_img[0], neighbor_img[0])
        y_min, y_max = min(current_img[1], neighbor_img[1]), max(current_img[1], neighbor_img[1])
        
        # 确保索引在有效范围内
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(self.cost_map.shape[0] - 1, x_max)
        y_max = min(self.cost_map.shape[1] - 1, y_max)
        
        # 获取区域
        region = self.cost_map[x_min:x_max+1, y_min:y_max+1]
        
        # 基础移动成本
        base_cost = np.sqrt((current[0] - neighbor[0])**2 + (current[1] - neighbor[1])**2)
        
        # 根据代价地图调整成本
        if region.size > 0:  # 确保区域不为空
            max_cost = np.max(region)
            if max_cost >= 255:  # 如果路径穿过障碍物
                return float('inf')
            elif max_cost >= 100:  # 如果路径穿过高代价区域
                return base_cost * 2.0
            elif max_cost >= 50:   # 如果路径穿过低代价区域
                return base_cost * 1.5
        
        return base_cost

    def full_a_star_planning(self, start_point, goal_point):
        """
        完整的 A* 路徑規劃，使用改進的代價計算
        """
        # 轉換為圖像坐標
        start = self.gazebo_to_image_coords(*start_point)
        goal = self.gazebo_to_image_coords(*goal_point)
        
        open_set = {start}
        closed_set = set()
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        while open_set:
            current = min(open_set, key=lambda x: f_score.get(x, float('inf')))
            
            if self.is_near_goal(current, goal):
                return self.reconstruct_path(came_from, current)
            
            open_set.remove(current)
            closed_set.add(current)
            
            # 檢查8個方向
            for dx, dy in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,1), (1,-1), (-1,-1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if neighbor in closed_set:
                    continue
                
                if not self.is_valid_point(neighbor):
                    continue
                
                # 使用新的代價計算方法
                movement_cost = self.calculate_movement_cost(current, neighbor)
                if movement_cost == float('inf'):
                    continue
                
                tentative_g_score = g_score[current] + movement_cost
                
                if neighbor not in open_set:
                    open_set.add(neighbor)
                elif tentative_g_score >= g_score.get(neighbor, float('inf')):
                    continue
                
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, goal)
        
        return None

    def smooth_path(self, path, smoothing_strength=0.5):
        """
        使用加權平均對路徑進行平滑處理
        """
        if len(path) <= 2:
            return path
        
        smoothed = []
        smoothed.append(path[0])  # 保持起點不變
        
        for i in range(1, len(path)-1):
            prev = np.array(path[i-1])
            current = np.array(path[i])
            next_point = np.array(path[i+1])
            
            # 計算平滑後的點
            smoothed_point = (
                current * (1 - smoothing_strength) +
                (prev + next_point) * smoothing_strength / 2
            )
            
            # 確保平滑後的點不會太靠近障礙物
            img_x, img_y = self.gazebo_to_image_coords(*smoothed_point)
            if self.is_valid_point((img_x, img_y)):
                smoothed.append(tuple(smoothed_point))
            else:
                smoothed.append(path[i])
        
        smoothed.append(path[-1])  # 保持終點不變
        return smoothed

    def is_valid_point(self, point):
        """
        檢查點是否有效（不在障礙物內且在地圖範圍內）
        """
        x, y = point
        
        # 檢查是否在地圖範圍內
        if not (0 <= x < self.slam_map.shape[1] and 0 <= y < self.slam_map.shape[0]):
            return False
        
        # 檢查是否在障礙物內或太靠近障礙物
        safety_margin = 5  # 像素單位的安全邊距
        region = self.slam_map[
            max(0, y-safety_margin):min(y+safety_margin, self.slam_map.shape[0]),
            max(0, x-safety_margin):min(x+safety_margin, self.slam_map.shape[1])
        ]
        
        # 檢查區域內是否有障礙物（牆壁或膨脹區域）
        if np.any(region < 250):  # 假設 < 250 的值表示障礙物或膨脹區域
            return False
        
        return True
    
    # 第一部份
    def initialize_path_planning(self):
        """初始化路徑規劃過程"""
        self.load_slam_map('/home/ash/Downloads/0822-1floor/my_map0924.yaml')
        self.generate_costmap()
        
        # 2. 獲取參考 waypoints
        reference_waypoints = self.generate_random_waypoints()
        
        # 3. 使用 A* 生成基礎路徑
        start_point = reference_waypoints[0]
        goal_point = (self.target_x, self.target_y)
        print("準備進行a star 規劃")
        self.base_path = self.full_a_star_planning(start_point, goal_point)
        
        # 4. 將基礎路徑分段，準備進行強化學習優化
        self.segment_path(segment_length=10)  # 每10個點作為一

        transformed_waypoints = []
        for point in reference_waypoints:
            img_x, img_y = self.gazebo_to_image_coords(*point)
            transformed_waypoints.append((img_x, img_y))
        return transformed_waypoints

    def segment_path(self, segment_length):
        """將基礎路徑分段"""
        if self.base_path is None:
            return
            
        self.path_segments = []
        for i in range(0, len(self.base_path), segment_length):
            segment = self.base_path[i:i + segment_length]
            if len(segment) >= 2:  # 確保段至少有起點和終點
                self.path_segments.append(segment)
                
    def optimize_path_segment(self, segment_index):
        """使用強化學習優化特定路徑段"""
        if segment_index >= len(self.path_segments):
            return None
            
        current_segment = self.path_segments[segment_index]
        start_point = current_segment[0]
        end_point = current_segment[-1]
        
        # 設置環境起始狀態
        self.reset_to_position(start_point)
        
        # 進行強化學習優化
        optimized_path = []
        state = self.get_current_state()
        
        while not self.is_near_goal(self.get_robot_position()[:2], end_point):
            action = self.model.act(torch.tensor(state, device=device))
            next_state, reward, done, _ = self.step(action)
            
            current_position = self.get_robot_position()[:2]
            optimized_path.append(current_position)
            
            if done:
                break
                
            state = next_state
            
        return optimized_path
        
    def execute_optimized_navigation(self):
        """執行優化後的導航"""
        # 初始化路徑規劃
        self.initialize_path_planning()
        
        # 開始分段優化
        for i in range(len(self.path_segments)):
            print(f"Optimizing segment {i+1}/{len(self.path_segments)}")
            
            # 對當前段進行多次優化
            best_segment_path = None
            best_segment_reward = float('-inf')
            
            for attempt in range(5):  # 每段優化5次
                optimized_segment = self.optimize_path_segment(i)
                if optimized_segment:
                    segment_reward = self.evaluate_segment(optimized_segment)
                    if segment_reward > best_segment_reward:
                        best_segment_path = optimized_segment
                        best_segment_reward = segment_reward
                        
            # 儲存最佳優化結果
            if best_segment_path:
                self.optimized_segments[i] = best_segment_path
                
        return self.create_final_path()
        
    def create_final_path(self):
        """合併所有優化後的路徑段"""
        final_path = []
        for i in range(len(self.path_segments)):
            if i in self.optimized_segments:
                final_path.extend(self.optimized_segments[i])
            else:
                final_path.extend(self.path_segments[i])
        return final_path

    def heuristic(self, start, goal):
        dx, dy = goal[0] - start[0], goal[1] - start[1]
        return np.sqrt(dx**2 + dy**2) + np.abs(dx) * 0.5 + np.abs(dy) * 0.5

    def is_near_goal(self, current, goal, threshold=5):
        """檢查是否到達目標點"""
        distance = np.sqrt((current[0] - goal[0])**2 + (current[1] - goal[1])**2)
        return distance < threshold

    def reset_to_position(self, position):
        """重置機器人到指定位置"""
        x, y = position
        state_msg = ModelState()
        state_msg.model_name = 'my_robot'
        state_msg.pose.position.x = x
        state_msg.pose.position.y = y
        state_msg.pose.position.z = 0.2  # 設置適當的高度
        
        # 保持原始方向
        quaternion = quaternion_from_euler(0.0, 0.0, 0.0)
        state_msg.pose.orientation.x = quaternion[0]
        state_msg.pose.orientation.y = quaternion[1]
        state_msg.pose.orientation.z = quaternion[2]
        state_msg.pose.orientation.w = quaternion[3]
        
        try:
            self.set_model_state(state_msg)
            rospy.sleep(0.1)  # 等待位置更新
            return self.generate_occupancy_grid(x, y)
        except rospy.ServiceException as e:
            rospy.logerr(f"Reset position failed: {e}")
            return None

    def get_current_state(self):
        """獲取當前狀態"""
        robot_x, robot_y, _ = self.get_robot_position()
        return self.generate_occupancy_grid(robot_x, robot_y)

    def evaluate_segment(self, segment):
        """評估路徑段的品質"""
        if not segment:
            return float('-inf')
        
        total_reward = 0
        for i in range(len(segment)-1):
            # 計算相鄰點間的距離
            dist = np.linalg.norm([segment[i+1][0] - segment[i][0], 
                                segment[i+1][1] - segment[i][1]])
            # 檢查點是否在可行區域內
            img_x, img_y = self.gazebo_to_image_coords(segment[i][0], segment[i][1])
            if not self.is_valid_point((img_x, img_y)):
                return float('-inf')
                
            # 計算平滑度（與前後點的角度）
            if i > 0:
                v1 = np.array(segment[i]) - np.array(segment[i-1])
                v2 = np.array(segment[i+1]) - np.array(segment[i])
                angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
                smoothness_penalty = -angle * 10  # 角度越大，懲罰越大
                total_reward += smoothness_penalty
                
            total_reward += -dist  # 距離越短越好
            
        return total_reward

class ActorCritic(nn.Module):
    def __init__(self, observation_space, action_space):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(128)

        self._to_linear = self._get_conv_output_size(observation_space)

        self.fc1 = nn.Linear(self._to_linear, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 128)

        self.actor = nn.Linear(128, action_space)
        self.critic = nn.Linear(128, 1)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_space))
        
    def _get_conv_output_size(self, shape):
        x = torch.zeros(1, *shape)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = x.view(1, -1)
        return x.size(1)

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)

        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)

        action_mean = self.actor(x)
        action_log_std = self.actor_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        value = self.critic(x)
        return action_mean, action_std, value

    def act(self, state):
        # 確保 state 是 tensor，如果是 numpy，轉換為 tensor
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32).to(device)

        # 去除多餘維度直到 <= 4
        while state.dim() > 4:
            state = state.squeeze(0)

        # 添加缺少的維度直到 = 4
        while state.dim() < 4:
            state = state.unsqueeze(0)

        # 最終確認 state 是 4D
        if state.dim() != 4:
            raise ValueError(f"Expected state to be 4D, but got {state.dim()}D")

        action_mean, action_std, _ = self(state)
        action = action_mean + action_std * torch.randn_like(action_std)
        action = torch.tanh(action)
        max_action = torch.tensor([2.0, 0.6], device=action.device)
        min_action = torch.tensor([-2.0, -0.6], device=action.device)
        action = min_action + (action + 1) * (max_action - min_action) / 2
        return action.detach()

    def evaluate(self, state, action):
        action_mean, action_std, value = self(state)
        dist = torch.distributions.Normal(action_mean, action_std)
        action_log_probs = dist.log_prob(action).sum(dim=-1, keepdim=True)
        dist_entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return action_log_probs, value, dist_entropy
def ppo_update(ppo_epochs, env, model, optimizer, memory):
    for _ in range(ppo_epochs):
        # 简化采样，直接假设 memory 提供数据而不调整权重
        state_batch, action_batch, reward_batch, done_batch, next_state_batch = memory.sample(BATCH_SIZE)

        with torch.no_grad():
            old_log_probs, _, _ = model.evaluate(state_batch, action_batch)

        for _ in range(PPO_EPOCHS):
            # 移除混合精度训练
            log_probs, state_values, dist_entropy = model.evaluate(state_batch, action_batch)
            advantages = reward_batch + (1 - done_batch) * GAMMA * model(next_state_batch)[2].detach() - state_values

            ratio = (log_probs - old_log_probs).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - CLIP_PARAM, 1 + CLIP_PARAM) * advantages

            # 计算 PPO 的 actor 和 critic 损失
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(state_values.squeeze(-1), (reward_batch + (1 - done_batch) * GAMMA * model(next_state_batch)[2].detach()).unsqueeze(-1))
            entropy_loss = -0.01 * dist_entropy.mean()  # 熵正则项

            # 总损失
            loss = actor_loss + 0.5 * critic_loss + entropy_loss

            # 标准反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
#--------------
def load_slam_map(self, yaml_path):
        # 讀取 YAML 檔案
        with open(yaml_path, 'r') as file:
            map_metadata = yaml.safe_load(file)
            self.map_origin = map_metadata['origin']  # 地圖原點
            self.map_resolution = map_metadata['resolution']  # 地圖解析度
            png_path = map_metadata['image'].replace(".pgm", ".png")  # 修改為png檔案路徑
            
            # 使用 PIL 讀取PNG檔
            png_image = Image.open(png_path).convert('L')
            self.slam_map = np.array(png_image)  # 轉為NumPy陣列

def main():
    # 初始化环境
    env = GazeboEnv(None)
    model = ActorCritic(env.observation_space, env.action_space).to(device)
    env.model = model
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    memory = PrioritizedMemory(MEMORY_SIZE)

    # 初始化路径规划
    random_waypoints = env.initialize_path_planning()
    random_waypoints = np.array(random_waypoints)

    # 获取随机生成的路径点
    # random_waypoints = env.generate_random_waypoints()
    # random_waypoints = np.array(random_waypoints)

    # 检查并处理路径规划结果
    if env.base_path is None or len(env.base_path) == 0:
        print("A* did not find a valid path. Visualization skipped.")
        return

    # 转换路径为 NumPy 数组并检查格式
    path = np.array(env.base_path)
    if path.ndim != 2 or path.shape[1] != 2:
        raise ValueError(f"Path data has unexpected dimensions: {path.shape}")

    # 可视化路径和代价地图
    plt.figure(figsize=(12, 8))
    plt.imshow(env.cost_map, cmap='gray', origin='lower')
    
    # 绘制 A* 路径
    plt.plot(path[:, 0], path[:, 1], 'r-', label='Initial A* path', linewidth=2)
    
    # 绘制随机路径点
    plt.scatter(random_waypoints[:, 0], random_waypoints[:, 1], 
               color='yellow', s=50, label='Random Waypoints', zorder=4)
    
    # 连接随机路径点
    plt.plot(random_waypoints[:, 0], random_waypoints[:, 1], 
            'y--', alpha=0.5, label='Random Waypoints Path', zorder=3)
    
    # 标记起点和终点
    plt.scatter(path[0, 0], path[0, 1], color='green', s=100, label='Start', zorder=5)
    plt.scatter(path[-1, 0], path[-1, 1], color='blue', s=100, label='Goal', zorder=5)
    
    plt.title('Path Planning Visualization')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == '__main__':
    # main()
    try:
        main()  # 调用主逻辑
    except KeyboardInterrupt:
        print("Program interrupted by user!")
    finally:
        cv2.destroyAllWindows()  # 确保关闭窗口