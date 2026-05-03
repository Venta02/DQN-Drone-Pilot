"""
Custom Gym Environment untuk AirSim Drone
==========================================
PERBAIKAN v6 - OPTIMIZED:
- Reward shaping yang lebih baik untuk faster learning
- Jarak aman lebih realistis (4m safe, 7m acceptable)
- Ketinggian terbang: 3.5 meter
- Obstacle detection threshold yang lebih masuk akal
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import airsim
import time

from config import ENV_CONFIG, REWARD_CONFIG, STATE_CONFIG, ACTION_CONFIG


class DroneObstacleEnv(gym.Env):
    """
    Custom Environment untuk Drone dengan AirSim
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, render_mode=None):
        super(DroneObstacleEnv, self).__init__()
        
        # Konfigurasi
        self.drone_name = ENV_CONFIG["drone_name"]
        self.initial_position = ENV_CONFIG["initial_position"]
        self.forward_speed = ENV_CONFIG["forward_speed"]
        self.duration = ENV_CONFIG["duration"]
        self.safe_distance = ENV_CONFIG["safe_distance"]
        self.too_far_distance = ENV_CONFIG["too_far_distance"]
        self.collision_distance = ENV_CONFIG["collision_distance"]
        self.max_steps = ENV_CONFIG["max_steps_per_episode"]
        
        # ===== KETINGGIAN TERBANG: 3.5 METER =====
        self.flight_height = -4.3  # Z = -3.5 berarti 3.5 meter di atas ground
        
        # State normalization
        self.normalize_state = STATE_CONFIG["normalize_state"]
        self.max_distance = STATE_CONFIG["max_distance"]
        self.max_velocity = STATE_CONFIG["max_velocity"]
        
        # =========================================================================
        # OBSTACLE DETECTION THRESHOLD - DIPERBAIKI
        # =========================================================================
        # Threshold untuk menentukan "ada obstacle" vs "clear path"
        # Sebelumnya: 80m (terlalu jauh, tidak masuk akal)
        # Sekarang: 15m (lebih realistis untuk obstacle avoidance)
        self.obstacle_detection_threshold = 15.0
        
        # Action dan Observation space
        self.action_space = spaces.Discrete(ACTION_CONFIG["action_size"])
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]) if self.normalize_state else np.array([self.max_distance, self.max_velocity, 1.0]),
            dtype=np.float32
        )
        
        # AirSim client
        self.client = None
        self.is_connected = False
        
        # Episode state
        self.current_step = 0
        self.is_moving = True
        self.episode_ended = False
        self.previous_action = None
        self.previous_distance = None  # Track distance untuk reward shaping
        
        self.render_mode = render_mode
        
    def _connect_airsim(self):
        """Koneksi ke AirSim simulator"""
        try:
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            self.client.enableApiControl(True, self.drone_name)
            self.client.armDisarm(True, self.drone_name)
            self.is_connected = True
            print("[INFO] Berhasil terhubung ke AirSim!")
            print(f"[INFO] Ketinggian terbang: {abs(self.flight_height)} meter")
            print(f"[INFO] Jarak aman: {self.safe_distance}m, Acceptable: {self.too_far_distance}m")
        except Exception as e:
            print(f"[ERROR] Gagal terhubung ke AirSim: {e}")
            self.is_connected = False
            raise ConnectionError("Pastikan AirSim simulator sudah berjalan!")
    
    def _get_distance_to_obstacle(self):
        """Mendapatkan jarak ke obstacle"""
        try:
            distance_data = self.client.getDistanceSensorData(
                distance_sensor_name="DistanceSensorFront",
                vehicle_name=self.drone_name
            )
            distance = distance_data.distance
            
            if distance < 0 or distance > self.max_distance:
                return self.max_distance
            return distance
            
        except Exception as e:
            try:
                lidar_data = self.client.getLidarData(vehicle_name=self.drone_name)
                if len(lidar_data.point_cloud) >= 3:
                    points = np.array(lidar_data.point_cloud).reshape(-1, 3)
                    front_points = points[points[:, 0] > 0]
                    if len(front_points) > 0:
                        distances = np.linalg.norm(front_points, axis=1)
                        return min(np.min(distances), self.max_distance)
                return self.max_distance
            except:
                return self.max_distance
    
    def _get_velocity(self):
        """Mendapatkan kecepatan drone"""
        try:
            kinematics = self.client.simGetGroundTruthKinematics(self.drone_name)
            velocity = kinematics.linear_velocity
            speed = np.sqrt(velocity.x_val**2 + velocity.y_val**2 + velocity.z_val**2)
            return speed
        except:
            return 0.0
    
    def _get_state(self):
        """Mendapatkan state saat ini"""
        distance = self._get_distance_to_obstacle()
        velocity = self._get_velocity()
        
        if self.normalize_state:
            distance_norm = np.clip(distance / self.max_distance, 0.0, 1.0)
            velocity_norm = np.clip(velocity / self.max_velocity, 0.0, 1.0)
        else:
            distance_norm = distance
            velocity_norm = velocity
        
        state = np.array([
            distance_norm,
            velocity_norm,
            float(self.is_moving)
        ], dtype=np.float32)
        
        return state
    
    def _check_collision(self):
        """Cek apakah drone tabrakan"""
        try:
            collision_info = self.client.simGetCollisionInfo(self.drone_name)
            return collision_info.has_collided
        except:
            return False
    
    def _calculate_reward(self, action, distance):
        """
        Menghitung reward - DIPERBAIKI untuk faster learning
        
        Reward structure:
        - Clear signal untuk success/failure
        - Graduated rewards untuk encourage proper behavior
        - Approach rewards untuk encourage forward movement
        """
        reward = REWARD_CONFIG["time_penalty"]  # Base penalty per step
        done = False
        info = {"reason": "moving", "distance": distance}
        
        obstacle_detected = distance < self.obstacle_detection_threshold
        
        # =====================================================================
        # CEK COLLISION (prioritas tertinggi)
        # =====================================================================
        if self._check_collision() or distance <= self.collision_distance:
            reward = REWARD_CONFIG["collision_penalty"]  # -100
            done = True
            info = {"reason": "collision", "distance": distance}
            return reward, done, info
        
        # =====================================================================
        # ACTION: STOP (action == 0)
        # =====================================================================
        if action == 0:
            self.is_moving = False
            
            if obstacle_detected:
                if distance <= self.safe_distance:
                    # PERFECT STOP - dalam zona aman
                    reward = REWARD_CONFIG["stop_near_obstacle"]  # +100
                    done = True
                    info = {"reason": "success_stop", "distance": distance}
                    
                elif distance <= self.too_far_distance:
                    # GOOD STOP - acceptable tapi tidak optimal
                    reward = REWARD_CONFIG["partial_success"]  # +50
                    done = True
                    info = {"reason": "partial_stop", "distance": distance}
                    
                else:
                    # STOP TERLALU JAUH - obstacle terdeteksi tapi masih jauh
                    # Penalty kecil, episode TIDAK selesai agar agent belajar maju
                    reward = REWARD_CONFIG["stop_too_far_penalty"]  # -10
                    done = False
                    info = {"reason": "stop_too_far", "distance": distance}
            else:
                # STOP TANPA OBSTACLE - tidak ada alasan untuk stop
                reward = REWARD_CONFIG["stop_no_obstacle_penalty"]  # -20
                done = False
                info = {"reason": "stop_no_obstacle", "distance": distance}
        
        # =====================================================================
        # ACTION: MOVE_FORWARD (action == 1)
        # =====================================================================
        elif action == 1:
            self.is_moving = True
            
            if obstacle_detected:
                if distance <= self.safe_distance:
                    # DANGER! Terlalu dekat tapi masih maju
                    reward = REWARD_CONFIG["danger_zone_penalty"]  # -5
                    done = False
                    info = {"reason": "danger_too_close", "distance": distance}
                    
                elif distance <= self.too_far_distance:
                    # APPROACHING - mendekati zona target, bagus!
                    reward = REWARD_CONFIG["approaching_reward"]  # +2
                    done = False
                    info = {"reason": "approaching", "distance": distance}
                    
                else:
                    # MOVING TOWARD - masih jauh tapi bergerak ke obstacle
                    # Reward kecil untuk encourage movement
                    reward = REWARD_CONFIG["approaching_reward"] * 0.5  # +1
                    done = False
                    info = {"reason": "moving_toward", "distance": distance}
            else:
                # MOVING - tidak ada obstacle, neutral
                reward = 0.0
                done = False
                info = {"reason": "moving", "distance": distance}
        
        # =====================================================================
        # BONUS: Progress reward (optional, untuk faster learning)
        # =====================================================================
        # Jika drone mendekat ke obstacle, berikan bonus kecil
        if self.previous_distance is not None and obstacle_detected:
            distance_delta = self.previous_distance - distance
            if distance_delta > 0 and action == 1:
                # Mendekat ke obstacle - bonus proportional to progress
                reward += distance_delta * 0.5
        
        self.previous_distance = distance
        
        return reward, done, info
    
    def reset(self, seed=None, options=None):
        """Reset environment ke kondisi awal"""
        super().reset(seed=seed)
        
        if not self.is_connected:
            self._connect_airsim()
        
        # Reset drone
        self.client.reset()
        self.client.enableApiControl(True, self.drone_name)
        self.client.armDisarm(True, self.drone_name)
        
        # Set posisi awal dengan teleport
        pose = airsim.Pose(
            airsim.Vector3r(
                self.initial_position[0],
                self.initial_position[1],
                self.flight_height
            ),
            airsim.Quaternionr(0, 0, 0, 1)
        )
        self.client.simSetVehiclePose(pose, True, self.drone_name)
        time.sleep(0.1)
        
        # Stabilkan drone di ketinggian yang tepat
        self.client.moveToZAsync(
            z=self.flight_height,
            velocity=1,
            vehicle_name=self.drone_name
        ).join()
        
        # Hover sebentar untuk stabilisasi
        self.client.hoverAsync(vehicle_name=self.drone_name).join()
        time.sleep(0.2)
        
        # Reset state
        self.current_step = 0
        self.is_moving = True
        self.episode_ended = False
        self.previous_action = None
        self.previous_distance = None
        
        observation = self._get_state()
        info = {"step": 0}
        
        return observation, info
    
    def step(self, action):
        """
        Eksekusi satu step
        
        Menggunakan moveByVelocityZAsync untuk MENJAGA KETINGGIAN KONSTAN
        """
        self.current_step += 1
        
        # Dapatkan jarak SEBELUM action
        raw_distance = self._get_distance_to_obstacle()
        
        # Eksekusi action hanya jika berubah
        if action != self.previous_action:
            if action == 1:  # MOVE_FORWARD
                self.client.moveByVelocityZAsync(
                    vx=self.forward_speed,
                    vy=0,
                    z=self.flight_height,
                    duration=100,
                    drivetrain=airsim.DrivetrainType.ForwardOnly,
                    yaw_mode=airsim.YawMode(False, 0),
                    vehicle_name=self.drone_name
                )
                
            elif action == 0:  # STOP
                self.client.moveByVelocityZAsync(
                    vx=0,
                    vy=0,
                    z=self.flight_height,
                    duration=100,
                    drivetrain=airsim.DrivetrainType.ForwardOnly,
                    yaw_mode=airsim.YawMode(False, 0),
                    vehicle_name=self.drone_name
                )
            
            self.previous_action = action
        
        # Tunggu untuk timing step yang konsisten
        time.sleep(self.duration)
        
        # Hitung reward
        reward, terminated, info = self._calculate_reward(action, raw_distance)
        
        # Dapatkan state baru
        observation = self._get_state()
        
        # Cek truncated
        truncated = self.current_step >= self.max_steps
        
        info["step"] = self.current_step
        info["action"] = ACTION_CONFIG["actions"][action]
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode == "human":
            pass
    
    def close(self):
        if self.is_connected and self.client:
            self.client.armDisarm(False, self.drone_name)
            self.client.enableApiControl(False, self.drone_name)
            print("[INFO] Environment ditutup.")


class DroneObstacleEnvSimulated(DroneObstacleEnv):
    """
    Versi simulasi (tanpa AirSim) - DIPERBAIKI
    """
    
    def __init__(self, render_mode=None):
        super().__init__(render_mode)
        self.simulated_distance = 30.0
        self.simulated_velocity = 0.0
        self.is_connected = True
        
    def _connect_airsim(self):
        self.is_connected = True
        print("[INFO] Mode simulasi aktif")
        print(f"[INFO] Jarak aman: {self.safe_distance}m, Acceptable: {self.too_far_distance}m")
    
    def _get_distance_to_obstacle(self):
        return self.simulated_distance
    
    def _get_velocity(self):
        return self.simulated_velocity
    
    def _check_collision(self):
        return self.simulated_distance <= self.collision_distance
    
    def reset(self, seed=None, options=None):
        # Jarak awal random antara 10-30m (dalam range deteksi)
        self.simulated_distance = np.random.uniform(10.0, 30.0)
        self.simulated_velocity = 0.0
        self.current_step = 0
        self.is_moving = True
        self.episode_ended = False
        self.previous_distance = None
        
        observation = self._get_state()
        info = {
            "step": 0, 
            "mode": "simulated", 
            "initial_distance": self.simulated_distance
        }
        return observation, info
    
    def step(self, action):
        self.current_step += 1
        distance_before = self.simulated_distance
        
        if action == 1:  # MOVE_FORWARD
            self.simulated_velocity = self.forward_speed
            movement = self.forward_speed * self.duration
            self.simulated_distance -= movement
            self.simulated_distance = max(0, self.simulated_distance)
            self.is_moving = True
        else:  # STOP
            self.simulated_velocity = 0.0
            self.is_moving = False
        
        reward, terminated, info = self._calculate_reward(action, distance_before)
        observation = self._get_state()
        truncated = self.current_step >= self.max_steps
        
        info["step"] = self.current_step
        info["action"] = ACTION_CONFIG["actions"][action]
        info["mode"] = "simulated"
        
        return observation, reward, terminated, truncated, info
    
    def close(self):
        print("[INFO] Simulated environment ditutup.")


def register_env():
    from gymnasium.envs.registration import register
    try:
        register(id='DroneObstacle-v0', entry_point='drone_env:DroneObstacleEnv', max_episode_steps=500)
        register(id='DroneObstacle-Simulated-v0', entry_point='drone_env:DroneObstacleEnvSimulated', max_episode_steps=500)
    except:
        pass


if __name__ == "__main__":
    print("="*60)
    print("Testing Simulated Environment (OPTIMIZED)")
    print("="*60)
    
    env = DroneObstacleEnvSimulated()
    obs, info = env.reset()
    
    print(f"\nKonfigurasi:")
    print(f"  Ketinggian terbang: {abs(env.flight_height)} meter")
    print(f"  Safe distance: {env.safe_distance}m")
    print(f"  Too far distance: {env.too_far_distance}m")
    print(f"  Collision distance: {env.collision_distance}m")
    print(f"  Detection threshold: {env.obstacle_detection_threshold}m")
    print(f"\nInitial distance: {info['initial_distance']:.1f}m")
    print("-"*60)
    
    total_reward = 0
    for i in range(100):
        # Simple policy: stop jika dalam safe_distance
        if env.simulated_distance <= env.safe_distance:
            action = 0  # STOP
        else:
            action = 1  # FORWARD
            
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"Step {i+1}: {info['action']:12} | "
              f"Distance: {info['distance']:6.2f}m | "
              f"Reward: {reward:+7.2f} | "
              f"Reason: {info['reason']}")
        
        if terminated or truncated:
            print("-"*60)
            print(f"Episode ended: {info['reason']}")
            break
    
    print(f"\nTotal reward: {total_reward:.2f}")
    env.close()
