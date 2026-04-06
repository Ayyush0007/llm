"""
indian_carla_env.py — Custom Gymnasium environment for training on Indian road conditions.

Wraps the CARLA Python API into a standard RL interface for Stable-Baselines3.
Incorporates outputs from our simulated Indian sensors (LiDAR, Dust, etc.)
and includes severe penalties for hitting humans, cows, and falling in potholes/cliffs.

Requires:
  pip install gymnasium stable-baselines3 opencv-python
"""

import time
import math
import random
import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces

try:
    import carla
    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False


class IndianCarlaEnv(gym.Env):
    """
    OpenAI Gym Environment wrapper for CARLA tailored for Indian driving.
    """
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, host="localhost", port=2000, max_steps=1000):
        super().__init__()
        
        self.host = host
        self.port = port
        self.max_steps = max_steps
        
        # Continuous action space: [Throttle (0 to 1), Steer (-1 to 1), Brake (0 to 1)]
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Observation space: Minimal fused world model + 64x64 compressed depth image
        # Using a Dict space since we have both image data and telemetry vector.
        self.observation_space = spaces.Dict({
            "depth_map": spaces.Box(low=0, high=255, shape=(64, 64, 1), dtype=np.uint8),
            "telemetry": spaces.Box(
                # [Speed(m/s), Danger_Center, Danger_Left, Danger_Right, LiDAR_Front, Dust_AQI, Cliff_Drop]
                low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                high=np.array([30.0, 1.0, 1.0, 1.0, 15.0, 500.0, 1.0]),
                dtype=np.float32
            )
        })

        if not CARLA_AVAILABLE:
            print("⚠️  CARLA not found. Environment will run in Dry-Run Mock Mode.")
            
        self.client = None
        self.world = None
        self.vehicle = None
        self.camera = None
        self.actor_list = []
        
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        
        self.current_step = 0
        self.total_reward = 0.0
        
        self._latest_image = np.zeros((64, 64, 1), dtype=np.uint8)
        self._collision_history = []
        
        if CARLA_AVAILABLE:
            self._connect_to_carla()

    def _connect_to_carla(self):
        try:
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(5.0)
            self.world = self.client.get_world()
            self.blueprint_library = self.world.get_blueprint_library()
        except Exception as e:
            print(f"Failed to connect to CARLA: {e}")
            self.client = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.total_reward = 0.0
        self._collision_history.clear()
        
        if not self.client:
            # Mock reset
            return self._get_mock_obs(), {}

        self._cleanup_actors()

        # Spawn Vehicle
        vehicle_bp = self.blueprint_library.filter("vehicle.tesla.model3")[0]
        spawn_points = self.world.get_map().get_spawn_points()
        self.vehicle = self.world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
        
        while self.vehicle is None: # Retry if blocked
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
            
        self.actor_list.append(self.vehicle)

        # Attach sensors
        self._setup_sensors()
        self._randomize_indian_weather()

        # Wait for agents to drop to ground
        time.sleep(0.5)
        
        obs = self._get_observation()
        info = {}
        return obs, info

    def _setup_sensors(self):
        # Depth camera (gives direct depth rather than processing RGB, faster for RL)
        cam_bp = self.blueprint_library.find("sensor.camera.depth")
        cam_bp.set_attribute("image_size_x", "64")
        cam_bp.set_attribute("image_size_y", "64")
        cam_bp.set_attribute("fov", "90")
        cam_transform = carla.Transform(carla.Location(x=2.0, z=1.4))
        
        self.camera = self.world.spawn_actor(cam_bp, cam_transform, attach_to=self.vehicle)
        self.actor_list.append(self.camera)
        self.camera.listen(lambda image: self._process_image(image))

        # Collision sensor
        col_bp = self.blueprint_library.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(col_bp, carla.Transform(), attach_to=self.vehicle)
        self.actor_list.append(self.collision_sensor)
        self.collision_sensor.listen(lambda event: self._collision_history.append(event))

    def _process_image(self, image):
        # CARLA depth map conversion
        image.convert(carla.ColorConverter.Depth)
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))
        # Keep only R channel which has most depth info in this converter, normalize to 0-255
        self._latest_image = array[:, :, 0:1]

    def _randomize_indian_weather(self):
        # 1. Hot & Clear, 2. Dusty (High Fog), 3. Monsoon (Heavy Rain)
        weather = carla.WeatherParameters()
        choice = random.choice(["clear", "dusty", "monsoon"])
        
        if choice == "dusty":
            weather.fog_density = random.uniform(20.0, 50.0)
            weather.sun_altitude_angle = 45.0
        elif choice == "monsoon":
            weather.precipitation = 80.0
            weather.precipitation_deposits = 50.0
            weather.cloudiness = 90.0
            weather.wetness = 100.0
        else:
            weather.sun_altitude_angle = 80.0 # Hot sun
            
        self.world.set_weather(weather)

    def step(self, action):
        self.current_step += 1
        
        throttle, steer, brake = float(action[0]), float(action[1]), float(action[2])
        
        if self.client and self.vehicle:
            control = carla.VehicleControl(
                throttle=max(0.0, min(1.0, throttle)),
                steer=max(-1.0, min(1.0, steer)),
                brake=max(0.0, min(1.0, brake))
            )
            self.vehicle.apply_control(control)
        
        # Wait for tick
        time.sleep(0.05)
        
        obs = self._get_observation()
        reward, terminated, truncated = self._compute_reward_and_done()
        
        self.total_reward += reward
        info = {"total_reward": self.total_reward}
        
        return obs, reward, terminated, truncated, info

    def _compute_reward_and_done(self):
        reward = 0.0
        terminated = False
        truncated = self.current_step >= self.max_steps

        # 1. Forward progress reward
        speed = 0.0
        if self.vehicle:
            v = self.vehicle.get_velocity()
            speed = math.sqrt(v.x**2 + v.y**2 + v.z**2)
        else:
            speed = 5.0 # Mock speed
            
        # Reward maintaining a safe speed (around 5-10 m/s for congested roads)
        if 2.0 < speed < 10.0:
            reward += 1.0
        elif speed < 0.5:
            reward -= 0.5 # Penalty for not moving
            
        # 2. Collision penalties (Indian Specific)
        if len(self._collision_history) > 0:
            terminated = True
            hit_actor = self._collision_history[0].other_actor
            
            if hit_actor and hit_actor.type_id:
                # In CARLA, walkers represent people/pedestrians
                if "walker" in hit_actor.type_id:
                    reward -= 1000.0  # Hit living being!
                    print("💥 COLLISION: Pedestrian/Living Being (-1000)")
                # Two-wheelers
                elif "vehicle.bh" in hit_actor.type_id or "vehicle.yamaha" in hit_actor.type_id:
                    reward -= 500.0
                    print("💥 COLLISION: Two-Wheeler (-500)")
                else:
                    reward -= 200.0   # General object/car
                    print("💥 COLLISION: Vehicle/Wall (-200)")
            else:
                reward -= 100.0
                
        # 3. Drop-off / Cliff penalty (Z-velocity drops sharply)
        if self.vehicle:
            v_z = self.vehicle.get_velocity().z
            if v_z < -3.0: # Falling!
                terminated = True
                reward -= 300.0
                print("💥 FALL: Dropped into pothole/cliff (-300)")

        return reward, terminated, truncated

    def _get_observation(self):
        if not self.client:
            return self._get_mock_obs()
            
        v = self.vehicle.get_velocity()
        speed = math.sqrt(v.x**2 + v.y**2 + v.z**2)
        
        # We derive "Danger" from the deep center of the depth map
        center_zone = self._latest_image[20:44, 20:44]
        danger_center = float(255 - np.mean(center_zone)) / 255.0  # Closer = higher danger
        
        telemetry = np.array([
            speed,
            danger_center,
            0.0, # danger_left (simplified)
            0.0, # danger_right (simplified)
            10.0,# lidar front (mocked here, ideally wired to real Carla Lidar)
            15.0,# Dust AQI (mocked here, ideally from weather)
            0.0  # Cliff drop (mocked)
        ], dtype=np.float32)

        return {
            "depth_map": self._latest_image,
            "telemetry": telemetry
        }

    def _get_mock_obs(self):
        """Returns dummy data when CARLA is not running for --dry-run testing"""
        return {
            "depth_map": np.zeros((64, 64, 1), dtype=np.uint8),
            "telemetry": np.array([5.0, 0.1, 0.1, 0.1, 10.0, 50.0, 0.0], dtype=np.float32)
        }

    def _cleanup_actors(self):
        for actor in self.actor_list:
            if actor.is_alive:
                actor.destroy()
        self.actor_list.clear()

    def close(self):
        self._cleanup_actors()
