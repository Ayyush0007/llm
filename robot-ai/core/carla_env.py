import carla
import gym
import numpy as np
import cv2
import random
import math
from gym import spaces

class CarlaIndianEnv(gym.Env):
    """
    CARLA environment tuned for Indian road chaos.
    Observation : RGB camera (84x84x3)
    Action      : [steer, throttle, brake] — continuous
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, host="localhost", port=2000, image_size=84):
        super().__init__()
        self.image_size   = image_size
        self.client       = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world        = self.client.get_world()
        self.bp_lib       = self.world.get_blueprint_library()

        # Action space: steer [-1,1], throttle [0,1], brake [0,1]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )

        # Observation: RGB image
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(image_size, image_size, 3),
            dtype=np.uint8
        )

        self.vehicle    = None
        self.camera     = None
        self.collision  = None
        self._image     = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        self._collision_hist = []
        self._actor_list = []

    def reset(self):
        self._cleanup()
        self._collision_hist = []

        # Spawn ego vehicle
        vehicle_bp = self.bp_lib.filter("vehicle.tesla.model3")[0]
        spawn_point = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        self._actor_list.append(self.vehicle)

        # Attach RGB camera
        cam_bp = self.bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(self.image_size))
        cam_bp.set_attribute("image_size_y", str(self.image_size))
        cam_bp.set_attribute("fov", "90")
        cam_transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.camera = self.world.spawn_actor(cam_bp, cam_transform, attach_to=self.vehicle)
        self.camera.listen(self._on_camera)
        self._actor_list.append(self.camera)

        # Attach collision sensor
        col_bp = self.bp_lib.find("sensor.other.collision")
        self.collision = self.world.spawn_actor(col_bp, carla.Transform(), attach_to=self.vehicle)
        self.collision.listen(lambda e: self._collision_hist.append(e))
        self._actor_list.append(self.collision)

        self._set_random_indian_weather()

        self.world.tick()
        return self._image.copy()

    def step(self, action):
        steer, throttle, brake = float(action[0]), float(action[1]), float(action[2])

        control = carla.VehicleControl(
            steer=np.clip(steer, -1.0, 1.0),
            throttle=np.clip(throttle, 0.0, 1.0),
            brake=np.clip(brake, 0.0, 1.0),
        )
        self.vehicle.apply_control(control)
        self.world.tick()

        obs     = self._image.copy()
        reward  = self._compute_reward()
        done    = len(self._collision_hist) > 0
        info    = {"collision": done}

        return obs, reward, done, info

    def _compute_reward(self):
        v       = self.vehicle.get_velocity()
        speed   = math.sqrt(v.x**2 + v.y**2 + v.z**2) * 3.6  # km/h

        # Reward forward motion (target: 20–40 km/h)
        reward = speed / 40.0

        if len(self._collision_hist) > 0:
            reward -= 10.0

        if speed < 1.0:
            reward -= 0.2

        ctrl = self.vehicle.get_control()
        reward -= abs(ctrl.steer) * 0.1

        return reward

    def _on_camera(self, image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))
        self._image = arr[:, :, :3].copy()

    def _set_random_indian_weather(self):
        presets = [
            carla.WeatherParameters(cloudiness=10, precipitation=0, sun_altitude_angle=80),
            carla.WeatherParameters(cloudiness=90, precipitation=80, wetness=80, wind_intensity=40),
            carla.WeatherParameters(cloudiness=40, fog_density=25, sun_altitude_angle=50),
            carla.WeatherParameters(cloudiness=20, precipitation=0, sun_altitude_angle=-10),
        ]
        self.world.set_weather(random.choice(presets))

    def _cleanup(self):
        for actor in self._actor_list:
            try:
                actor.destroy()
            except:
                pass
        self._actor_list = []

    def render(self, mode="human"):
        cv2.imshow("Robot POV", self._image)
        cv2.waitKey(1)

    def close(self):
        self._cleanup()
        cv2.destroyAllWindows()
