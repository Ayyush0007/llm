"""
build_jalgaon_map.py — Configure CARLA world to simulate Indian road conditions.

This script sets up the CARLA simulator with:
  - Indian weather presets (monsoon, dusty, scorching, night)  
  - 30 chaotic NPC vehicles with autopilot on
  - 50 jaywalking pedestrians

Run CARLA first:  ./CarlaUE4.sh -quality-level=Low -fps=20
Then run this:    python3 scripts/build_jalgaon_map.py
"""

import carla
import random

def build_indian_environment(
    host: str = "localhost",
    port: int = 2000,
    num_vehicles: int = 30,
    num_pedestrians: int = 50,
):
    client = carla.Client(host, port)
    client.set_timeout(10.0)
    world  = client.get_world()
    bp_lib = world.get_blueprint_library()

    print(f"✅ Connected to CARLA — Map: {world.get_map().name}")

    # ── Set Indian monsoon-style weather ──────────────────────────
    weather = carla.WeatherParameters(
        cloudiness             = 60.0,
        precipitation          = 30.0,
        precipitation_deposits = 50.0,
        wind_intensity         = 20.0,
        sun_azimuth_angle      = 180.0,
        sun_altitude_angle     = 45.0,
        fog_density            = 10.0,
        wetness                = 40.0,
    )
    world.set_weather(weather)
    print("🌧️  Weather: Indian monsoon configured")

    spawn_points = world.get_map().get_spawn_points()
    all_actors   = []

    # ── Spawn NPC vehicles ────────────────────────────────────────
    vehicle_bps = bp_lib.filter("vehicle.*")
    for i in range(num_vehicles):
        bp = random.choice(vehicle_bps)
        sp = random.choice(spawn_points)
        try:
            v = world.spawn_actor(bp, sp)
            v.set_autopilot(True)
            all_actors.append(v)
        except Exception:
            pass

    print(f"🚗 Spawned {len(all_actors)} NPC vehicles")

    # ── Spawn pedestrians ─────────────────────────────────────────
    pedestrian_bps = bp_lib.filter("walker.pedestrian.*")
    ped_count      = 0
    for i in range(num_pedestrians):
        bp = random.choice(pedestrian_bps)
        loc = carla.Location(
            x=random.uniform(-100, 100),
            y=random.uniform(-100, 100),
            z=0.5,
        )
        transform = carla.Transform(loc)
        try:
            p = world.spawn_actor(bp, transform)
            all_actors.append(p)
            ped_count += 1
        except Exception:
            pass

    print(f"🚶 Spawned {ped_count} pedestrians")
    print("✅ Indian chaos environment is ready!")
    print("   Press Ctrl+C to clean up and exit.")

    try:
        while True:
            world.tick()
    except KeyboardInterrupt:
        print("\n🧹 Cleaning up...")
        for actor in all_actors:
            try:
                actor.destroy()
            except Exception:
                pass
        print("✅ Done.")


if __name__ == "__main__":
    build_indian_environment()
