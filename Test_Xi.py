import carla

client = carla.Client("localhost", 2000)
world = client.get_world()

bp = world.get_blueprint_library().find('vehicle.lincoln.mkz_2017')
spawn_point = world.get_map().get_spawn_points()[0]
vehicle = world.try_spawn_actor(bp, spawn_point)

if vehicle is not None:
    bb = vehicle.bounding_box
    print(f"Vehicle dimensions:")
    print(f"  Length: {bb.extent.x * 2:.3f} m")
    print(f"  Width : {bb.extent.y * 2:.3f} m")
    print(f"  Height: {bb.extent.z * 2:.3f} m")
    vehicle.destroy()