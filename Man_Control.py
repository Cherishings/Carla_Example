import carla
import pygame
import cv2
import numpy as np
import time
import random

def main():
    pygame.init()
    display = pygame.display.set_mode((1024, 512))
    pygame.display.set_caption("CARLA Manual Control")
    clock = pygame.time.Clock()

    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.find("vehicle.lincoln.mkz_2017")
    spawn_point = random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    # Attach camera
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '1024')
    camera_bp.set_attribute('image_size_y', '512')
    camera_bp.set_attribute('fov', '90')
    camera_transform = carla.Transform(carla.Location(x=-1.5, z=2))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    image_data = {"image": None}

    def process_image(image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        image_data["image"] = array[:, :, :3]

    camera.listen(lambda image: process_image(image))

    control = carla.VehicleControl()
    try:
        while True:
            clock.tick(30)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

            keys = pygame.key.get_pressed()
            control.throttle = 1.0 if keys[pygame.K_w] else 0.0
            control.brake = 1.0 if keys[pygame.K_s] else 0.0
            control.steer = -1.0 if keys[pygame.K_a] else (1.0 if keys[pygame.K_d] else 0.0)
            control.hand_brake = keys[pygame.K_SPACE]

            vehicle.apply_control(control)

            if image_data["image"] is not None:
                frame = image_data["image"]
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Compute vehicle speed in km/h
                velocity = vehicle.get_velocity()
                speed = 3.6 * (velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5  # m/s to km/h

                # Convert to Pygame surface
                surface = pygame.surfarray.make_surface(np.flipud(np.rot90(frame)))

                # Display image
                display.blit(surface, (0, 0))

                # Draw speed as text
                font = pygame.font.Font(None, 36)
                speed_text = font.render(f"Speed: {speed:.1f} km/h", True, (255, 255, 255))
                display.blit(speed_text, (20, 20))

                pygame.display.flip()


    finally:
        camera.stop()
        vehicle.destroy()
        pygame.quit()
        print("Cleaned up.")

if __name__ == '__main__':
    main()
