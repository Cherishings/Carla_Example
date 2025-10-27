import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch import nn

import carla
import pygame
import time
import random

# ----------------------------
# Set device
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------
# Constants & Transforms
# ----------------------------
CLASS_MAP = {
    1: 1,   # Road
    14: 2,  # Cars
    24: 3,   # Road Lane Marks
    21: 3, # Obstacles
    12: 4, # Pedestrians
    6: 5 # Traffic lights
}
BACKGROUND_LABEL = 0  # Everything else becomes background
NUM_CLASSES = 7       # (Background + Road + Cars + Lane Mark, etc)

transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

def remap_mask(mask, class_map, background_label=0):
    remapped = np.full_like(mask, background_label)
    for old_class, new_class in class_map.items():
        remapped[mask == old_class] = new_class
    return remapped

# ----------------------------
# Define the Model
# ----------------------------
class VehicleNNController(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(VehicleNNController, self).__init__()
        self.unet = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes
        )
        self.encoder = self.unet.encoder
        self.decoder = self.unet.decoder
        self.segmentation_head = self.unet.segmentation_head

        self.flatten = nn.Flatten()
        self.fc_backbone = nn.Sequential(
            nn.Linear(512 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
        )

        self.meta_input_net = nn.Sequential(
            nn.BatchNorm1d(3),
            nn.Linear(3, 20),
            nn.ReLU(),
            nn.Linear(20, 6),
            nn.ReLU(),
            nn.Linear(6, 3),
            nn.ReLU()
        )

        self.fc_combined = nn.Sequential(
            nn.Linear(128 + 3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.output_waypoints = nn.Linear(32, 10)
        self.output_speed = nn.Linear(32, 1)
        self.output_brake = nn.Linear(32, 1)

    def forward(self, x_img, target_point, current_speed):
        features = self.encoder(x_img)
        decoder_output = self.decoder(*features)
        semantic_out = self.segmentation_head(decoder_output)

        x_deep = features[-1]
        flat_feat = self.flatten(x_deep)
        features_128 = self.fc_backbone(flat_feat)

        meta = torch.cat([target_point, current_speed.unsqueeze(1)], dim=1)
        meta_feat = self.meta_input_net(meta)
        combined = torch.cat([features_128, meta_feat], dim=1)
        fused = self.fc_combined(combined)

        waypoints = self.output_waypoints(fused)
        speed = self.output_speed(fused)
        brake = torch.sigmoid(self.output_brake(fused))

        return semantic_out, waypoints, speed, brake


model = VehicleNNController().to(device)
model.load_state_dict(torch.load("Corrected_VehicleNNController_NAG.pth", map_location=device))
model.eval()
print("Loaded trained model")


def get_world_waypoints(vehicle, relative_waypoints):
    """
    Convert relative waypoints to world coordinates based on vehicle transform.
    """
    world_waypoints = []
    transform = vehicle.get_transform()
    yaw = np.radians(transform.rotation.yaw)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    for x, y in relative_waypoints:
        # Rotate and translate
        world_x = transform.location.x + x * cos_yaw - y * sin_yaw
        world_y = transform.location.y + x * sin_yaw + y * cos_yaw
        world_waypoints.append((world_x, world_y))

    return world_waypoints


def main():
    pygame.init()
    display = pygame.display.set_mode((896 * 2, 512))
    pygame.display.set_caption("Model V3 Control")
    clock = pygame.time.Clock()

    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    # Load Town05
    client.load_world('Town04')
    time.sleep(2)  # Give it a moment to finish loading
    world = client.get_world()



    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.find("vehicle.lincoln.mkz_2017")
    spawn_index = 0  # Choose a fixed index
    spawn_point = world.get_map().get_spawn_points()[spawn_index]

    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    # Attach camera
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '1024')
    camera_bp.set_attribute('image_size_y', '512')
    camera_bp.set_attribute('fov', '90')
    camera_transform = carla.Transform(carla.Location(x=-1.5, z=2))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

        # Spawn a static vehicle 5 meters in front of ego vehicle
    static_vehicle_bp = blueprint_library.find("vehicle.audi.tt")  # You can change this
    static_transform = vehicle.get_transform()
    forward_vector = static_transform.get_forward_vector()
    spawn_location = static_transform.location + carla.Location(
        x=forward_vector.x * 40.0,
        y=forward_vector.y * 40.0,
        z=forward_vector.z * 40.0
    )
    spawn_rotation = static_transform.rotation
    static_transform = carla.Transform(spawn_location, spawn_rotation)

    try:
        static_vehicle = world.spawn_actor(static_vehicle_bp, static_transform)
        static_vehicle.set_autopilot(False)  # Ensure it's static
        static_vehicle.apply_control(carla.VehicleControl(hand_brake=True))  # Park it
        print("Static vehicle spawned 5m ahead of ego.")
    except RuntimeError as e:
        print(f"Failed to spawn static vehicle: {e}")


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
            control.throttle = 0.5 + (0.5 if keys[pygame.K_w] else 0.0)
            control.brake = 1.0 if keys[pygame.K_s] else 0.0
            control.steer = -0.2 if keys[pygame.K_a] else (0.2 if keys[pygame.K_d] else 0.0)
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

                image = frame
                target_pt = np.array([90, 0], dtype=np.float32) # target
                speed = np.array([speed], dtype=np.float32)
                # Apply preprocessing
                augmented = transform(image=image)
                input_tensor = augmented["image"].unsqueeze(0).to(device)
                target_pt_tensor = torch.tensor(target_pt).unsqueeze(0).to(device)
                speed_tensor = torch.tensor(speed).to(device)

                with torch.no_grad():
                    semantic_pred, wp_pred, speed_pred, brake_pred = model(input_tensor, target_pt_tensor, speed_tensor)

                # Semantic mask processing (same as before)
                semantic_mask = torch.argmax(semantic_pred.squeeze(), dim=0).cpu().numpy()

                # Convert to RGB image
                color_map = {
                    0: [0, 0, 0],          # Background - Black
                    1: [0, 0, 255],        # Road - Blue
                    2: [0, 255, 0],        # Cars - Green
                    3: [255, 0, 0],        # Lane Marks/Obstacles - Red
                    4: [255, 255, 0],      # Pedestrians - Yellow
                    5: [255, 0, 255],      # Traffic Lights - Magenta
                    6: [0, 255, 255],      # (Optional: if you expand NUM_CLASSES) - Cyan
                }

                # Convert mask to color image (256x256x3)
                semantic_color = np.zeros((semantic_mask.shape[0], semantic_mask.shape[1], 3), dtype=np.uint8)
                for class_id, color in color_map.items():
                    semantic_color[semantic_mask == class_id] = color

                # Resize to match main display size (1024×512)
                semantic_color_resized = cv2.resize(semantic_color, (896, 512), interpolation=cv2.INTER_NEAREST)

                # Convert to Pygame surface and display next to RGB camera
                semantic_surface = pygame.surfarray.make_surface(np.flipud(np.rot90(semantic_color_resized)))
                display.blit(semantic_surface, (1024, 0))  # Right half of the screen

                waypoints = wp_pred.squeeze().cpu().numpy().reshape(-1, 2)
                # Draw in CARLA world
                world_waypoints = get_world_waypoints(vehicle, waypoints)

                for i, (x, y) in enumerate(world_waypoints):
                    loc = carla.Location(x=x, y=y, z=0.5)
                    world.debug.draw_point(loc, size=0.15, color=carla.Color(0, 255, 0), life_time=0.07, persistent_lines=False)
                    label = f"W{i+1}"
                    world.debug.draw_string(loc + carla.Location(z=0.3), label, draw_shadow=False,
                                            color=carla.Color(255, 255, 0), life_time=0.1, persistent_lines=False)
                    
                                # Brake prediction (sigmoid output between 0 and 1)
                brake_value = brake_pred.item()  # scalar value between 0 and 1

                # Predicted speed output (from model) - adjust if needed for units
                target_speed_value = speed_pred.item()

                # Draw brake prediction and target speed text
                # brake_text = font.render(f"Brake Pred: {brake_value:.2f}", True, (255, 255, 255))
                # target_speed_text = font.render(f"Target Speed: {target_speed_value:.2f}", True, (255, 255, 255))

                # display.blit(brake_text, (20, 60))          # Slightly below the current speed text
                # display.blit(target_speed_text, (20, 100))  # Below brake prediction text

                pygame.display.flip()


    finally:
        camera.stop()
        vehicle.destroy()
        pygame.quit()
        print("Cleaned up.")

if __name__ == '__main__':
    main()
