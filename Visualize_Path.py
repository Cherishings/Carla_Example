import torch
import numpy as np
import pygame
import cv2
import carla
import time
import math
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model_inference import VehicleNNController


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing transform (same as used during training)
transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VehicleNNController().to(device)
model.load_state_dict(torch.load("VehicleNNController_NAG.pth"))
model.eval()

def preprocess_image(image):
    """Preprocess image to match the training transformation."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    augmented = transform(image=image)
    image_tensor = augmented['image']
    return image_tensor.unsqueeze(0)  # Add batch dimension

def draw_waypoints(world, waypoints, z=0.5, life_time=60.0):
    """Draw the predicted waypoints on the CARLA world."""
    for i, wp in enumerate(waypoints):
        location = carla.Location(x=wp[0], y=wp[1], z=z)
        world.debug.draw_point(location, size=0.2, color=carla.Color(0, 255, 0), life_time=life_time, persistent_lines=True)
        label = f"W{i}"
        world.debug.draw_string(location + carla.Location(z=0.5), label, draw_shadow=False, color=carla.Color(0, 255, 0), life_time=life_time, persistent_lines=True)

def capture_image_from_car(camera):
    """Capture the current image from the camera attached to the vehicle."""
    image = camera.listen(lambda image: process_image(image))
    return image
def process_image(image, current_speed):
    """Process image from camera and feed to the model."""
    # Convert the raw data to a numpy array
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))  # RGBA format
    array = array[:, :, :3]  # Take RGB part

    # Preprocess the image for the model
    image_tensor = preprocess_image(array)

    # Set target point (just an example for now)
    # Set target point (example, shape [1, 2])
    target_point = torch.tensor([[10.0, 0.0]], dtype=torch.float32).to(device)  # Ensure it's on the correct device

    # Ensure current_speed is a 2D tensor [1, 1]
    current_speed_tensor = torch.tensor([current_speed], dtype=torch.float32).to(device)  # Move to the same device as target_point
    print(f"current_speed_tensor.shape before unsqueeze: {current_speed_tensor.shape}")

    # If you need to match target_point's shape for concatenation:
    current_speed_tensor = current_speed_tensor.unsqueeze(0)  # shape: [1, 1]
    print(f"current_speed_tensor.shape after unsqueeze: {current_speed_tensor.shape}")
    print(f"target_point.shape: {target_point.shape}")

    # Ensure both tensors are on the same device
    target_point = target_point.to(device)

    # Concatenate the target_point and current_speed_tensor
    meta = torch.cat([target_point, current_speed_tensor], dim=1)  # Concatenate along dimension 1
    print(f"meta.shape: {meta.shape}")

    # Model prediction
    with torch.no_grad():
        semantic_out, wp_pred, speed_pred, brake_pred = model(
            image_tensor.to(device), meta
        )

    # Extract predicted waypoints
    waypoints = wp_pred.squeeze().cpu().numpy().reshape(-1, 2)  # Reshape to (5, 2) for (x, y) pairs

    # Draw the waypoints on the world
    draw_waypoints(world, waypoints)


def simulate():
    # Initialize Carla Client and World
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # Get vehicle blueprint and spawn
    blueprint = world.get_blueprint_library().filter('vehicle.lincoln.mkz_2017')[0]
    spawn_points = world.get_map().get_spawn_points()
    vehicle = world.spawn_actor(blueprint, spawn_points[0])

    # Camera Setup
    camera_blueprint = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_transform = carla.Transform(carla.Location(x=-1.5, y=0, z=2))  # Behind the vehicle
    camera = world.spawn_actor(camera_blueprint, camera_transform, attach_to=vehicle)
    
    pygame.init()
    display = pygame.display.set_mode((800, 600), pygame.HWSURFACE | pygame.DOUBLEBUF)
    velocity = vehicle.get_velocity()
    current_speed = math.sqrt(velocity.x**2 + velocity.y**2)
    camera.listen(lambda image: process_image(image,current_speed))

    # Vehicle control setup
    carla_control = carla.VehicleControl()

    # Main loop
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                vehicle.destroy()
                pygame.quit()
                return
        
        # Read keys for manual control
        keys = pygame.key.get_pressed()
        carla_control.steer = 0
        carla_control.throttle = 0
        carla_control.brake = 0
        carla_control.hand_brake = False

        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            carla_control.steer = -1.0
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            carla_control.steer = 1.0
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            carla_control.throttle = 1.0
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            carla_control.brake = 1.0

        vehicle.apply_control(carla_control)

        time.sleep(0.05)

simulate()