import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import pygame
import carla
import random
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ----------------------------
# Device
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------
# Model Definition
# ----------------------------
class VehicleTransformerMAnet(nn.Module):
    def __init__(self, num_classes=7, num_waypoints=30, d_model=128, nhead=8, num_layers=3):
        super(VehicleTransformerMAnet, self).__init__()
        self.num_waypoints = num_waypoints

        # 只加载 encoder（你加载的是这个部分）
        self.encoder = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=num_classes
        ).encoder

        self.visual_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, d_model),
            nn.ReLU()
        )

        self.input_proj = nn.Conv2d(512, d_model, kernel_size=1)

        self.meta_fc = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU()
        )

        self.query_embed = nn.Parameter(torch.randn(num_waypoints, d_model))
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.waypoint_head = nn.Linear(d_model, 2)

    def forward(self, x_img, target_point, current_speed):
        features = self.encoder(x_img)  # No decoder!
        last_feat = features[-1]       # [B, 512, H, W]

        visual_feat = self.visual_fc(last_feat)

        meta_input = torch.cat([target_point, current_speed.unsqueeze(1)], dim=1)
        meta_feat = self.meta_fc(meta_input)

        x = self.input_proj(last_feat)  # [B, d_model, H, W]
        memory = x.flatten(2).permute(0, 2, 1)
        memory = torch.cat([meta_feat.unsqueeze(1), memory], dim=1)

        queries = self.query_embed.unsqueeze(0).expand(x_img.size(0), -1, -1)
        decoded = self.transformer_decoder(tgt=queries, memory=memory)
        waypoints = self.waypoint_head(decoded)

        return waypoints

# ----------------------------
# Model Load
# ----------------------------
# model_path = "/home/ali_carla/Desktop/Ali/Leader_Board_2/CARLA_Leaderboard_2.0/CARLA_Leaderboard_20/PythonAPI/examples/Transformer_Full_result_20250803_233451/transformer_corrected_VehicleNNController_NAG.pth"
# model_path = "/home/ali_carla/Desktop/Ali/Leader_Board_2/CARLA_Leaderboard_2.0/CARLA_Leaderboard_20/PythonAPI/examples/Transformer_Full_result_20250803_233451/fine_tuned_waypoint_model.pth"
model_path = "/home/ali_carla/Desktop/Ali/Leader_Board_2/CARLA_Leaderboard_2.0/CARLA_Leaderboard_20/PythonAPI/examples/Transformer_Full_result_20250806_025436/transformer_best_model.pth"
# model_path = "Corrected_VehicleNNController_NAG.pth"
model = VehicleTransformerMAnet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
model.eval()
print("✅ Model loaded from:", model_path)

# ----------------------------
# Image Transform
# ----------------------------
transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

# ----------------------------
# Waypoints Conversion
# ----------------------------
def get_world_waypoints(vehicle, relative_waypoints):
    transform = vehicle.get_transform()
    yaw = np.radians(transform.rotation.yaw)
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)

    world_waypoints = []
    for x, y in relative_waypoints:
        world_x = transform.location.x + x * cos_yaw - y * sin_yaw
        world_y = transform.location.y + x * sin_yaw + y * cos_yaw
        world_waypoints.append((world_x, world_y))
    return world_waypoints

# ----------------------------
# Main Loop
# ----------------------------
def main():
    pygame.init()
    display = pygame.display.set_mode((960, 540))
    pygame.display.set_caption("RGB + Waypoints Only")
    clock = pygame.time.Clock()

    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    client.load_world("Town04")
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    vehicle_bp = blueprint_library.find("vehicle.lincoln.mkz_2017")
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '1024')
    camera_bp.set_attribute('image_size_y', '512')
    camera_bp.set_attribute('fov', '90')
    camera_transform = carla.Transform(carla.Location(x=-1.5, z=2.0))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    image_data = {"image": None}
    def process_image(image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        image_data["image"] = arr[:, :, :3]
    camera.listen(process_image)

    font = pygame.font.Font(None, 36)
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

            if image_data["image"] is None:
                continue

            rgb_img = cv2.cvtColor(image_data["image"], cv2.COLOR_BGR2RGB)

            # ---------- 模型推理 ----------
            velocity = vehicle.get_velocity()
            speed = 3.6 * np.linalg.norm([velocity.x, velocity.y, velocity.z])

            aug = transform(image=rgb_img)
            input_tensor = aug["image"].unsqueeze(0).to(device)
            target_pt = torch.tensor([[90.0, 0.0]], dtype=torch.float32).to(device)
            speed_tensor = torch.tensor([speed], dtype=torch.float32).to(device)

            with torch.no_grad():
                wp_out = model(input_tensor, target_pt, speed_tensor)

            # ---------- 坐标变换 + 可视化 ----------
            waypoints = wp_out.squeeze().cpu().numpy().reshape(-1, 2)
            world_waypoints = get_world_waypoints(vehicle, waypoints)
            for i, (x, y) in enumerate(world_waypoints):
                loc = carla.Location(x=x, y=y, z=0.5)
                world.debug.draw_point(loc, size=0.15, color=carla.Color(0, 255, 0), life_time=0.07)
                world.debug.draw_string(loc + carla.Location(z=0.3), f"W{i+1}",
                                        color=carla.Color(255, 255, 0), life_time=0.1)

            # ---------- 显示 ----------
            rgb_vis = cv2.resize(rgb_img, (960, 540))
            rgb_surface = pygame.surfarray.make_surface(np.flipud(np.rot90(rgb_vis)))
            display.blit(rgb_surface, (0, 0))
            display.blit(font.render(f"Speed: {speed:.1f} km/h", True, (255, 255, 255)), (20, 10))
            pygame.display.flip()

    finally:
        camera.stop()
        vehicle.destroy()
        pygame.quit()
        print("Cleaned up.")

if __name__ == "__main__":
    main()
