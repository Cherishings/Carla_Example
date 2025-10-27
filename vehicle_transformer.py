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
    def __init__(self, num_classes=7, hidden_size=64, num_waypoints=5, debug=False):
        super(VehicleTransformerMAnet, self).__init__()
        self.debug = debug

        self.manet = smp.MAnet(
            encoder_name="mit_b2",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.visual_fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU()
        )

        self.meta_input_net = nn.Sequential(
            nn.Linear(3, 20),
            nn.ReLU(),
            nn.Linear(20, 8),
            nn.ReLU()
        )

        self.depth_head = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 8, kernel_size=1)
        )

        self.gru = nn.GRU(input_size=64 + 8 + 2 + 2, hidden_size=hidden_size, batch_first=True)
        self.gru_out = nn.Linear(hidden_size, 2)

        self.output_speed = nn.Linear(64 + 8, 1)
        self.output_brake = nn.Linear(64 + 8, 1)

    def forward(self, x_img, target_point, current_speed, teacher_forcing_ratio=0.0):
        waypoints_gt = getattr(self, '_last_gt_waypoints', None)
        features = self.manet.encoder(x_img)
        last_feat = features[-1]
        B, C, H, W = last_feat.shape
        pooled_feat = self.avg_pool(last_feat)
        flat_feat = pooled_feat.view(B, -1)
        img_feat = self.visual_fc(flat_feat)

        decoder_out = self.manet.decoder(*features)
        seg_out = self.manet(x_img)
        depth_logits = self.depth_head(decoder_out)

        meta = torch.cat([target_point, current_speed.unsqueeze(1)], dim=1)
        meta_feat = self.meta_input_net(meta)
        context = torch.cat([img_feat, meta_feat], dim=1)

        prev_wp = torch.zeros((x_img.size(0), 2), device=x_img.device)
        waypoints = []
        hidden = None

        for t in range(5):
            input_step = torch.cat([context, prev_wp, target_point], dim=1).unsqueeze(1)
            out, hidden = self.gru(input_step, hidden)
            pred_wp = self.gru_out(out[:, -1, :])
            prev_wp = pred_wp
            waypoints.append(pred_wp)
        waypoints = torch.stack(waypoints, dim=1)

        speed = self.output_speed(torch.cat([img_feat, meta_feat], dim=1))
        brake = torch.sigmoid(self.output_brake(torch.cat([img_feat, meta_feat], dim=1)))

        return seg_out, waypoints, speed, brake, depth_logits

# ----------------------------
# Model Load
# ----------------------------
model_path = "/home/ali_carla/Desktop/Ali/Leader_Board_2/CARLA_Leaderboard_2.0/CARLA_Leaderboard_20/PythonAPI/examples/Transformer_GRU_result_20250730_141837/transformer_corrected_VehicleNNController_NAG.pth"
model = VehicleTransformerMAnet(num_classes=7).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("Model loaded from:", model_path)

# ----------------------------
# Image Transform
# ----------------------------
transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

# ----------------------------
# Color Mapping
# ----------------------------
color_map = {
    0: [0, 0, 0],
    1: [0, 0, 255],
    2: [0, 255, 0],
    3: [255, 0, 0],
    4: [255, 255, 0],
    5: [255, 0, 255],
    6: [0, 255, 255],
}

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
    display = pygame.display.set_mode((960 , 720))
    pygame.display.set_caption("RGB | Segmentation | Depth")
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
                seg_out, wp_out, speed_pred, brake_pred, depth_out = model(input_tensor, target_pt, speed_tensor)

            # ---------- 后处理 ----------
            seg_mask = torch.argmax(seg_out.squeeze(), dim=0).cpu().numpy()
            seg_img = np.zeros((seg_mask.shape[0], seg_mask.shape[1], 3), dtype=np.uint8)
            for cid, color in color_map.items():
                seg_img[seg_mask == cid] = color

            depth_mask = torch.argmax(depth_out.squeeze(), dim=0).cpu().numpy()
            depth_vis = cv2.applyColorMap((depth_mask * (255 // 7)).astype(np.uint8), cv2.COLORMAP_JET)

            # ---------- Resize 展示 ----------
            rgb_vis = cv2.resize(rgb_img, (960, 360))
            seg_img = cv2.resize(seg_img, (480, 360))
            depth_vis = cv2.resize(depth_vis, (480, 360))

            # ---------- 坐标变换 ----------
            waypoints = wp_out.squeeze().cpu().numpy().reshape(-1, 2)
            world_waypoints = get_world_waypoints(vehicle, waypoints)
            for i, (x, y) in enumerate(world_waypoints):
                loc = carla.Location(x=x, y=y, z=0.5)
                world.debug.draw_point(loc, size=0.15, color=carla.Color(0, 255, 0), life_time=0.07)
                world.debug.draw_string(loc + carla.Location(z=0.3), f"W{i+1}",
                                        color=carla.Color(255, 255, 0), life_time=0.1)

            # ---------- Pygame 显示 ----------
            rgb_surface = pygame.surfarray.make_surface(np.flipud(np.rot90(rgb_vis)))
            seg_surface = pygame.surfarray.make_surface(np.flipud(np.rot90(seg_img)))
            depth_surface = pygame.surfarray.make_surface(np.flipud(np.rot90(depth_vis)))

            display.blit(rgb_surface, (0, 0))
            display.blit(seg_surface, (0, 360))
            display.blit(depth_surface, (480, 360))

            display.blit(font.render(f"Speed: {speed:.1f} km/h", True, (255, 255, 255)), (20, 10))

            pygame.display.flip()
    finally:
        camera.stop()
        vehicle.destroy()
        pygame.quit()
        print("Cleaned up.")

if __name__ == "__main__":
    main()