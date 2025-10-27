import os
import cv2
import torch
import numpy as np
import pygame
import time
import carla
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch import nn
import torch.nn.functional as F
import timm

# ----------------------------
# Config
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

NUM_CLASSES = 7
NUM_DEPTH_BINS = 8  # 和训练时一致
INPUT_H, INPUT_W = 224, 224

# ✅ 改成你训练后保存的模型路径
CKPT_PATH = "/home/ali_carla/Desktop/Ali/Leader_Board_2/CARLA_Leaderboard_2.0/CARLA_Leaderboard_20/PythonAPI/examples/Transformer_Full_result_20250807_140205/transformer_corrected_VehicleNNController_NAG.pth"

# 显示窗口：原图 + 语义分割 + 深度 可视化
WIN_W, WIN_H = 1024 * 3, 512

# Albumentations：和训练一致
transform = A.Compose([
    A.Resize(INPUT_H, INPUT_W),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

# ----------------------------
# 模型定义（与训练一致）
# ----------------------------
class UNetDecoderWithSkip(nn.Module):
    def __init__(self, encoder_channels, decoder_channels):
        super().__init__()
        self.num_stages = len(decoder_channels)
        self.up_convs = nn.ModuleList()
        self.fuse_convs = nn.ModuleList()
        in_channels = encoder_channels[-1]
        for i in range(self.num_stages):
            self.up_convs.append(nn.ConvTranspose2d(in_channels, decoder_channels[i], kernel_size=2, stride=2))
            skip_channels = encoder_channels[-(i + 2)] if (i + 2) <= len(encoder_channels) else 0
            self.fuse_convs.append(nn.Sequential(
                nn.Conv2d(decoder_channels[i] + skip_channels, decoder_channels[i], kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ))
            in_channels = decoder_channels[i]

    def forward(self, features):
        x = features[-1]
        for i in range(self.num_stages):
            x = self.up_convs[i](x)
            if i + 2 <= len(features):
                skip = features[-(i + 2)]
                if x.shape[2:] != skip.shape[2:]:
                    skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
                x = torch.cat([x, skip], dim=1)
            x = self.fuse_convs[i](x)
        return x

class SwinMultiTaskModel(nn.Module):
    def __init__(self, swin_name="swin_tiny_patch4_window7_224", num_classes=7, num_depth_bins=8,
                 num_waypoints=30, d_model=128, nhead=8, num_layers=3):
        super().__init__()
        self.encoder = timm.create_model(swin_name, pretrained=False, features_only=True)
        enc_channels = [f["num_chs"] for f in self.encoder.feature_info]
        enc_out_ch = enc_channels[-1]

        self.decoder = UNetDecoderWithSkip(
            encoder_channels=[96, 192, 384, 768],
            decoder_channels=[256, 128, 64]
        )
        self.segmentation_head = nn.Conv2d(64, num_classes, kernel_size=1)
        self.depth_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, num_depth_bins, kernel_size=1)
        )

        self.input_proj = nn.Conv2d(enc_out_ch, d_model, 1)
        self.visual_fc = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(), nn.Linear(enc_out_ch, d_model), nn.ReLU())
        self.meta_fc = nn.Sequential(nn.Linear(3, d_model), nn.ReLU())

        self.query_embed = nn.Parameter(torch.randn(num_waypoints, d_model))
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.waypoint_head = nn.Linear(d_model, 2)
        self.speed_head = nn.Linear(d_model, 1)
        self.brake_head = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())

    def forward(self, x, target_point, current_speed):
        feats = self.encoder(x)                # list of 4 tensors (B, H, W, C)
        feats = [f.permute(0, 3, 1, 2) for f in feats]  # -> (B, C, H, W)
        dec_out = self.decoder(feats)

        seg_out = self.segmentation_head(dec_out)
        seg_out = F.interpolate(seg_out, size=(INPUT_H, INPUT_W), mode='bilinear', align_corners=False)

        depth_out = self.depth_head(dec_out)
        depth_out = F.interpolate(depth_out, size=(INPUT_H, INPUT_W), mode='bilinear', align_corners=False)

        last_feat = feats[-1]
        visual_feat = self.visual_fc(last_feat)
        meta = self.meta_fc(torch.cat([target_point, current_speed.unsqueeze(1)], dim=1))

        memory = self.input_proj(last_feat).flatten(2).permute(0, 2, 1)  # (B, N, d)
        memory = torch.cat([meta.unsqueeze(1), memory], dim=1)
        queries = self.query_embed.unsqueeze(0).expand(x.size(0), -1, -1)
        decoded = self.transformer_decoder(tgt=queries, memory=memory)
        waypoints = self.waypoint_head(decoded)  # (B, 30, 2)

        control_feat = visual_feat + meta
        speed = self.speed_head(control_feat)
        brake = self.brake_head(control_feat)

        return seg_out, depth_out, waypoints, speed, brake

# ----------------------------
# 工具函数
# ----------------------------
def get_world_waypoints(vehicle, relative_waypoints):
    world_waypoints = []
    transform = vehicle.get_transform()
    yaw = np.radians(transform.rotation.yaw)
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    for x, y in relative_waypoints:
        wx = transform.location.x + x * cos_yaw - y * sin_yaw
        wy = transform.location.y + x * sin_yaw + y * cos_yaw
        world_waypoints.append((wx, wy))
    return world_waypoints

def to_surface(arr_rgb):
    # arr_rgb: HxWx3 RGB uint8
    return pygame.surfarray.make_surface(np.flipud(np.rot90(arr_rgb)))

# ----------------------------
# 载入模型
# ----------------------------
# 1. 创建模型
model = SwinMultiTaskModel(
    num_classes=NUM_CLASSES,
    num_depth_bins=NUM_DEPTH_BINS
).to(device)

# 2. 加载权重
state = torch.load(CKPT_PATH, map_location=device)

# 如果保存的时候是 {'state_dict': ...} 这种结构，先取出里面的 'state_dict'
if isinstance(state, dict) and 'state_dict' in state:
    state = state['state_dict']

# 如果保存的时候用了 DataParallel，需要去掉 'module.' 前缀
state = {k.replace('module.', ''): v for k, v in state.items()}

# 3. 允许缺少那些非训练参数（比如 relative_position_index / attn_mask）
missing, unexpected = model.load_state_dict(state, strict=False)
print("权重已加载（strict=False）")
print("缺少的键:", missing)
print("多余的键:", unexpected)

# 4. 切到 eval 模式
model.eval()
print("Loaded SwinMultiTaskModel weights")

# ----------------------------
# 运行 CARLA + 显示
# ----------------------------
def main():
    pygame.init()
    display = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("SwinMultiTaskModel Test (RGB / SEG / DEPTH)")
    clock = pygame.time.Clock()

    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    client.load_world("Town04")
    time.sleep(2)
    world = client.get_world()

    bp = world.get_blueprint_library()
    veh_bp = bp.find("vehicle.lincoln.mkz_2017")
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(veh_bp, spawn_point)

    cam_bp = bp.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", "1024")
    cam_bp.set_attribute("image_size_y", "512")
    cam_bp.set_attribute("fov", "90")
    cam_transform = carla.Transform(carla.Location(x=-1.5, z=2.0))
    camera = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)

    image_data = {"image": None}
    def on_image(img):
        arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape((img.height, img.width, 4))
        image_data["image"] = arr[:, :, :3]  # RGB
    camera.listen(on_image)

    font = pygame.font.Font(None, 36)

    try:
        while True:
            clock.tick(30)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

            # 手动控制（和你之前一致，可按需改）
            keys = pygame.key.get_pressed()
            ctrl = carla.VehicleControl()
            ctrl.throttle = 0.5 + (0.5 if keys[pygame.K_w] else 0.0)
            ctrl.brake = 1.0 if keys[pygame.K_s] else 0.0
            ctrl.steer = -0.2 if keys[pygame.K_a] else (0.2 if keys[pygame.K_d] else 0.0)
            ctrl.hand_brake = keys[pygame.K_SPACE]
            vehicle.apply_control(ctrl)

            if image_data["image"] is None:
                pygame.display.flip()
                continue

            frame = image_data["image"]                 # 1024x512x3 RGB
            rgb_surface = to_surface(frame)
            display.blit(rgb_surface, (0, 0))

            # 当前速度（km/h）
            v = vehicle.get_velocity()
            speed = 3.6 * (v.x**2 + v.y**2 + v.z**2) ** 0.5

            # 预处理送入模型
            aug = transform(image=cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # 和训练一致做 Normalize/Resize
            inp = aug["image"].unsqueeze(0).to(device)

            # 目标点 + 当前速度（和训练一致，target_point 可按需设置）
            target_pt = torch.tensor([[90.0, 0.0]], dtype=torch.float32, device=device)
            cur_speed = torch.tensor([speed], dtype=torch.float32, device=device)

            with torch.no_grad():
                seg_pred, depth_pred, wp_pred, spd_pred, brk_pred = model(inp, target_pt, cur_speed)

            # ========== 语义可视化 ==========
            seg_lbl = torch.argmax(seg_pred.squeeze(0), dim=0).cpu().numpy()  # 224x224
            color_map = {
                0: [0, 0, 0],
                1: [0, 0, 255],      # Road
                2: [0, 255, 0],      # Cars
                3: [255, 0, 0],      # Lane/Obstacle
                4: [255, 255, 0],    # Pedestrian
                5: [255, 0, 255],    # Traffic light
                6: [0, 255, 255],
            }
            seg_rgb = np.zeros((seg_lbl.shape[0], seg_lbl.shape[1], 3), dtype=np.uint8)
            for k, c in color_map.items():
                seg_rgb[seg_lbl == k] = c
            seg_rgb = cv2.resize(seg_rgb, (1024, 512), interpolation=cv2.INTER_NEAREST)
            seg_surface = to_surface(seg_rgb)
            display.blit(seg_surface, (1024, 0))

            # ========== 深度可视化（离散 bins）==========
            depth_cls = torch.argmax(depth_pred.squeeze(0), dim=0).cpu().numpy()  # 224x224, 0..NUM_DEPTH_BINS-1
            # 映射到 0..255 再上色
            depth_255 = (depth_cls.astype(np.float32) / (NUM_DEPTH_BINS - 1) * 255.0).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_255, cv2.COLORMAP_PLASMA)
            depth_color = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)
            depth_color = cv2.resize(depth_color, (1024, 512), interpolation=cv2.INTER_NEAREST)
            depth_surface = to_surface(depth_color)
            display.blit(depth_surface, (1024 * 2, 0))

            # ========== Waypoints 画到 CARLA 世界 ==========
            waypoints = wp_pred.squeeze(0).cpu().numpy()  # (30,2)
            world_wps = get_world_waypoints(vehicle, waypoints)
            for i, (wx, wy) in enumerate(world_wps):
                loc = carla.Location(x=wx, y=wy, z=0.5)
                world.debug.draw_point(loc, size=0.12, color=carla.Color(0, 255, 0), life_time=0.07, persistent_lines=False)
                world.debug.draw_string(loc + carla.Location(z=0.3), f"W{i+1}", draw_shadow=False,
                                        color=carla.Color(255, 255, 0), life_time=0.1, persistent_lines=False)

            # ========== 文本信息 ==========
            txt_speed = font.render(f"Speed: {speed:.1f} km/h", True, (255, 255, 255))
            display.blit(txt_speed, (20, 20))

            brake_value = float(brk_pred.item())
            target_speed_value = float(spd_pred.item())
            txt_brake = font.render(f"Brake Pred: {brake_value:.2f}", True, (255, 255, 255))
            txt_tspd = font.render(f"Target Speed: {target_speed_value:.2f}", True, (255, 255, 255))
            display.blit(txt_brake, (20, 60))
            display.blit(txt_tspd, (20, 100))

            pygame.display.flip()

    finally:
        camera.stop()
        vehicle.destroy()
        pygame.quit()
        print("Cleaned up.")

if __name__ == "__main__":
    main()