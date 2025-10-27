# -*- coding: utf-8 -*-
"""
测试推理（与训练像素严格一致版）+ 可视化增强：
- 推理前执行与训练一致的裁剪：上 384 × 中 1024
- 顶部RGB可显示裁剪区域，或在整帧上用黄框标出实际使用区域
"""

import os, sys, math, time
import numpy as np
import cv2
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
import carla

# ============================= 可调参数 =============================
CKPT_PATH = "/home/ali_carla/Desktop/Ali/Leader_Board_2/CARLA_Leaderboard_2.0/CARLA_Leaderboard_20/PythonAPI/examples/Swin_WP5_with2DPos_20250818_030444/swin_wp5_best.pth"  # ← 改成你的权重

# —— 相机与窗口（显存紧张可降；务必 >= 1024×384 才能做裁剪）——
CAM_W, CAM_H = 1280, 720        # 也可设为 (1024, 384)
FOV_DEG = 90.0                  # 与训练一致
WIN_W, WIN_H = 960, 720         # 窗口总大小（保持 960x720 排版）
TILE_W, TILE_H = 480, 360       # 底部左右两块面板的尺寸

# —— 显示选项 —— 
SHOW_TOP_CROPPED = False         # True: 顶部显示裁剪区域；False: 显示整帧并画黄框
DRAW_CROP_BOX_ON_FULL = False    # 仅当 SHOW_TOP_CROPPED=False 时生效

# —— 路点绘制与平滑 —— 
Z_OFFSET = 1.2
EMA_DECAY = 0.85
SAMPLES_PER_SEG = 24
USE_AMP = False

# —— 模型/数据设定（需与训练一致） —— 
INPUT_W, INPUT_H = 224, 224
NUM_CLASSES = 7
NUM_DEPTH_BINS = 8
NUM_WAYPOINTS = 5
TARGET_POINT = (90.0, 0.0)      # 训练时使用的引导点（自车系，单位m）

# 裁剪参数：与训练完全一致（上 384 × 中 1024）
CROP_H, CROP_W = 384, 1024

# ============================= 预处理 =============================
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Info] Using device: {device}")

transform = A.Compose([
    A.Resize(INPUT_H, INPUT_W),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

# ============================= 模型（与训练一致） =============================
class UNetDecoderWithSkip(nn.Module):
    def __init__(self, encoder_channels, decoder_channels):
        super().__init__()
        self.num_stages = len(decoder_channels)
        self.up_convs = nn.ModuleList()
        self.fuse_convs = nn.ModuleList()
        in_ch = encoder_channels[-1]
        for i in range(self.num_stages):
            self.up_convs.append(nn.ConvTranspose2d(in_ch, decoder_channels[i], 2, 2))
            skip_ch = encoder_channels[-(i+2)] if (i+2) <= len(encoder_channels) else 0
            self.fuse_convs.append(nn.Sequential(
                nn.Conv2d(decoder_channels[i] + skip_ch, decoder_channels[i], 3, padding=1),
                nn.ReLU(inplace=True)
            ))
            in_ch = decoder_channels[i]
    def forward(self, feats):
        x = feats[-1]
        for i in range(self.num_stages):
            x = self.up_convs[i](x)
            if i + 2 <= len(feats):
                skip = feats[-(i+2)]
                if x.shape[2:] != skip.shape[2:]:
                    skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
                x = torch.cat([x, skip], dim=1)
            x = self.fuse_convs[i](x)
        return x

def build_2d_sincos_pos_embed(d_model, h, w, temperature=10000.0, device=None):
    if device is None:
        device = torch.device("cpu")
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing='ij'
    )
    xx = xx.reshape(-1).float()
    yy = yy.reshape(-1).float()
    half_dim = d_model // 4
    if half_dim < 1:
        raise ValueError("d_model should be >= 4 for 2D sincos pos embed.")
    omega = torch.arange(half_dim, device=device, dtype=torch.float32)
    omega = 1.0 / (temperature ** (omega / max(1, half_dim - 1)))
    x_sin = torch.sin(xx[:, None] * omega[None, :])
    x_cos = torch.cos(xx[:, None] * omega[None, :])
    y_sin = torch.sin(yy[:, None] * omega[None, :])
    y_cos = torch.cos(yy[:, None] * omega[None, :])
    pos = torch.cat([x_sin, x_cos, y_sin, y_cos], dim=1)
    if pos.shape[1] < d_model:
        pos = torch.cat([pos, torch.zeros((pos.shape[0], d_model - pos.shape[1]), device=device)], dim=1)
    elif pos.shape[1] > d_model:
        pos = pos[:, :d_model]
    return pos

class SwinMultiTaskModel(nn.Module):
    def __init__(self, swin_name="swin_tiny_patch4_window7_224",
                 num_classes=NUM_CLASSES, num_depth_bins=NUM_DEPTH_BINS,
                 num_waypoints=NUM_WAYPOINTS, d_model=128, nhead=8, num_layers=3, has_depth=True):
        super().__init__()
        self.has_depth = has_depth
        self.num_waypoints = num_waypoints
        self.d_model = d_model

        self.encoder = timm.create_model(swin_name, pretrained=False, features_only=True)
        enc_channels = [f["num_chs"] for f in self.encoder.feature_info]  # [96,192,384,768]
        enc_out_ch = enc_channels[-1]

        self.decoder = UNetDecoderWithSkip(encoder_channels=enc_channels, decoder_channels=[256,128,64])
        self.seg_head = nn.Conv2d(64, num_classes, 1)
        if self.has_depth:
            self.depth_head = nn.Sequential(nn.Conv2d(64,64,3,padding=1), nn.ReLU(), nn.Conv2d(64,num_depth_bins,1))

        self.input_proj = nn.Conv2d(enc_out_ch, d_model, 1)
        self.visual_fc  = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(), nn.Linear(enc_out_ch, d_model), nn.ReLU())
        self.meta_fc    = nn.Sequential(nn.Linear(3, d_model), nn.ReLU())

        self.query_embed = nn.Parameter(torch.randn(num_waypoints, d_model))
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, batch_first=True)
        self.tf_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.wp_head = nn.Linear(d_model, 2)

        self.speed_head = nn.Linear(d_model, 1)
        self.brake_head = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())

    def forward(self, x, target_point, current_speed):
        feats = self.encoder(x)                           # timm==1.0.16 返回 NHWC
        feats = [f.permute(0,3,1,2).contiguous() for f in feats]  # -> NCHW

        dec_out = self.decoder(feats)
        seg_out = self.seg_head(dec_out)
        seg_out = F.interpolate(seg_out, size=(INPUT_H, INPUT_W), mode='bilinear', align_corners=False)

        if self.has_depth:
            depth_out = self.depth_head(dec_out)
            depth_out = F.interpolate(depth_out, size=(INPUT_H, INPUT_W), mode='bilinear', align_corners=False)
        else:
            depth_out = None

        last = feats[-1]
        H, W = last.shape[2], last.shape[3]
        mem = self.input_proj(last).flatten(2).permute(0,2,1)   # [B,H*W,d]
        pos2d = build_2d_sincos_pos_embed(self.d_model, H, W, device=mem.device)
        mem = mem + pos2d.unsqueeze(0)

        q = self.query_embed.unsqueeze(0).expand(x.size(0), -1, -1)
        q_idx = torch.arange(self.num_waypoints, device=x.device, dtype=torch.float32)
        q_pos = torch.stack([torch.sin(q_idx/10.0), torch.cos(q_idx/10.0)], dim=1)
        if q_pos.shape[1] < self.d_model:
            q_pos = F.pad(q_pos, (0, self.d_model - q_pos.shape[1]))
        q = q + q_pos.unsqueeze(0)

        vis_g = self.visual_fc(last)
        meta  = self.meta_fc(torch.cat([target_point, current_speed.unsqueeze(1)], dim=1))
        control_feat = vis_g + meta

        dec = self.tf_decoder(tgt=q, memory=mem)
        waypoints = self.wp_head(dec)          # [B,5,2]
        speed = self.speed_head(control_feat)
        brake = self.brake_head(control_feat)
        return seg_out, depth_out, waypoints, speed, brake

def load_model(ckpt_path):
    model = SwinMultiTaskModel(has_depth=True).to(device)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("[Info] Weights loaded (strict=False)")
    if missing:   print("[Missing]:", missing)
    if unexpected:print("[Unexpected]:", unexpected)
    model.eval()
    return model

# ============================= 工具函数（关键：复制训练裁剪） =============================
def crop_array(img):
    """与训练完全一致：取上 384 × 中 1024 的区域。"""
    h, w = img.shape[:2]
    assert h >= CROP_H and w >= CROP_W, f"Frame too small: {(w,h)} needs >= {(CROP_W,CROP_H)}"
    side = (w - CROP_W) // 2
    return img[:CROP_H, side:w-side]

def adjust_pix_for_train_crop(pix_list, full_w, crop_w=CROP_W, crop_h=CROP_H):
    """
    将基于整帧坐标的像素点，映射到“训练裁剪坐标系”。
    不在裁剪区域内的点返回 None。
    """
    side = (full_w - crop_w) // 2
    out = []
    for p in pix_list:
        if p is None:
            out.append(None)
            continue
        u, v = p
        if 0 <= v < crop_h and side <= u < (side + crop_w):
            out.append((u - side, v))
        else:
            out.append(None)
    return out

def to_surface(arr_rgb_uint8):
    return pygame.surfarray.make_surface(np.flipud(np.rot90(arr_rgb_uint8)))

def ego_to_world(vehicle, rel_pts, z_offset=Z_OFFSET):
    tf = vehicle.get_transform()
    yaw = math.radians(tf.rotation.yaw)
    c, s = math.cos(yaw), math.sin(yaw)
    bx, by, bz = tf.location.x, tf.location.y, tf.location.z
    out = []
    for x, y in rel_pts:
        X = bx + x * c - y * s
        Y = by + x * s + y * c
        Z = bz + z_offset
        out.append((X, Y, Z))
    return out

def get_camera_K(camera):
    w = int(camera.attributes["image_size_x"])
    h = int(camera.attributes["image_size_y"])
    fov = float(camera.attributes["fov"])
    f = w / (2.0 * math.tan(fov * math.pi / 360.0))
    K = np.array([[f, 0, w/2.0],
                  [0, f, h/2.0],
                  [0, 0, 1]])
    return K

def world_to_camera_matrix(camera):
    return np.array(camera.get_transform().get_inverse_matrix())

def project_world_to_image(K, world_to_cam, pts_world):
    out = []
    for X, Y, Z in pts_world:
        pw = np.array([X, Y, Z, 1.0])
        pc = world_to_cam @ pw
        x, y, z = pc[0], pc[1], pc[2]
        if x <= 0:
            out.append(None)
            continue
        u = K[0,0] * (y / x) + K[0,2]
        v = K[1,1] * (-z / x) + K[1,2]
        out.append((int(u), int(v)))
    return out

class EMAFilter:
    def __init__(self, decay=EMA_DECAY):
        self.decay = decay
        self.state = None
    def update(self, pts_world):
        if self.state is None:
            self.state = list(pts_world)
        else:
            d = self.decay
            self.state = [
                (d*px + (1-d)*x, d*py + (1-d)*y, d*pz + (1-d)*z)
                for (px,py,pz),(x,y,z) in zip(self.state, pts_world)
            ]
        return self.state

def catmull_rom_spline(pts2d, samples_per_seg=SAMPLES_PER_SEG):
    if len(pts2d) < 4:
        return np.array(pts2d, dtype=np.int32)
    pts = np.array(pts2d, dtype=np.float32)
    out = []
    for i in range(1, len(pts)-2):
        p0, p1, p2, p3 = pts[i-1], pts[i], pts[i+1], pts[i+2]
        for t in np.linspace(0, 1, samples_per_seg, endpoint=False):
            t2, t3 = t*t, t*t*t
            a = 2*p1
            b = -p0 + p2
            c = 2*p0 - 5*p1 + 4*p2 - p3
            d = -p0 + 3*p1 - 3*p2 + p3
            pt = 0.5*(a + b*t + c*t2 + d*t3)
            out.append(pt)
    out.append(pts[-2])
    return np.array(out, dtype=np.int32)

def draw_waypoints_on_rgb(rgb_vis, pix_pts, draw_ids=True, smooth=True, color=(0,255,0)):
    h, w, _ = rgb_vis.shape
    in_pts = []
    for i, p in enumerate(pix_pts):
        if p is None:
            continue
        u, v = p
        if 0 <= u < w and 0 <= v < h:
            in_pts.append((u, v, i))
    for (u, v, i) in in_pts:
        cv2.rectangle(rgb_vis, (u-4, v-4), (u+4, v+4), color, thickness=-1)
        if draw_ids:
            cv2.putText(rgb_vis, f"W{i+1}", (u+6, v-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    if len(in_pts) >= 2:
        pts2d = [(u, v) for (u, v, _) in in_pts]
        if smooth and len(in_pts) >= 4:
            curve = catmull_rom_spline(pts2d, samples_per_seg=SAMPLES_PER_SEG)
            cv2.polylines(rgb_vis, [curve.reshape(-1,1,2)], False, color, 2, cv2.LINE_AA)
        else:
            arr = np.array(pts2d, dtype=np.int32).reshape(-1,1,2)
            cv2.polylines(rgb_vis, [arr], False, color, 2, cv2.LINE_AA)

# ============================= 主流程 =============================
def main():
    # ---- pygame ----
    pygame.init()
    display = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("RGB | Segmentation | Depth (train-aligned)")
    clock = pygame.time.Clock()
    font_big = pygame.font.Font(None, 48)
    font_small = pygame.font.Font(None, 28)

    # ---- CARLA ----
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    bp = world.get_blueprint_library()

    veh_bp = bp.find("vehicle.lincoln.mkz_2017")
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.try_spawn_actor(veh_bp, spawn_point) or world.spawn_actor(veh_bp, spawn_point)

    cam_bp = bp.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(CAM_W))
    cam_bp.set_attribute("image_size_y", str(CAM_H))
    cam_bp.set_attribute("fov", str(FOV_DEG))  # FOV 与训练一致
    cam_transform = carla.Transform(carla.Location(x=-1.5, z=2.0))
    camera = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)

    shared = {"rgb": None}
    def on_image(img):
        arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape((img.height, img.width, 4))
        rgb = arr[:, :, :3][:, :, ::-1].copy()  # BGRA -> RGB
        shared["rgb"] = rgb
    camera.listen(on_image)

    # 模型与 EMA
    model = load_model(CKPT_PATH)
    ema = EMAFilter(decay=EMA_DECAY)

    # 颜色表（语义可视化）
    color_map = {
        0: [0, 0, 0],
        1: [0, 0, 255],
        2: [0, 255, 0],
        3: [255, 0, 0],
        4: [255, 255, 0],
        5: [255, 0, 255],
        6: [0, 255, 255],
    }

    try:
        while True:
            clock.tick(30)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

            # 简单手动控制
            keys = pygame.key.get_pressed()
            ctrl = carla.VehicleControl()
            ctrl.throttle = 0.5 + (0.5 if keys[pygame.K_w] else 0.0)
            ctrl.brake = 1.0 if keys[pygame.K_s] else 0.0
            ctrl.steer = (-0.3 if keys[pygame.K_a] else (0.3 if keys[pygame.K_d] else 0.0))
            ctrl.hand_brake = keys[pygame.K_SPACE]
            vehicle.apply_control(ctrl)

            if shared["rgb"] is None:
                pygame.display.flip()
                continue

            frame = shared["rgb"]  # [CAM_H, CAM_W, 3]

            # —— 断言：确保可裁剪 —— 
            fh, fw = frame.shape[:2]
            assert fh >= CROP_H and fw >= CROP_W, f"Camera frame {(fw,fh)} too small for crop {(CROP_W,CROP_H)}"

            # 推理一致的裁剪
            frame_c = crop_array(frame)

            # 当前速度
            vel = vehicle.get_velocity()
            speed = 3.6 * (vel.x**2 + vel.y**2 + vel.z**2) ** 0.5

            # 预处理（与训练一致）
            aug = transform(image=frame_c)
            inp = aug["image"].unsqueeze(0).to(device)
            tgt_pt = torch.tensor([TARGET_POINT], dtype=torch.float32, device=device)
            cur_sp = torch.tensor([speed], dtype=torch.float32, device=device)

            # 推理
            with torch.no_grad():
                if USE_AMP:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        seg_logits, depth_logits, wp_pred, spd_pred, brk_pred = model(inp, tgt_pt, cur_sp)
                else:
                    seg_logits, depth_logits, wp_pred, spd_pred, brk_pred = model(inp, tgt_pt, cur_sp)

            # 语义
            seg_lbl = torch.argmax(seg_logits.squeeze(0), dim=0).cpu().numpy()
            seg_rgb = np.zeros((seg_lbl.shape[0], seg_lbl.shape[1], 3), dtype=np.uint8)
            for k, c in color_map.items():
                seg_rgb[seg_lbl == k] = c
            seg_img = cv2.resize(seg_rgb, (TILE_W, TILE_H), interpolation=cv2.INTER_NEAREST)

            # 深度
            if depth_logits is not None:
                depth_cls = torch.argmax(depth_logits.squeeze(0), dim=0).cpu().numpy()
                depth_255 = (depth_cls.astype(np.float32) / (NUM_DEPTH_BINS - 1) * 255.0).astype(np.uint8)
                depth_vis = cv2.applyColorMap(depth_255, cv2.COLORMAP_PLASMA)
                depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_BGR2RGB)
                depth_vis = cv2.resize(depth_vis, (TILE_W, TILE_H), interpolation=cv2.INTER_NEAREST)
            else:
                depth_vis = np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)

            # 预测路点 → 世界 → 像素（在整帧坐标系）
            wps_ego = wp_pred.squeeze(0).detach().cpu().numpy()  # (5,2)
            wps_world_raw = ego_to_world(vehicle, [(float(x), float(y)) for x, y in wps_ego], z_offset=Z_OFFSET)
            wps_world_smooth = ema.update(wps_world_raw)

            K = get_camera_K(camera)
            W2C = world_to_camera_matrix(camera)
            pix_full = project_world_to_image(K, W2C, wps_world_smooth)  # 基于整帧的像素坐标

            # 顶部 RGB 图
            if SHOW_TOP_CROPPED:
                # 显示与模型输入完全一致的区域
                rgb_vis = frame_c.copy()
                # 像素坐标也需要映射到裁剪坐标系
                pix_draw = adjust_pix_for_train_crop(pix_full, full_w=fw, crop_w=CROP_W, crop_h=CROP_H)
            else:
                # 显示整帧，并画出训练裁剪框
                rgb_vis = frame.copy()
                pix_draw = pix_full
                if DRAW_CROP_BOX_ON_FULL:
                    side = (fw - CROP_W) // 2
                    cv2.rectangle(rgb_vis, (side, 0), (side + CROP_W, CROP_H), (0, 255, 255), 2)  # 提示框

            draw_waypoints_on_rgb(rgb_vis, pix_draw, draw_ids=True, smooth=True, color=(0,255,0))

            # ---------- Pygame 显示 ----------
            rgb_surface = to_surface(rgb_vis)     # 顶部
            seg_surface = to_surface(seg_img)     # 左下
            depth_surface = to_surface(depth_vis) # 右下

            display.blit(rgb_surface, (0, 0))
            display.blit(seg_surface, (0, 360))
            display.blit(depth_surface, (480, 360))

            display.blit(font_big.render(f"Speed: {speed:.1f} km/h", True, (255,255,255)), (20, 10))
            display.blit(font_small.render(f"Pred Target Speed: {float(spd_pred.item()):.2f}", True, (255,255,255)), (20, 60))
            display.blit(font_small.render(f"Pred Brake: {float(brk_pred.item()):.2f}", True, (255,255,255)), (20, 90))

            pygame.display.flip()

    finally:
        try: camera.stop()
        except Exception: pass
        try: vehicle.destroy()
        except Exception: pass
        pygame.quit()
        print("[Info] Cleaned up.")

if __name__ == "__main__":
    main()