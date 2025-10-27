# -*- coding: utf-8 -*-
"""
CARLA 可视化：顶部整幅 RGB（叠加稳定后的 5 个预测路点与顺滑曲线）
底部左：语义分割；底部右：离散深度

✅ 关键可调参数（脚本顶部即可改）：
- CKPT_PATH：你的权重路径
- CAM_W, CAM_H：相机分辨率（显存紧张就降）
- FOV_DEG：相机视场角
- Z_OFFSET：将路点投到世界坐标时加的高度（抬高可避免地面遮挡）
- EMA_DECAY：世界坐标上的额外 EMA 平滑（可适当降低到 0.75~0.85）
- SAMPLES_PER_SEG：像素平面样条采样数（越大越丝滑）
- 稳定器参数（ANCHORS_M, KF_Q, KF_R, RATE_LIMIT, DEAD_BAND, LOOKAHEAD_S）
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
CKPT_PATH = "/home/ali_carla/Desktop/Ali/Leader_Board_2/CARLA_Leaderboard_2.0/CARLA_Leaderboard_20/PythonAPI/examples/Swin_WP5_with2DPos_20250818_030444/swin_wp5_best.pth"

# —— 相机与窗口 ——（显存紧张就把分辨率降到 640x240 或 800x300）
CAM_W, CAM_H = 960, 360          # 顶部 RGB 原生分辨率（同时也是窗口上半部分大小）
FOV_DEG = 90.0                   # 视场角（小一些抖动会更小，常用 60~90）
WIN_W, WIN_H = 960, 720          # 窗口总大小（保持 960x720 排版）
TILE_W, TILE_H = 480, 360        # 底部左右两块面板的尺寸

# —— 路点绘制与世界坐标额外 EMA —— 
Z_OFFSET = 1.2                   # 投影前在世界系抬高的 z（单位 m），避免地面遮挡。0.5~1.5 合理
EMA_DECAY = 0.82                 # 世界系的额外 EMA：0.75~0.85；越大越稳、越滞后
SAMPLES_PER_SEG = 24             # 样条每段采样数（越大越丝滑，CPU 开销稍增；20~32 常用）
USE_AMP = False                  # 推理是否用混合精度（True 可省显存/更快；False 最稳）

# —— 模型/数据设定（需与训练一致） —— 
INPUT_W, INPUT_H = 224, 224
NUM_CLASSES = 7
NUM_DEPTH_BINS = 8
NUM_WAYPOINTS = 5
TARGET_POINT = (90.0, 0.0)       # 与训练一致的引导点（单位：自车坐标系，m）

# —— 轨迹“时域稳定器”参数（核心）——
# 把每帧的 5 个 (x,y) 投到这些固定前向距离（锚点）上做插值，然后对各锚点的 y 做时域滤波
ANCHORS_M = [1.0, 2.0, 3.0, 4.0, 5.0]  # 适合低速/路口；高速可改为 [5,10,15,20,30]
KF_Q = 0.50           # 过程噪声标准差（m/s 对应的量纲经模型离散化）；大→更活，小→更稳
KF_R = 0.65           # 测量噪声标准差（m）；大→更信任滤波，压抖更强
RATE_LIMIT = 0.35      # 速率限制：横向每秒最多变化（m/s），城市低速 0.6~1.0
DEAD_BAND = 0.05      # 死区：小于该幅度的变化直接忽略（m）d，0.01~0.03
LOOKAHEAD_S = 0.10    # 前视补偿：抵消滞后，0.08~0.15

# ============================= 基础设置 =============================
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

# ============================= 时域稳定器（卡尔曼 + 速率限制 + 死区 + 前视） =============================
class _KF1D:
    """常速模型：状态 [y, y_dot]，观测为 y。"""
    def __init__(self, q=KF_Q**2, r=KF_R**2):
        self.q, self.r = float(q), float(r)
        self.x = None
        self.P = None

    def predict(self, dt):
        if self.x is None:
            return
        A = np.array([[1.0, dt],
                      [0.0, 1.0]], dtype=np.float32)
        Q = np.array([[dt**3/3, dt**2/2],
                      [dt**2/2, dt      ]], dtype=np.float32) * self.q
        self.x = A @ self.x
        self.P = A @ self.P @ A.T + Q

    def update(self, z):
        H = np.array([[1.0, 0.0]], dtype=np.float32)
        R = np.array([[self.r]], dtype=np.float32)
        if self.x is None:
            self.x = np.array([float(z), 0.0], dtype=np.float32)
            self.P = np.eye(2, dtype=np.float32)
            return self.x.copy()
        S = H @ self.P @ H.T + R
        K = (self.P @ H.T) @ np.linalg.inv(S)
        y = np.array([[float(z)]], dtype=np.float32) - (H @ self.x).reshape(1,1)
        self.x = self.x + (K @ y).reshape(2,)
        I = np.eye(2, dtype=np.float32)
        self.P = (I - K @ H) @ self.P
        return self.x.copy()

class TrajectoryStabilizer:
    """
    1) 把 [5,2] 的 (x,y) 在固定锚点 ANCHORS_M 上做线性插值得到 y(a_i)
    2) 对每个锚点的 y 做 1D KF + 死区 + 速率限制
    3) 前视补偿 LOOKAHEAD_S
    返回稳定后的 (x=a_i, y_i)
    """
    def __init__(self, anchors, fps_target=30,
                 q=KF_Q**2, r=KF_R**2,
                 rate_limit_per_s=RATE_LIMIT,
                 deadband_m=DEAD_BAND,
                 lookahead_s=LOOKAHEAD_S):
        self.anchors = np.asarray(anchors, dtype=np.float32)
        self.kf = [_KF1D(q=q, r=r) for _ in anchors]
        self.prev_y = None
        self.prev_t = None
        self.rate_limit_per_s = float(rate_limit_per_s)
        self.deadband_m = float(deadband_m)
        self.lookahead_s = float(lookahead_s)
        self.dt_default = 1.0 / max(1, int(fps_target))

    def _now(self):
        try:
            return time.perf_counter()
        except Exception:
            return None

    def _interp_to_anchors(self, xy):
        """xy: (5,2) -> 在 anchors 上线性插值得到 y(a_i)"""
        order = np.argsort(xy[:,0])
        xs, ys = xy[order,0], xy[order,1]
        return np.interp(self.anchors, xs, ys, left=ys[0], right=ys[-1]).astype(np.float32)

    def update(self, wp_5x2):
        """
        wp_5x2: torch/numpy -> shape (5,2)
        返回：稳定后的 (N,2)，其中 x=anchors, y=滤波后的 y
        """
        if hasattr(wp_5x2, "detach"):
            xy = wp_5x2.detach().cpu().numpy().reshape(-1,2).astype(np.float32)
        else:
            xy = np.asarray(wp_5x2, dtype=np.float32).reshape(-1,2)
        y_meas = self._interp_to_anchors(xy)  # (N,)

        t = self._now()
        if self.prev_t is None or t is None:
            dt = self.dt_default
        else:
            dt = max(0.001, min(0.2, t - self.prev_t))
        self.prev_t = t

        y_out = np.zeros_like(y_meas, dtype=np.float32)
        for i, z in enumerate(y_meas):
            kf = self.kf[i]
            kf.predict(dt)
            x = kf.update(float(z))   # x=[y, v]
            y_now, v_now = float(x[0]), float(x[1])

            # 死区 + 速率限制
            if self.prev_y is not None:
                dy = y_now - self.prev_y[i]
                if abs(dy) < self.deadband_m:
                    y_now = self.prev_y[i]
                else:
                    max_step = self.rate_limit_per_s * dt
                    if dy >  max_step: y_now = self.prev_y[i] + max_step
                    if dy < -max_step: y_now = self.prev_y[i] - max_step

            # 前视补偿
            y_out[i] = y_now + v_now * self.lookahead_s

        self.prev_y = y_out.copy()
        xy_out = np.stack([self.anchors, y_out], axis=1)
        return xy_out

# ============================= 工具函数 =============================
def to_surface(arr_rgb_uint8):
    return pygame.surfarray.make_surface(np.flipud(np.rot90(arr_rgb_uint8)))

def ego_to_world(vehicle, rel_pts, z_offset=Z_OFFSET):
    """rel_pts: [(x,y),...] in ego -> world [(X,Y,Z)]"""
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
    """
    透视投影（CARLA 相机坐标：x 向前、y 向右、z 向上）
    返回像素坐标列表（可能含 None）
    """
    out = []
    for X, Y, Z in pts_world:
        pw = np.array([X, Y, Z, 1.0])
        pc = world_to_cam @ pw
        x, y, z = pc[0], pc[1], pc[2]
        if x <= 0:  # 在相机后面
            out.append(None)
            continue
        u = K[0,0] * (y / x) + K[0,2]
        v = K[1,1] * (-z / x) + K[1,2]
        out.append((int(u), int(v)))
    return out

class EMAFilter:
    """对世界系路点做指数滑动平均，减少投影后的抖动（可选）。"""
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
        if p is None: continue
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
    pygame.display.set_caption("RGB | Segmentation | Depth")
    clock = pygame.time.Clock()
    font_big = pygame.font.Font(None, 48)
    font_small = pygame.font.Font(None, 28)

    # ---- CARLA ----
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    TARGET_TOWN = "Town04"
    world = client.get_world()
    cur = world.get_map().name
    if TARGET_TOWN not in cur:
        world = client.load_world(TARGET_TOWN)
        world.wait_for_tick()
        print(f"[Info] Loaded map: {world.get_map().name}")
    bp = world.get_blueprint_library()

    # 车辆
    veh_bp = bp.find("vehicle.lincoln.mkz_2017")
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.try_spawn_actor(veh_bp, spawn_point) or world.spawn_actor(veh_bp, spawn_point)

    # 相机（分辨率、FOV 均可在顶部参数里改）
    cam_bp = bp.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(CAM_W))
    cam_bp.set_attribute("image_size_y", str(CAM_H))
    cam_bp.set_attribute("fov", str(FOV_DEG))
    cam_transform = carla.Transform(carla.Location(x=-1.5, z=2.0))
    camera = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)

    shared = {"rgb": None}
    def on_image(img):
        arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape((img.height, img.width, 4))
        rgb = arr[:, :, :3][:, :, ::-1].copy()  # BGRA -> RGB
        shared["rgb"] = rgb
    camera.listen(on_image)

    # 模型 + 稳定器 + 世界系 EMA
    model = load_model(CKPT_PATH)
    stab = TrajectoryStabilizer(
        anchors=ANCHORS_M,
        fps_target=30,
        q=KF_Q**2, r=KF_R**2,
        rate_limit_per_s=RATE_LIMIT,
        deadband_m=DEAD_BAND,
        lookahead_s=LOOKAHEAD_S
    )
    ema_world = EMAFilter(decay=EMA_DECAY)

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

            # 简单手动控制（W/A/S/D，空格手刹）
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

            frame = shared["rgb"]  # [CAM_H, CAM_W, 3] RGB
            vel = vehicle.get_velocity()
            speed = 3.6 * (vel.x**2 + vel.y**2 + vel.z**2) ** 0.5

            # 预处理
            aug = transform(image=frame)
            inp = aug["image"].unsqueeze(0).to(device)
            tgt_pt = torch.tensor([TARGET_POINT], dtype=torch.float32, device=device)
            cur_sp = torch.tensor([speed], dtype=torch.float32, device=device)

            # 推理（可选混合精度）
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

            # === 轨迹稳定（关键）===
            # 1) 模型输出 (5,2) -> 在固定 anchors 上得到稳定后的 (x=anchors, y_filtered)
            wp_ego_raw = wp_pred.squeeze(0)  # torch (5,2)
            wp_ego_stable = stab.update(wp_ego_raw)  # numpy (N,2), N=len(ANCHORS_M)

            # 2) 转世界坐标并可选 EMA
            wps_world_raw = ego_to_world(vehicle, [(float(x), float(y)) for x, y in wp_ego_stable], z_offset=Z_OFFSET)
            wps_world_smooth = ema_world.update(wps_world_raw)

            # 3) 投影到像素并绘制
            K = get_camera_K(camera)
            W2C = world_to_camera_matrix(camera)
            pix = project_world_to_image(K, W2C, wps_world_smooth)

            # 顶部 RGB：叠加（稳定后的）路点 + 曲线
            rgb_vis = frame.copy()
            draw_waypoints_on_rgb(rgb_vis, pix, draw_ids=True, smooth=True, color=(0,255,0))

            # ---------- Pygame 显示 ----------
            rgb_surface = to_surface(rgb_vis)             # 顶部整幅 RGB
            seg_surface = to_surface(seg_img)             # 左下
            depth_surface = to_surface(depth_vis)         # 右下

            display.blit(rgb_surface, (0, 0))
            display.blit(seg_surface, (0, 360))
            display.blit(depth_surface, (480, 360))

            # 文本信息
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