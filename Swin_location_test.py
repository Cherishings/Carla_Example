# -*- coding: utf-8 -*-
"""
CARLA 可视化：顶部整幅 RGB（叠加 5 个预测路点与顺滑曲线）
底部左：语义分割；底部右：离散深度

✅ 参数集中在顶部：
- CKPT_PATH：你的权重路径
- CAM_W, CAM_H：相机分辨率（显存紧张就降）
- FOV_DEG：相机视场角（小 → 远处更稳、抖动小；大 → 视野广、远处抖动大）
- Z_OFFSET：将路点投到世界坐标时加的高度（抬高可避免地面遮挡）
- EMA_DECAY：路点世界坐标的时间平滑（大 → 更稳更迟钝；小 → 更灵敏更抖）
- SAMPLES_PER_SEG：像素平面 Catmull–Rom 样条每段采样点数（大 → 曲线更丝滑，CPU 开销略增）
- USE_AMP：推理是否用混合精度（True 可省显存/提速；False 最稳）
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
import time
# ============================= 可调参数 =============================
CKPT_PATH = "/home/ali_carla/Desktop/Ali/Leader_Board_2/CARLA_Leaderboard_2.0/CARLA_Leaderboard_20/PythonAPI/examples/Swin_WP5_with2DPos_20250818_030444/swin_wp5_best.pth"

# —— 相机与窗口 ——（显存紧张就把分辨率降到 640x240 或 800x300）
CAM_W, CAM_H = 960, 360          # 顶部 RGB 原生分辨率（同时也是窗口上半部分大小）
FOV_DEG = 90.0                   # 视场角（小一些抖动会更小，常用 60~90）
WIN_W, WIN_H = 960, 720          # 窗口总大小（保持 960x720 排版）
TILE_W, TILE_H = 480, 360        # 底部左右两块面板的尺寸

# —— 路点绘制与平滑 —— 
Z_OFFSET = 1.2                  # 投影前在世界系抬高的 z（单位 m），避免地面遮挡。0.5~1.5 合理
EMA_DECAY = 0.85                 # 指数滑动平均（0.80~0.92 常用；越大越稳、越滞后）
SAMPLES_PER_SEG = 24             # 样条每段采样数（越大越丝滑，CPU 开销稍增；20~32 常用）
USE_AMP = False                  # 混合精度推理（True 更省显存/更快；False 更稳）

# —— 模型/数据设定（需与训练一致） —— 
INPUT_W, INPUT_H = 224, 224
NUM_CLASSES = 7
NUM_DEPTH_BINS = 8
NUM_WAYPOINTS = 5
TARGET_POINT = (90.0, 0.0)       # 与训练一致的引导点（单位：自车坐标系，m）

# ============================= 基础设置 =============================
# 开启/关闭 cuDNN 自动调优（固定输入尺寸时 True 更快；显存紧张不影响）
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Info] Using device: {device}")

# 预处理（需与训练一致）
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
        indexing='ij'     # 避免 future warning，明确行优先
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

# ============================= 工具函数 =============================
def to_surface(arr_rgb_uint8):
    # Pygame 的 surface 需要旋转 + 翻转
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
    # CARLA 提供的逆矩阵就是 world->camera
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
    """对 5 个世界系路点做指数滑动平均，减少帧间抖动。"""
    def __init__(self, decay=EMA_DECAY):
        self.decay = decay
        self.state = None  # [(x,y,z), ...] 长度=5

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
    """
    像素平面 Catmull–Rom 样条插值，让折线更顺滑。
    至少 4 个点更有效；否则直接返回原始点。
    """
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
    """在 RGB 图上画点 + (可选)编号 + (可选)顺滑样条线。"""
    h, w, _ = rgb_vis.shape
    in_pts = []
    for i, p in enumerate(pix_pts):
        if p is None:
            continue
        u, v = p
        if 0 <= u < w and 0 <= v < h:
            in_pts.append((u, v, i))  # 带原索引 i

    # 点与编号
    for (u, v, i) in in_pts:
        cv2.rectangle(rgb_vis, (u-4, v-4), (u+4, v+4), color, thickness=-1)
        if draw_ids:
            cv2.putText(rgb_vis, f"W{i+1}", (u+6, v-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # 曲线/折线
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
    cur = world.get_map().name  # 形如 'Carla/Maps/Town04'
    if TARGET_TOWN not in cur:
        world = client.load_world(TARGET_TOWN)  # 也可传 'Carla/Maps/Town04'
        world.wait_for_tick()  # 或 time.sleep(2)
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
    cam_bp.set_attribute("fov", str(FOV_DEG))          # << FOV：小则更“稳”、大则视野更广
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

            # 顶部 RGB：叠加（平滑后的）路点 + 曲线
            rgb_vis = frame.copy()
            wps_ego = wp_pred.squeeze(0).detach().cpu().numpy()  # (5,2)
            wps_world_raw = ego_to_world(vehicle, [(float(x), float(y)) for x, y in wps_ego], z_offset=Z_OFFSET)
            wps_world_smooth = ema.update(wps_world_raw)

            K = get_camera_K(camera)
            W2C = world_to_camera_matrix(camera)
            pix = project_world_to_image(K, W2C, wps_world_smooth)

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