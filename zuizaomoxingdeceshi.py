# -*- coding: utf-8 -*-
import os, math, time
import numpy as np
import cv2
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import carla
import cvxpy as cp
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ====================== 可调参数 ======================
TOWN            = "Town05"
CAM_W, CAM_H    = 1024, 512
FOV_DEG         = 90.0
WIN_W, WIN_H    = 1024, 768            # 顶部 1024x512 + 底部两块 512x256

# 控制/约束
AUTO_DRIVE      = True
MAX_THROTTLE    = 0.5
STEER_LIMIT     = 0.5
THROTTLE_SLEW   = 0.05
BRAKE_THRESH    = 0.95

SPEED_LIMIT_KMH = 30.0
SPEED_BAND_KMH  = 2.0
SPEED_MIN_MPS   = (SPEED_LIMIT_KMH - SPEED_BAND_KMH) / 3.6
SPEED_MAX_MPS   = (SPEED_LIMIT_KMH + SPEED_BAND_KMH) / 3.6
SPEED_SET_MPS   = SPEED_LIMIT_KMH / 3.6
SPEED_RAMP_A    = 1.0  # m/s^2

# 模型与输入
INPUT_W, INPUT_H = 224, 224
NUM_CLASSES      = 7
META_TP          = (90.0, 0.0)   # 传给 meta 的 Target Point（自车系）
PRED_SPEED_IS_KMH = False        # 如果你这个模型的 speed 头是 km/h，改为 True

# 权重路径（改成你的）
MODEL_PATH = "/home/ali_carla/Desktop/Ali/Leader_Board_2/CARLA_Leaderboard_2.0/CARLA_Leaderboard_20/PythonAPI/examples/Transformer_GRU_result_20250730_141837/transformer_corrected_VehicleNNController_NAG.pth"

# ====================== 设备/预处理 ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Info] Using device: {device}")

transform = A.Compose([
    A.Resize(INPUT_H, INPUT_W),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

# ====================== 模型定义（你的 MAnet+GRU） ======================
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

    def forward(self, x_img, target_point, current_speed):
        features = self.manet.encoder(x_img)
        last_feat = features[-1]                     # [B, 512, H/32, W/32] (对 mit_b2 这里是 8x8)
        B, C, H, W = last_feat.shape
        pooled_feat = self.avg_pool(last_feat).view(B, -1)  # [B,512]
        img_feat = self.visual_fc(pooled_feat)              # [B,64]

        decoder_out = self.manet.decoder(*features)         # [B,16,h,w]
        seg_out = self.manet(x_img)                         # [B,num_classes,224,224]
        depth_logits = self.depth_head(decoder_out)         # [B,8,224,224]（与 seg 同尺寸）

        meta = torch.cat([target_point, current_speed.unsqueeze(1)], dim=1)  # [B,3]
        meta_feat = self.meta_input_net(meta)               # [B,8]
        context = torch.cat([img_feat, meta_feat], dim=1)   # [B,72]

        prev_wp = torch.zeros((x_img.size(0), 2), device=x_img.device)
        waypoints = []
        hidden = None
        for t in range(5):
            # 输入 = 上下文(72) + prev_wp(2) + target_point(2) -> [B,76] 再 unsqueeze 成 [B,1,76]
            input_step = torch.cat([context, prev_wp, target_point], dim=1).unsqueeze(1)
            out, hidden = self.gru(input_step, hidden)      # out:[B,1,hidden]
            pred_wp = self.gru_out(out[:, -1, :])           # [B,2]
            prev_wp = pred_wp
            waypoints.append(pred_wp)
        waypoints = torch.stack(waypoints, dim=1)           # [B,5,2]

        fused = torch.cat([img_feat, meta_feat], dim=1)     # [B,72]
        speed = self.output_speed(fused)                    # [B,1]
        brake = torch.sigmoid(self.output_brake(fused))     # [B,1]

        return seg_out, waypoints, speed, brake, depth_logits

# ====================== 投影/可视化工具 ======================
def to_surface(arr_rgb_uint8):
    return pygame.surfarray.make_surface(np.flipud(np.rot90(arr_rgb_uint8)))

def get_camera_K(camera):
    w = int(camera.attributes["image_size_x"]); h = int(camera.attributes["image_size_y"])
    fov = float(camera.attributes["fov"])
    f = w / (2.0 * math.tan(fov * math.pi / 360.0))
    return np.array([[f, 0, w/2.0],
                     [0, f, h/2.0],
                     [0, 0, 1]])

def world_to_camera_matrix(camera):
    return np.array(camera.get_transform().get_inverse_matrix())

def ego_to_world(vehicle, rel_pts, z=0.3):
    tf = vehicle.get_transform()
    yaw = math.radians(tf.rotation.yaw)
    c, s = math.cos(yaw), math.sin(yaw)
    bx, by, bz = tf.location.x, tf.location.y, tf.location.z
    out = []
    for x, y in rel_pts:
        X = bx + x*c - y*s
        Y = by + x*s + y*c
        out.append((X, Y, bz + z))
    return out

def project_world_to_image(K, W2C, pts_world):
    out = []
    for X, Y, Z in pts_world:
        pw = np.array([X, Y, Z, 1.0])
        pc = W2C @ pw
        x, y, z = pc[0], pc[1], pc[2]
        if x <= 0: out.append(None); continue
        u = K[0,0]*(y/x) + K[0,2]
        v = K[1,1]*(-z/x) + K[1,2]
        out.append((int(u), int(v)))
    return out

def draw_wps_on_rgb(rgb, pix, color=(0,255,0)):
    h,w,_ = rgb.shape
    good=[]
    for i,p in enumerate(pix):
        if p is None: continue
        u,v=p
        if 0<=u<w and 0<=v<h:
            good.append((u,v,i))
    for (u,v,i) in good:
        cv2.rectangle(rgb,(u-4,v-4),(u+4,v+4),color,-1)
        cv2.putText(rgb,f"W{i+1}",(u+6,v-6),cv2.FONT_HERSHEY_SIMPLEX,0.45,color,1,cv2.LINE_AA)
    if len(good)>=2:
        arr = np.array([(u,v) for (u,v,_) in good], dtype=np.int32).reshape(-1,1,2)
        cv2.polylines(rgb,[arr],False,color,2,cv2.LINE_AA)

# ====================== 你的 MPC（A,B,Q,p） ======================
A_m = np.array([
    [ 0.6800279,   0.4065171,   0.00412744],
    [ 0.00738894,  0.6344572,  -0.01294562],
    [ 0.01396624, -0.03162604,  0.57808036]
])
B_m = np.array([
    [-0.03505046, -0.3509874 ],
    [ 0.02046476, -0.00288475],
    [ 0.00736699, -0.07007284]
])
Qblk = np.array([
    [ 9.0583766e-01,  6.4924471e-02, -1.1931288e+00,  1.0546644e-01, -1.1077980e+00],
    [ 6.4924471e-02,  1.1044665e-02, -3.0973318e-01,  4.3207701e-02, -1.5907243e-02],
    [-1.1931288e+00, -3.0973318e-01,  1.5252293e+01, -2.5793946e+00,  7.5840425e-01],
    [ 1.0546644e-01,  4.3207701e-02, -2.5793946e+00,  4.7711173e-01, -5.4839104e-01],
    [-1.1077980e+00, -1.5907243e-02,  7.5840425e-01, -5.4839104e-01,  1.1870452e+01]
])
pvec = np.array([-0.17293791,  0.02827822,  0.6870698,  -0.11809217,  0.16456106])

nx = A_m.shape[0]; nu = B_m.shape[1]; N = 10
Q_xx = Qblk[:nx,:nx]; p_x = pvec[:nx]
LOOKAHEAD_INDEX = 4

def mpc_control(x0):
    x = cp.Variable((nx, N+1))
    u = cp.Variable((nu, N))
    cost = 0; constr = [x[:,0] == x0]
    for k in range(N):
        constr += [x[:,k+1] == A_m @ x[:,k] + B_m @ u[:,k]]
        xu = cp.hstack([x[:,k], u[:,k]])
        cost += 0.5*cp.quad_form(xu, Qblk) + pvec @ xu
        constr += [u[0,k] <= MAX_THROTTLE, u[0,k] >= 0.0]
        constr += [u[1,k] <=  STEER_LIMIT, u[1,k] >= -STEER_LIMIT]
    cost += 0.5*cp.quad_form(x[:,N], Q_xx) + p_x @ x[:,N]
    prob = cp.Problem(cp.Minimize(cost), constr)
    prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
    if prob.status not in ["optimal","optimal_inaccurate"]:
        raise RuntimeError(f"MPC failed: {prob.status}")
    return np.asarray(u[:,0].value).reshape(-1)

def build_state_from_wps(wps_ego, cur_mps, tgt_mps):
    if wps_ego is None or len(wps_ego)==0: return np.zeros(3, np.float32)
    i = min(max(LOOKAHEAD_INDEX,0), len(wps_ego)-1)
    x,y = float(wps_ego[i,0]), float(wps_ego[i,1])
    if x < 1.0 and len(wps_ego)>=2:
        x,y = float(wps_ego[-1,0]), float(wps_ego[-1,1])
    e_y   = float(np.clip(y, -5.0, 5.0))
    e_psi = float(np.clip(math.atan2(y, max(1e-3, x)), -math.pi/2, math.pi/2))
    e_v   = float(np.clip(tgt_mps - cur_mps, -10.0, 10.0))
    return np.array([e_y, e_psi, e_v], dtype=np.float32)

def apply_control(vehicle, u, brake_prob, cur_mps, prev_throttle):
    thr = float(np.clip(u[0], 0.0, MAX_THROTTLE))
    steer = float(np.clip(u[1], -STEER_LIMIT, STEER_LIMIT))
    brake = 0.0
    if brake_prob > BRAKE_THRESH and cur_mps > 0.2:
        brake = max(brake, 0.5); thr = 0.0
    if cur_mps > SPEED_MAX_MPS:
        overs = cur_mps - SPEED_MAX_MPS
        brake = max(brake, float(np.clip(0.2 + 0.3*overs, 0.3, 1.0))); thr = 0.0
    d = np.clip(thr - prev_throttle, -THROTTLE_SLEW, THROTTLE_SLEW)
    thr = float(np.clip(prev_throttle + d, 0.0, MAX_THROTTLE))
    prev_throttle = thr
    vehicle.apply_control(carla.VehicleControl(throttle=thr, steer=steer, brake=brake))
    return thr, steer, brake, prev_throttle

# ====================== 主流程 ======================
def main():
    # pygame
    pygame.init()
    display = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("MAnet-GRU + MPC")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 32)

    # CARLA
    client = carla.Client("localhost", 2000); client.set_timeout(10.0)
    world = client.get_world()
    if TOWN not in world.get_map().name:
        world = client.load_world(TOWN)
        world.wait_for_tick()
        print(f"[Info] Loaded: {world.get_map().name}")

    bp = world.get_blueprint_library()
    veh_bp = bp.find("vehicle.lincoln.mkz_2017")
    spawn = world.get_map().get_spawn_points()[0]
    ego = world.try_spawn_actor(veh_bp, spawn) or world.spawn_actor(veh_bp, spawn)

    cam_bp = bp.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', str(CAM_W))
    cam_bp.set_attribute('image_size_y', str(CAM_H))
    cam_bp.set_attribute('fov', str(FOV_DEG))
    cam_tf = carla.Transform(carla.Location(x=-1.5, z=2.0))
    cam = world.spawn_actor(cam_bp, cam_tf, attach_to=ego)

    shared = {"rgb": None}
    def on_image(img):
        arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape((img.height, img.width, 4))
        shared["rgb"] = arr[:, :, :3]
    cam.listen(on_image)

    # 模型加载
    model = VehicleTransformerMAnet(num_classes=NUM_CLASSES).to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print("[Info] Model loaded.")

    # 颜色表
    color_map = {
        0:[0,0,0],1:[0,0,255],2:[0,255,0],3:[255,0,0],
        4:[255,255,0],5:[255,0,255],6:[0,255,255]
    }

    # 目标速度缓升初始化
    vel0 = ego.get_velocity()
    sp0_kmh = 3.6 * ((vel0.x**2 + vel0.y**2 + vel0.z**2)**0.5)
    tgt_speed = float(np.clip(sp0_kmh/3.6, SPEED_MIN_MPS, SPEED_MAX_MPS))
    prev_t = time.time()
    prev_thr = 0.0

    try:
        while True:
            clock.tick(30)
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    return

            rgb = shared["rgb"]
            if rgb is None:
                pygame.display.flip(); continue

            # 车速
            vel = ego.get_velocity()
            sp_kmh = 3.6 * ((vel.x**2 + vel.y**2 + vel.z**2)**0.5)
            sp_mps = sp_kmh / 3.6

            # 预处理 & 前向
            aug = transform(image=rgb)
            inp = aug["image"].unsqueeze(0).to(device)
            tp  = torch.tensor([META_TP], dtype=torch.float32, device=device)
            cur_sp_meta = torch.tensor([sp_kmh], dtype=torch.float32, device=device)  # meta 用 km/h

            with torch.no_grad():
                seg_logits, wp_pred, sp_pred, brk_pred, depth_logits = model(inp, tp, cur_sp_meta)

            # 解析输出
            seg_lbl = torch.argmax(seg_logits.squeeze(0), dim=0).cpu().numpy()
            seg_rgb = np.zeros((seg_lbl.shape[0], seg_lbl.shape[1], 3), dtype=np.uint8)
            for k,c in color_map.items(): seg_rgb[seg_lbl==k] = c
            seg_vis = cv2.resize(seg_rgb, (512, 256), interpolation=cv2.INTER_NEAREST)

            depth_cls = torch.argmax(depth_logits.squeeze(0), dim=0).cpu().numpy()
            depth_255 = (depth_cls.astype(np.float32) / max(1, (8-1)) * 255.0).astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_255, cv2.COLORMAP_PLASMA)
            depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_BGR2RGB)
            depth_vis = cv2.resize(depth_vis, (512, 256), interpolation=cv2.INTER_NEAREST)

            wps = wp_pred.squeeze(0).detach().cpu().numpy().reshape(-1,2)   # (5,2)
            brk = float(brk_pred.item())
            mdl_tgt_mps = float(sp_pred.item())/3.6 if PRED_SPEED_IS_KMH else float(sp_pred.item())

            # 顶部 RGB 叠路点
            rgb_vis = rgb.copy()
            wps_world = ego_to_world(ego, wps, z=0.3)
            K = get_camera_K(cam); W2C = world_to_camera_matrix(cam)
            pix = project_world_to_image(K, W2C, wps_world)
            draw_wps_on_rgb(rgb_vis, pix, color=(0,255,0))

            # ===== 控制 =====
            if AUTO_DRIVE:
                # 目标速度：默认用固定 30km/h 中心；想用模型速度就把 desired 换成 mdl_tgt_mps
                desired = SPEED_SET_MPS
                now = time.time(); dt = max(1e-3, now - prev_t); prev_t = now
                tgt_speed += float(np.clip(desired - tgt_speed, -SPEED_RAMP_A*dt, SPEED_RAMP_A*dt))
                tgt_speed = float(np.clip(tgt_speed, SPEED_MIN_MPS, SPEED_MAX_MPS))

                x0 = build_state_from_wps(wps, cur_mps=sp_mps, tgt_mps=tgt_speed)
                try:
                    u0 = mpc_control(x0)  # [throttle, steer]
                except Exception as e:
                    print(f"[Warn] MPC failed: {e}. fallback.")
                    e_y, e_psi, e_v = x0.tolist()
                    u0 = np.array([
                        np.clip(0.05 + 0.08*e_v, 0.0, MAX_THROTTLE),
                        np.clip(0.8*e_psi, -STEER_LIMIT, STEER_LIMIT)
                    ], dtype=np.float32)

                thr, steer, brake, prev_thr = apply_control(ego, u0, brake_prob=brk, cur_mps=sp_mps, prev_throttle=prev_thr)
            else:
                keys = pygame.key.get_pressed()
                ctrl = carla.VehicleControl()
                ctrl.throttle = 0.5 + (0.5 if keys[pygame.K_w] else 0.0)
                ctrl.brake    = 1.0 if keys[pygame.K_s] else 0.0
                ctrl.steer    = (-0.3 if keys[pygame.K_a] else (0.3 if keys[pygame.K_d] else 0.0))
                ctrl.hand_brake = keys[pygame.K_SPACE]
                ego.apply_control(ctrl)
                thr, steer, brake = ctrl.throttle, ctrl.steer, ctrl.brake

            # ===== 画面拼接 =====
            top = cv2.resize(rgb_vis, (CAM_W, CAM_H))
            top_surf   = to_surface(top)
            seg_surf   = to_surface(seg_vis)
            depth_surf = to_surface(depth_vis)

            display.blit(top_surf,   (0, 0))
            display.blit(seg_surf,   (0, 512))
            display.blit(depth_surf, (512, 512))

            # 文字
            display.blit(font.render(f"Speed: {sp_kmh:.1f} km/h", True, (255,255,255)), (20, 20))
            display.blit(font.render(f"Target: {tgt_speed*3.6:.1f} km/h", True, (255,255,255)), (20, 54))
            display.blit(font.render(f"Brake p: {brk:.2f}", True, (255,255,255)), (20, 88))
            if AUTO_DRIVE:
                display.blit(font.render(f"MPC thr:{thr:.2f} steer:{steer:.2f} brk:{brake:.2f}", True, (0,255,0)), (20, 122))
            else:
                display.blit(font.render("Manual: WASD/SPACE", True, (255,255,0)), (20, 122))

            pygame.display.flip()

    finally:
        try: cam.stop()
        except: pass
        try: ego.destroy()
        except: pass
        pygame.quit()
        print("[Info] Cleaned up.")

if __name__ == "__main__":
    main()