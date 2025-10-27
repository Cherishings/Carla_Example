import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import pygame
import carla
import random
import math
import casadi as ca
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ----------------------------
# Device
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------
# VehicleTransformerMAnet Definition
# ----------------------------
class VehicleTransformerMAnet(nn.Module):
    def __init__(self, num_classes=7, hidden_size=64):
        super(VehicleTransformerMAnet, self).__init__()
        self.manet = smp.MAnet(
            encoder_name="mit_b2",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.visual_fc = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 64), nn.ReLU()
        )
        self.meta_input_net = nn.Sequential(
            nn.Linear(3, 20), nn.ReLU(), nn.Linear(20, 8), nn.ReLU()
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
        last_feat = features[-1]
        pooled_feat = self.avg_pool(last_feat).view(x_img.size(0), -1)
        img_feat = self.visual_fc(pooled_feat)
        decoder_out = self.manet.decoder(*features)
        seg_out = self.manet(x_img)
        depth_logits = self.depth_head(decoder_out)
        meta = torch.cat([target_point, current_speed.unsqueeze(1)], dim=1)
        meta_feat = self.meta_input_net(meta)
        context = torch.cat([img_feat, meta_feat], dim=1)
        prev_wp = torch.zeros((x_img.size(0), 2), device=x_img.device)
        waypoints, hidden = [], None
        for _ in range(5):
            step_input = torch.cat([context, prev_wp, target_point], dim=1).unsqueeze(1)
            out, hidden = self.gru(step_input, hidden)
            pred_wp = self.gru_out(out[:, -1, :])
            prev_wp = pred_wp
            waypoints.append(pred_wp)
        waypoints = torch.stack(waypoints, dim=1)
        speed = self.output_speed(context)
        brake = torch.sigmoid(self.output_brake(context))
        return seg_out, waypoints, speed, brake, depth_logits

# ----------------------------
# Kinematic Bicycle Model & NMPC
# ----------------------------
class KinematicBicycleModel:
    def __init__(self, lf=1.0, lr=1.0, dt=0.1):
        self.lf, self.lr, self.dt = lf, lr, dt

    def step_rk4(self, state, control):
        def f(s, u):
            x, y, psi, v = s[0], s[1], s[2], s[3]
            delta, a = u[0], u[1]
            beta = ca.atan((self.lr * ca.tan(delta)) / (self.lf + self.lr))
            return ca.vertcat(
                v * ca.cos(psi + beta),
                v * ca.sin(psi + beta),
                (v * ca.tan(delta)) / (self.lf + self.lr),
                a
            )
        k1 = f(state, control)
        k2 = f(state + 0.5 * self.dt * k1, control)
        k3 = f(state + 0.5 * self.dt * k2, control)
        k4 = f(state + self.dt * k3, control)
        return state + (self.dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

class NMPCController:
    def __init__(self, model, N=10, dt=0.1):
        self.model = model
        self.N = N
        self.opti = ca.Opti()
        self.X = self.opti.variable(4, N+1)
        self.U = self.opti.variable(2, N)
        self.target = self.opti.parameter(4)
        self.X_init = self.opti.parameter(4)
        Qx = ca.diag(ca.DM([20, 20, 5, 5]))
        Qu = ca.diag(ca.DM([10, 1]))
        Q_terminal = ca.diag(ca.DM([50, 50, 0, 10]))
        cost = sum(
            ca.mtimes([(self.X[:, k] - self.target).T, Qx, (self.X[:, k] - self.target)]) +
            ca.mtimes([self.U[:, k].T, Qu, self.U[:, k]])
            for k in range(N)
        )
        cost += ca.mtimes([(self.X[:, -1] - self.target).T, Q_terminal, (self.X[:, -1] - self.target)])
        self.opti.minimize(cost)
        for k in range(N):
            x_next = self.model.step_rk4(self.X[:, k], self.U[:, k])
            self.opti.subject_to(self.X[:, k+1] == x_next)
        self.opti.subject_to(self.opti.bounded(-0.5, self.U[0, :], 0.5))
        self.opti.subject_to(self.opti.bounded(-1.0, self.U[1, :], 1.0))
        self.opti.subject_to(self.X[:, 0] == self.X_init)
        self.opti.solver("ipopt", {"ipopt.print_level": 0, "print_time": 0})

    def solve(self, state, target):
        self.opti.set_value(self.X_init, state)
        self.opti.set_value(self.target, target)
        sol = self.opti.solve()
        return sol.value(self.U[:, 0])

# ----------------------------
# Main Loop
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

def main():
    pygame.init()
    display = pygame.display.set_mode((960, 720))
    pygame.display.set_caption("RGB | Segmentation | Depth")
    clock = pygame.time.Clock()
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.find("vehicle.lincoln.mkz_2017")
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '1024')
    camera_bp.set_attribute('image_size_y', '512')
    camera_transform = carla.Transform(carla.Location(x=-1.5, z=2.0))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    image_data = {"image": None}
    def process_image(image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        image_data["image"] = arr[:, :, :3]
    camera.listen(process_image)

    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    color_map = {
        0: [0, 0, 0], 1: [0, 0, 255], 2: [0, 255, 0],
        3: [255, 0, 0], 4: [255, 255, 0], 5: [255, 0, 255], 6: [0, 255, 255],
    }

    model = VehicleTransformerMAnet(num_classes=7).to(device)
    model.load_state_dict(torch.load("./transformer_corrected_VehicleNNController_NAG.pth", map_location=device))
    model.eval()

    controller = NMPCController(KinematicBicycleModel())
    font = pygame.font.Font(None, 36)

    try:
        while True:
            clock.tick(30)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
            if image_data["image"] is None:
                continue

            rgb_img = cv2.cvtColor(image_data["image"], cv2.COLOR_BGR2RGB)
            aug = transform(image=rgb_img)
            input_tensor = aug["image"].unsqueeze(0).to(device)

            transform_carla = vehicle.get_transform()
            velocity = vehicle.get_velocity()
            speed = 3.6 * np.linalg.norm([velocity.x, velocity.y, velocity.z])
            speed_tensor = torch.tensor([speed], dtype=torch.float32).to(device)
            target_pt = torch.tensor([[90.0, 0.0]], dtype=torch.float32).to(device)

            with torch.no_grad():
                seg_out, wp_out, speed_pred, brake_pred, depth_out = model(input_tensor, target_pt, speed_tensor)

            seg_mask = torch.argmax(seg_out.squeeze(), dim=0).cpu().numpy()
            seg_img = np.zeros((*seg_mask.shape, 3), dtype=np.uint8)
            for cid, color in color_map.items():
                seg_img[seg_mask == cid] = color
            depth_mask = torch.argmax(depth_out.squeeze(), dim=0).cpu().numpy()
            depth_vis = cv2.applyColorMap((depth_mask * (255 // 7)).astype(np.uint8), cv2.COLORMAP_JET)

            rgb_vis = cv2.resize(rgb_img, (960, 360))
            seg_img = cv2.resize(seg_img, (480, 360))
            depth_vis = cv2.resize(depth_vis, (480, 360))

            relative_wp = wp_out.squeeze().cpu().numpy().reshape(-1, 2)
            world_wp = get_world_waypoints(vehicle, relative_wp)
            for i, (x, y) in enumerate(world_wp):
                loc = carla.Location(x=x, y=y, z=0.5)
                world.debug.draw_point(loc, 0.15, carla.Color(0, 255, 0), 0.07)

            yaw = np.radians(transform_carla.rotation.yaw)
            state = np.array([transform_carla.location.x, transform_carla.location.y, yaw, speed])
            target = np.array([*world_wp[0], yaw, speed_pred.item()])
            u = controller.solve(state, target)
            control = carla.VehicleControl()
            control.steer = float(u[0])
            control.throttle = max(0.0, float(u[1]))
            control.brake = max(0.0, -float(u[1]))
            vehicle.apply_control(control)

            display.blit(pygame.surfarray.make_surface(np.flipud(np.rot90(rgb_vis))), (0, 0))
            display.blit(pygame.surfarray.make_surface(np.flipud(np.rot90(seg_img))), (0, 360))
            display.blit(pygame.surfarray.make_surface(np.flipud(np.rot90(depth_vis))), (480, 360))
            display.blit(font.render(f"Speed: {speed:.1f} km/h", True, (255, 255, 255)), (20, 10))
            pygame.display.flip()
    finally:
        camera.stop()
        vehicle.destroy()
        pygame.quit()
        print("Cleaned up.")

if __name__ == "__main__":
    main()