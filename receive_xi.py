# ===============================================
# 文件 1：本地 Py37 - Carla 主控端（client）
# 保存为：`carla_client.py`
# ===============================================
import socket
import pickle
import struct
import numpy as np
import torch
import cv2
import pygame
import carla
import albumentations as A
from albumentations.pytorch import ToTensorV2

HOST = 'uosremote.shef.ac.uk'   # ✅ 或者你实际用的 IP，如 143.167.46.37
PORT = 50007                    # ✅ 注意：你服务端运行的端口是 50007

def send_image(sock, image):
    _, img_encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    data = pickle.dumps(img_encoded)
    sock.sendall(struct.pack("<L", len(data)) + data)

def receive_result(sock):
    data_len = struct.unpack("<L", sock.recv(4))[0]
    data = b""
    while len(data) < data_len:
        packet = sock.recv(data_len - len(data))
        if not packet:
            return None
        data += packet
    result = pickle.loads(data)
    return result  # dict: {"seg": ..., "depth": ..., "waypoints": ...}

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
    transform_aug = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    pygame.init()
    display = pygame.display.set_mode((960, 720))
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)

    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    vehicle_bp = blueprint_library.find("vehicle.lincoln.mkz_2017")
    vehicle = world.spawn_actor(vehicle_bp, world.get_map().get_spawn_points()[0])

    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '1024')
    camera_bp.set_attribute('image_size_y', '512')
    camera_bp.set_attribute('fov', '90')
    camera = world.spawn_actor(camera_bp, carla.Transform(carla.Location(x=-1.5, z=2.0)), attach_to=vehicle)

    image_data = {"image": None}
    def process_image(image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        image_data["image"] = arr[:, :, :3]
    camera.listen(process_image)

    control = carla.VehicleControl()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((HOST, PORT))
    print("✅ Connected to server")

    try:
        while True:
            clock.tick(30)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

            keys = pygame.key.get_pressed()
            control.throttle = 0.5 + (0.5 if keys[pygame.K_w] else 0.0)
            control.brake = 1.0 if keys[pygame.K_s] else 0.0
            control.steer = -0.3 if keys[pygame.K_a] else (0.3 if keys[pygame.K_d] else 0.0)
            vehicle.apply_control(control)

            if image_data["image"] is None:
                continue

            rgb_img = cv2.cvtColor(image_data["image"], cv2.COLOR_BGR2RGB)
            send_image(sock, rgb_img)
            result = receive_result(sock)
            if result is None:
                continue

            # ---------- 可视化 ----------
            print("📦 seg:", np.shape(result["seg"]), result["seg"].dtype)
            print("📦 depth:", np.shape(result["depth"]), result["depth"].dtype)

            seg_vis = cv2.resize(result["seg"], (480, 360))
            depth_vis = cv2.resize(result["depth"], (480, 360))
            rgb_vis = cv2.resize(rgb_img, (960, 360))

            # ---------- 轨迹点 ----------
            for x, y in get_world_waypoints(vehicle, result["waypoints"]):
                loc = carla.Location(x=x, y=y, z=0.5)
                world.debug.draw_point(loc, size=0.15, color=carla.Color(0, 255, 0), life_time=0.07)

            # ---------- pygame 显示 ----------
            display.blit(pygame.surfarray.make_surface(np.flipud(np.rot90(rgb_vis))), (0, 0))
            display.blit(pygame.surfarray.make_surface(np.flipud(np.rot90(seg_vis))), (0, 360))
            display.blit(pygame.surfarray.make_surface(np.flipud(np.rot90(depth_vis))), (480, 360))
            pygame.display.flip()

    finally:
        camera.stop()
        vehicle.destroy()
        sock.close()
        pygame.quit()
        print("✅ Cleaned up")

if __name__ == "__main__":
    main()
