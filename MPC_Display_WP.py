import numpy as np
import casadi as ca
import pygame
import carla
import math
import signal
import matplotlib.pyplot as plt

# Data collection
time_values = []
throttle_values = []
steering_values = []
x_values = []
y_values = []
simulation_running = True  # Flag to detect Ctrl+C

# Kinematic Bicycle Model
class KinematicBicycleModel:
    def __init__(self, lf=-0.090769015, lr=1.4178275, dt=0.1):
        self.lf = lf
        self.lr = lr
        self.dt = dt

    def step_rk4(self, state, control):
        X, Y, psi, V = state[0], state[1], state[2], state[3]
        delta, throttle = control[0], control[1]

        def f(state, control):
            X, Y, psi, V = state[0], state[1], state[2], state[3]
            delta, throttle = control[0], control[1]
            beta = ca.atan((self.lr * ca.tan(delta)) / (self.lf + self.lr))
            return ca.vertcat(
                V * ca.cos(psi + beta),
                V * ca.sin(psi + beta),
                (V * ca.tan(delta)) / (self.lf + self.lr),
                throttle
            )

        k1 = f(state, control)
        k2 = f(state + 0.5 * self.dt * k1, control)
        k3 = f(state + 0.5 * self.dt * k2, control)
        k4 = f(state + self.dt * k3, control)
        next_state = state + (self.dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        return next_state

class NMPCController:
    def __init__(self, model, N=10, dt=0.1):
        self.model = model
        self.N = N
        self.dt = dt

        self.opti = ca.Opti()
        self.X = self.opti.variable(4, N+1)
        self.U = self.opti.variable(2, N)

        self.target = self.opti.parameter(4)
        Qx = np.diag([30, 30, 0, 20])
        Qu = np.diag([10, 100])
        Q_Terminal = np.diag([10, 10, 0, 0])
        Qx_casadi = ca.DM(Qx)
        Qu_casadi = ca.DM(Qu)
        Q_Terminal_casadi = ca.DM(Q_Terminal)

        lambda_steering_rate = 50  
        lambda_throttle_rate = 30

        cost = 0
        for k in range(N):
            state_error = self.X[:, k] - self.target
            control_effort = self.U[:, k]
            cost += ca.mtimes([state_error.T, Qx_casadi, state_error])
            cost += ca.mtimes([control_effort.T, Qu_casadi, control_effort])
            if k < N - 1:
                delta_diff = self.U[0, k+1] - self.U[0, k]
                throttle_diff = self.U[1, k+1] - self.U[1, k]
                cost += lambda_steering_rate * ca.mtimes([delta_diff.T, delta_diff])
                cost += lambda_throttle_rate * ca.mtimes([throttle_diff.T, throttle_diff])
        cost += ca.mtimes([(self.X[:, -1] - self.target).T, Q_Terminal_casadi, (self.X[:, -1] - self.target)])

        self.opti.minimize(cost)

        for k in range(N):
            X_next = self.model.step_rk4(self.X[:, k], self.U[:, k])
            self.opti.subject_to(self.X[:, k+1] == X_next)

        self.opti.subject_to(self.opti.bounded(-1, self.U[0, :], 1))
        self.opti.subject_to(self.opti.bounded(0, self.U[1, :], 1))

        self.X_init = self.opti.parameter(4)
        self.opti.subject_to(self.X[:, 0] == self.X_init)

        opts = {"ipopt.print_level": 0, "print_time": 0}
        self.opti.solver("ipopt", opts)

    def solve(self, state, target):
        self.opti.set_value(self.X_init, state)
        self.opti.set_value(self.target, target)
        sol = self.opti.solve()
        return sol.value(self.U[:, 0])

def signal_handler(sig, frame):
    global simulation_running
    print("\nSimulation stopped. Saving data and plotting results...")
    simulation_running = False

signal.signal(signal.SIGINT, signal_handler)

def generate_reference_track():
    return np.array([
        # [0, 0], [10, 3]
        [0, 0], [10, -3]
    ])

def draw_waypoints(world, waypoints, z=0.5, life_time=60.0):
    for i, wp in enumerate(waypoints):
        location = carla.Location(x=wp[0], y=wp[1], z=z)
        # Draw point
        world.debug.draw_point(location, size=0.2, color=carla.Color(0, 255, 0), life_time=life_time, persistent_lines=True)
        # Draw label
        label = f"W{i}"
        world.debug.draw_string(location + carla.Location(z=0.5), label, draw_shadow=False,
                                color=carla.Color(0, 255, 0), life_time=life_time, persistent_lines=True)


def process_image(image, display):
    if not pygame.get_init():
        return
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))[:, :, :3]
    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    try:
        display.blit(surface, (0, 0))
        pygame.display.flip()
    except pygame.error as e:
        print(f"Pygame error: {e}")

def plot_results(transformed_target, radius, waypoints):
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))

    axs[0].plot(time_values, throttle_values, label="Throttle", color="r")
    axs[0].set_ylabel("Throttle")
    axs[0].legend()
    axs[0].grid()

    axs[1].plot(time_values, steering_values, label="Steering", color="b")
    axs[1].set_ylabel("Steering Angle")
    axs[1].legend()
    axs[1].grid()

    axs[2].plot(x_values, y_values, label="X-Y Path", color="g")
    target_x, target_y = transformed_target[0], transformed_target[1]
    circle = plt.Circle((target_x, target_y), radius, color='orange', fill=False, linestyle='--', label='Acceptable Bound')
    axs[2].add_patch(circle)

    axs[2].scatter(x_values[0], y_values[0], color="green", marker="o", s=100, label="Start")  
    axs[2].scatter(target_x, target_y, color="red", marker="x", s=100, label="Target")  

    waypoint_x = [wp[0] for wp in waypoints]
    waypoint_y = [wp[1] for wp in waypoints]
    axs[2].scatter(waypoint_x, waypoint_y, color="blue", marker="*", label="Waypoints")

    axs[2].set_xlabel("X Position")
    axs[2].set_ylabel("Y Position")
    axs[2].legend()
    axs[2].grid()

    plt.tight_layout()
    plt.show()

def simulate():
    global simulation_running

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    blueprint = world.get_blueprint_library().filter('vehicle.lincoln.mkz_2017')[0]
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        print("No spawn points available!")
        return
    vehicle = world.spawn_actor(blueprint, spawn_points[0])
    # Spawn a static vehicle in front of the main vehicle
    static_blueprint = world.get_blueprint_library().filter('vehicle.tesla.model3')[0]
    static_transform = carla.Transform(
        location=spawn_points[0].location + carla.Location(x=8.0, z=0),  # 8 meters ahead
        rotation=spawn_points[0].rotation
    )
    static_vehicle = world.try_spawn_actor(static_blueprint, static_transform)
    if static_vehicle:
        static_vehicle.set_autopilot(False)
        static_vehicle.set_simulate_physics(False)  # Make it static


    camera_blueprint = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_transform = carla.Transform(carla.Location(x=-5, z=2))
    camera = world.spawn_actor(camera_blueprint, camera_transform, attach_to=vehicle)

    pygame.init()
    display = pygame.display.set_mode((800, 600), pygame.HWSURFACE | pygame.DOUBLEBUF)
    camera.listen(lambda image: process_image(image, display))

    model = KinematicBicycleModel()
    controller = NMPCController(model)

    waypoints = generate_reference_track()
    transform = vehicle.get_transform()
    velocity = vehicle.get_velocity()
    vehicle_x = transform.location.x
    vehicle_y = transform.location.y
    Vehicle_inital_state = np.array([
        vehicle_x,
        vehicle_y,
        transform.rotation.yaw,
        math.sqrt(velocity.x**2 + velocity.y**2)
    ])

    transformed_waypoints = [[wp[0] + vehicle_x, wp[1] + vehicle_y] for wp in waypoints]
    draw_waypoints(world, transformed_waypoints)  # <- WAYPOINT VISUALIZATION

    transformed_waypoints.pop()
    current_target = np.array([
        waypoints[0][0] + vehicle_x,
        waypoints[0][1] + vehicle_y,
        Vehicle_inital_state[2],
        0.5
    ])

    waypoint_index = 0
    radius = 0.8
    T = 10000

    for t in range(T):
        if not simulation_running:
            break
        transform = vehicle.get_transform()
        velocity = vehicle.get_velocity()
        vehicle_state = np.array([
            transform.location.x,
            transform.location.y,
            transform.rotation.yaw,
            math.sqrt(velocity.x**2 + velocity.y**2)
        ])

        State_Devitation = vehicle_state - Vehicle_inital_state
        Target_Devitation = current_target - Vehicle_inital_state
        State_Devitation[2] = 0
        Target_Devitation[2] = 0

        control = controller.solve(State_Devitation, Target_Devitation)
        carla_control = carla.VehicleControl()
        carla_control.steer = float(control[0])
        carla_control.throttle = float(max(0, control[1]))
        vehicle.apply_control(carla_control)

        time_values.append(t)
        throttle_values.append(control[1])
        steering_values.append(control[0])
        x_values.append(vehicle_state[0])
        y_values.append(vehicle_state[1])

        if np.linalg.norm(vehicle_state[:2] - current_target[:2]) < radius:
            waypoint_index += 1
            if waypoint_index >= len(waypoints):
                print("All waypoints reached!")
                break
            Vehicle_inital_state = vehicle_state
            current_target = np.array([
                waypoints[waypoint_index][0] + Vehicle_inital_state[0],
                waypoints[waypoint_index][1] + Vehicle_inital_state[1],
                vehicle_state[2],
                1.0
            ])

    vehicle.destroy()
    pygame.quit()
    plot_results(current_target, radius, transformed_waypoints)

# Run the simulation
simulate()
