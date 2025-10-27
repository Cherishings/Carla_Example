import numpy as np
import casadi as ca
import pygame
import carla
import math
import random
import time
import signal
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
# For debugging 

time_values = []
throttle_values = []
steering_values = []
x_values = []
y_values = []
simulation_running = True  



# Kinematic Bicycle Model with RK4 integration
class KinematicBicycleModel:
    def __init__(self, lf=-0.090769015, lr=1.4178275, dt=0.1):  # Carla 20 FPS (self, lf=0.9, lr=2.096, dt=0.1): 
        self.lf = lf
        self.lr = lr
        self.dt = dt  # Time step for RK4 integration

    def step_rk4(self, state, control):
        """Compute next state using kinematic bicycle model with RK4 integration."""
        # Extract symbolic state variables (using CasADi operations)
        X = state[0]
        Y = state[1]
        psi = state[2]
        V = state[3]
        
        delta = control[0]
        throttle = control[1]
        
        # Define the system of equations (state derivatives)
        def f(state, control):
            X, Y, psi, V = state[0], state[1], state[2], state[3]
            delta, throttle = control[0], control[1]
            beta = ca.atan((self.lr * ca.tan(delta)) / (self.lf + self.lr))
            return ca.vertcat(
                V * ca.cos(psi + beta),       # dx/dt
                V * ca.sin(psi + beta),       # dy/dt
                (V * ca.tan(delta)) / (self.lf + self.lr),  # dpsi/dt
                throttle                       # dV/dt (velocity change)
            )

        # Compute k1, k2, k3, k4 for RK4
        k1 = f(state, control)
        k2 = f(state + 0.5 * self.dt * k1, control)
        k3 = f(state + 0.5 * self.dt * k2, control)
        k4 = f(state + self.dt * k3, control)

        # Compute the next state using the RK4 formula
        next_state = state + (self.dt / 6) * (k1 + 2*k2 + 2*k3 + k4) # check !
        print(next_state)
        return next_state


class NMPCController:
    def __init__(self, model, N=10, dt=0.1):  # Prediction horizon 10.
        self.model = model
        self.N = N
        self.dt = dt
        
        # Define optimization variables
        self.opti = ca.Opti()
        self.X = self.opti.variable(4, N+1)  # States [X, Y, psi, V]
        self.U = self.opti.variable(2, N)    # Controls [steering, throttle]

        # Target State
        self.targets = self.opti.parameter(4, N+1)  # Trajectory target over horizon  

        # Define weighting matrices for cost function
        Qx = np.diag([30, 30, 0, 10])  # Penalizing (X, Y, Psi, Velocity)
        Qu = np.diag([10, 10])  # Penalizing (Steering, Throttle)
        
        # Terminal penalty matrix
        Q_Terminal = np.diag([10, 10, 0, 0])  # Emphasizing position (X, Y), no penalty on yaw angle
        
        Qx_casadi = ca.DM(Qx)
        Qu_casadi = ca.DM(Qu)
        Q_Terminal_casadi = ca.DM(Q_Terminal)

        # Penalty for steering rate of change
        lambda_steering_rate = 2 
        lambda_throttle_rate = 2
        
        # Objective function (Cost Function)
        cost = 0
        for k in range(N):
            state_error = self.X[:, k] - self.targets[:, k] # State deviation from target
            control_effort = self.U[:, k]  # Control input

            cost += ca.mtimes([state_error.T, Qx_casadi, state_error])  # State penalty
            cost += ca.mtimes([control_effort.T, Qu_casadi, control_effort])  # Control penalty
            
            # Add penalty for rate of change of steering angle (delta[k+1] - delta[k])
            if k < N - 1:  # Avoid out-of-bounds for the last control input
                delta_diff = self.U[0, k+1] - self.U[0, k]  # d_delta/dt
                throttle_diff = self.U[1, k+1] - self.U[1, k]  # Throttle rate
                cost += lambda_steering_rate * ca.mtimes([delta_diff.T, delta_diff])  # Steering rate penalty
                cost += lambda_throttle_rate * ca.mtimes([throttle_diff.T, throttle_diff])

        # Terminal cost (final state tracking, using separate terminal penalty matrix)
        terminal_error = self.X[:, -1] - self.targets[:, -1]
        cost += ca.mtimes([terminal_error.T, Q_Terminal_casadi, terminal_error])


        self.opti.minimize(cost)

        # Dynamics Constraints (RK4 Integration)
        for k in range(N):
            X_next = self.model.step_rk4(self.X[:, k], self.U[:, k])  # RK4 integration
            self.opti.subject_to(self.X[:, k+1] == X_next)

        # Control Constraints
        self.opti.subject_to(self.opti.bounded(-1, self.U[0, :], 1))  # Steering angle
        self.opti.subject_to(self.opti.bounded(0, self.U[1, :], 1))  # Throttle

        # Initial Condition Constraint
        self.X_init = self.opti.parameter(4)
        self.opti.subject_to(self.X[:, 0] == self.X_init)

        # Solver Settings
        opts = {"ipopt.print_level": 0, "print_time": 0}
        self.opti.solver("ipopt", opts)

    def solve(self, state, reference_horizon):
        self.opti.set_value(self.X_init, state)
        self.opti.set_value(self.targets, reference_horizon)
        for k in range(self.N):
            print(f"Self_Target[{k}]:", reference_horizon[:, k])
        sol = self.opti.solve()
        return sol.value(self.U[:, 0])  # Return first control action

# Handle Ctrl+C to stop safely
def signal_handler(sig, frame):
    global simulation_running
    print("\nSimulation stopped. Saving data and plotting results...")
    simulation_running = False  # Stop loop safely

signal.signal(signal.SIGINT, signal_handler)  # Listen for Ctrl+C

# Testing Path tracking
'''
def generate_reference_track():
    waypoints = np.array([
        [0,0], [30, 0], [60,1] # , [10, -2]# [0, 0], [5, 1], [10, 3], [15, 6], [20, 10], [25, 15], [30, 21]
    ])
    return waypoints
'''

def generate_reference_track():
    raw_waypoints =  np.array([
    [0, 0], [10, 3]
])
    x, y = raw_waypoints[:, 0], raw_waypoints[:, 1]
    tck, u = splprep([x, y], s=0, k=1)
    u_fine = np.linspace(0, 1, 200)
    x_smooth, y_smooth = splev(u_fine, tck)
    return np.column_stack((x_smooth, y_smooth)), raw_waypoints

def build_reference_horizon(reference_path, current_index, horizon, target_speed=1.0):
    ref = np.zeros((4, horizon + 1))
    for i in range(horizon + 1):
        idx = min(current_index + i, len(reference_path) - 1)
        ref[0, i] = reference_path[idx][0]
        ref[1, i] = reference_path[idx][1]
        ref[2, i] = 0.0  # heading
        ref[3, i] = target_speed
    return ref

# Simulation and Carla Setup
def simulate():

    # Initalization
    global simulation_running

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # Get vehicle blueprint and spawn the vehicle
    blueprint = world.get_blueprint_library().filter('vehicle.lincoln.mkz_2017')[0]  # Choose a vehicle
    
    
    # Find a random spawn point or a safe one
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        print("No spawn points available!")
        return
    
    spawn_points = world.get_map().get_spawn_points()
    fixed_spawn_point = spawn_points[0]  # Always use the first spawn point

    vehicle = world.spawn_actor(blueprint, fixed_spawn_point)

        
    # Camera Setup
    camera_blueprint = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_transform = carla.Transform(carla.Location(x=-5, z=2))  # Behind the vehicle
    camera = world.spawn_actor(camera_blueprint, camera_transform, attach_to=vehicle)
    
    # Pygame setup for rendering camera feed
    pygame.init()
    display = pygame.display.set_mode((800, 600), pygame.HWSURFACE | pygame.DOUBLEBUF)
    camera.listen(lambda image: process_image(image, display))

    model = KinematicBicycleModel()
    controller = NMPCController(model)

    # waypoints = generate_reference_track()
    reference_path, raw_waypoints = generate_reference_track()
    # Initial State
    state = np.array([0, 0, 0, 0])  # [X, Y, psi, V]
    # target = np.array([3, 2, 0, 1])  # Target [X, Y, psi, V]
    radius = 0.8
    states = [state]
    controls = []
    # Get true vehicle state from Carla
    transform = vehicle.get_transform()
    velocity = vehicle.get_velocity()
    Vehicle_inital_state = np.array([
        transform.location.x,    # X
        transform.location.y,    # Y
        transform.rotation.yaw,  # Psi
        math.sqrt(velocity.x**2 + velocity.y**2)  # V (Speed), we drop z coordinate
    ])
    vehicle_x = transform.location.x
    vehicle_y = transform.location.y
    Vehicle_inital_state = Vehicle_inital_state
    # target[:2] = Vehicle_inital_state[:2] + target[:2]
    # transformed_target = np.array([target[0], target[1], Vehicle_inital_state[2], target[3]])
    T = 10000  # Total simulation steps
    path_index = 0
    for t in range(T):
        # Get true vehicle state from Carla
        if not simulation_running:
            break
        transform = vehicle.get_transform()
        velocity = vehicle.get_velocity()
        vehicle_state = np.array([
            transform.location.x,    # X
            transform.location.y,    # Y
            transform.rotation.yaw,  # Psi
            math.sqrt(velocity.x**2 + velocity.y**2)  # V (Speed)
        ])
        State_Devitation = vehicle_state - Vehicle_inital_state
        ref_horizon = build_reference_horizon(reference_path, path_index, controller.N)
        control = controller.solve(State_Devitation, ref_horizon)
        print(f"Control: {control}")
        # state = model.step_rk4(vehicle_state, control)  # Use RK4 version for simulation

        # Convert CasADi matrix (state) to NumPy array and extract values
        state_values = State_Devitation # state.full().flatten()  # Use full() to convert DM to NumPy array
        
        # Store data
        time_values.append(t)
        throttle_values.append(control[1])
        steering_values.append(control[0])
        x_values.append(State_Devitation[0])
        y_values.append(State_Devitation[1])
        states.append(state_values)
        controls.append(control)

        print(f"State t={t}: X={State_Devitation[0]:.2f}, Y={State_Devitation[1]:.2f}, Psi={State_Devitation[2]:.2f}, V={State_Devitation[3]:.2f}")
        print(f"Throttle: {control[1]}")
        # Convert NMPC control output to CARLA control format
        carla_control = carla.VehicleControl()
        carla_control.steer = float(control[0])  # Steering (bounded between -1 and 1)
        carla_control.throttle = float(max(0, control[1]))  # Ensure throttle is positive
        vehicle.apply_control(carla_control)
        # Stop if close to target
        if np.linalg.norm(state_values[:2] - reference_path[path_index][:2]) < radius:
            print(f"Reference point {path_index + 1} reached!")
            path_index += 1
            if path_index >= len(reference_path):
                print("All reference points reached!")
                break


    states = np.array(states)  # Ensure states is a numpy array
    controls = np.array(controls)  # Ensure controls is a numpy array
    vehicle.destroy()
    pygame.quit()
    plot_results(reference_path, radius, raw_waypoints, Vehicle_inital_state)

def process_image(image, display):
    if not pygame.get_init():  # Check if Pygame is still running
        return  

    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

    try:
        display.blit(surface, (0, 0))
        pygame.display.flip()
    except pygame.error as e:
        print(f"Pygame error: {e}")  # Debugging message if something goes wrong


def plot_results(reference_path, radius, waypoints, vehicle_initial_state):
    """Plots the collected simulation data after the run."""
    # Shift final target into local (relative) frame for plotting
    target_local_x = reference_path[-1][0] - vehicle_initial_state[0]
    target_local_y = reference_path[-1][1] - vehicle_initial_state[1]

    fig, axs = plt.subplots(3, 1, figsize=(10, 8))

    # Plot Throttle
    axs[0].plot(time_values, throttle_values, label="Throttle", color="r")
    axs[0].set_ylabel("Throttle")
    axs[0].legend()
    axs[0].grid()

    # Plot Steering
    axs[1].plot(time_values, steering_values, label="Steering", color="b")
    axs[1].set_ylabel("Steering Angle")
    axs[1].legend()
    axs[1].grid()

    # Plot X-Y Trajectory
    axs[2].plot(x_values, y_values, label="X-Y Path", color="g")

    # Add the disc (radius) around the **last waypoint**
    last_wp_x, last_wp_y = waypoints[-1]
    circle = plt.Circle((last_wp_x, last_wp_y), radius, color='orange', fill=False, linestyle='--', label='Acceptable Bound')
    axs[2].add_patch(circle)

    # Mark the start and target
    axs[2].scatter(x_values[0], y_values[0], color="green", marker="o", s=100, label="Start")  
    axs[2].scatter(last_wp_x, last_wp_y, color="red", marker="x", s=100, label="Target")
    # axs[2].scatter(target_local_x, target_local_y, color="red", marker="x", s=100, label="Target")  

    # Plot the waypoints as blue dots
    waypoint_x = [wp[0] for wp in waypoints]
    waypoint_y = [wp[1] for wp in waypoints]
    axs[2].scatter(waypoint_x, waypoint_y, color="blue", marker="*", label="Waypoints")

    axs[2].set_xlabel("X Position")
    axs[2].set_ylabel("Y Position")
    axs[2].legend()
    axs[2].grid()

    plt.tight_layout()
    plt.show()


simulate()
