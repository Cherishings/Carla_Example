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
velocities = []
states_log = []
simulation_running = True  # Flag to detect Ctrl+C

# Kinematic Bicycle Model with RK4 integration
class KinematicBicycleModel:
    def __init__(self, lf=-0.090769015, lr=1.4178275, dt=0.1):
        self.lf = lf
        self.lr = lr
        self.dt = dt

    def step_rk4(self, state, control):
        X = state[0]
        Y = state[1]
        psi = state[2]
        V = state[3]

        delta = control[0]
        throttle = control[1]

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
        self.targets = self.opti.parameter(4, N+1)  # Now targets over horizon

        Qx = np.diag([30, 30, 0, 20])
        Qu = np.diag([100, 100])
        Q_Terminal = np.diag([10, 10, 0, 0])

        Qx_casadi = ca.DM(Qx)
        Qu_casadi = ca.DM(Qu)
        Q_Terminal_casadi = ca.DM(Q_Terminal)

        lambda_steering_rate = 120
        lambda_throttle_rate = 120

        cost = 0
        for k in range(N):
            state_error = self.X[:, k] - self.targets[:, k]
            control_effort = self.U[:, k]

            cost += ca.mtimes([state_error.T, Qx_casadi, state_error])
            cost += ca.mtimes([control_effort.T, Qu_casadi, control_effort])

            if k < N - 1:
                delta_diff = self.U[0, k+1] - self.U[0, k]
                throttle_diff = self.U[1, k+1] - self.U[1, k]
                cost += lambda_steering_rate * ca.mtimes([delta_diff.T, delta_diff])
                cost += lambda_throttle_rate * ca.mtimes([throttle_diff.T, throttle_diff])

        terminal_error = self.X[:, -1] - self.targets[:, -1]
        cost += ca.mtimes([terminal_error.T, Q_Terminal_casadi, terminal_error])

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

    def solve(self, state, reference_horizon):
        self.opti.set_value(self.X_init, state)
        self.opti.set_value(self.targets, reference_horizon)
        sol = self.opti.solve()
        return sol.value(self.U[:, 0])

def signal_handler(sig, frame):
    global simulation_running
    print("\nSimulation stopped. Saving data and plotting results...")
    simulation_running = False

signal.signal(signal.SIGINT, signal_handler)

def generate_reference_track():
    raw_waypoints =  np.array([
    [0, 0], [10, 0], [20, 5], [30, 10], [40, 15], 
    [50, 15], [60, 10], [70, 0], [80, 5], [90, 8], [105,8]
])
    x, y = raw_waypoints[:, 0], raw_waypoints[:, 1]
    tck, u = splprep([x, y], s=0)
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

def simulate():
    reference_path, waypoints = generate_reference_track()
    model = KinematicBicycleModel()
    controller = NMPCController(model)

    state = np.array([0.0, 0.0, 0.0, 0.0])
    dt = model.dt

    t = 0.0
    path_index = 0

    while simulation_running and path_index < len(reference_path):
        ref_horizon = build_reference_horizon(reference_path, path_index, controller.N)
        control = controller.solve(state, ref_horizon)
        state = np.array(model.step_rk4(state, control)).flatten()

        x_values.append(state[0])
        y_values.append(state[1])
        time_values.append(t)
        throttle_values.append(control[1])
        steering_values.append(control[0])
        velocities.append(state[3])
        states_log.append(state.copy())

        print("----------------------------------------")
        print(f"Time step: {t:.2f}")
        print(f"Position: X={state[0]:.2f}, Y={state[1]:.2f}")
        print(f"Velocity: V={state[3]:.2f}")
        print(f"Steering: {control[0]:.2f}, Throttle: {control[1]:.2f}")
        print("----------------------------------------")

        t += dt

        distance_to_target = np.linalg.norm(state[:2] - reference_path[path_index][:2])
        if distance_to_target < 0.8:
            print("Target Reached!")
            path_index += 1

def plot_results():
    reference_path, waypoints = generate_reference_track()

    # Plot the reference path and the vehicle's trajectory
    plt.figure()
    plt.plot(reference_path[:, 0], reference_path[:, 1], 'b--', label='Reference Track')
    plt.plot(x_values, y_values, 'r-', label='Vehicle Path')
    plt.scatter(waypoints[:, 0], waypoints[:, 1], c='black', marker='x', s=80, label='Target Waypoints')

    # Define the acceptance bound as a circle around the last waypoint
    final_waypoint = reference_path[-1]  # Final target waypoint
    acceptance_radius = 1.0  # The radius for acceptance (tolerance)
    
    # Create a circle for the acceptance bound
    circle = plt.Circle((final_waypoint[0], final_waypoint[1]), acceptance_radius, color='g', fill=False, linestyle='--', label='Acceptance Bound')
    plt.gca().add_patch(circle)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Trajectoy Tracking')
    plt.legend()
    plt.grid(True)

    # Plot other results (steering, throttle, state logs)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(time_values[:len(steering_values)], steering_values, label='Steering Angle')
    plt.xlabel('Time (s)')
    plt.ylabel('Steering')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(time_values[:len(throttle_values)], throttle_values, label='Throttle')
    plt.xlabel('Time (s)')
    plt.ylabel('Throttle')
    plt.grid(True)
    plt.legend()

    states_np = np.array(states_log)
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(time_values, states_np[:, 0], label="X Position")
    plt.xlabel("Time (s)")
    plt.ylabel("X")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(time_values, states_np[:, 1], label="Y Position")
    plt.xlabel("Time (s)")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(time_values, states_np[:, 3], label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    simulate()
    plot_results()
