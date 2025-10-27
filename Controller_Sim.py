import numpy as np
import casadi as ca
import matplotlib.pyplot as plt


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

        print(f"K2: {k2}, K3: {k3}")
        # Compute the next state using the RK4 formula
        next_state = state + (self.dt / 6) * (k1 + 2*k2 + 2*k3 + k4) # check !
        print(next_state)
        return next_state


class NMPCController:
    def __init__(self, model, N=5, dt=0.1):  # Prediction horizon 10.
        self.model = model
        self.N = N
        self.dt = dt
        
        # Define optimization variables
        self.opti = ca.Opti()
        self.X = self.opti.variable(4, N+1)  # States [X, Y, psi, V]
        self.U = self.opti.variable(2, N)    # Controls [steering, throttle]

        # Target State
        self.target = self.opti.parameter(4)  

        # Define weighting matrices for cost function
        Qx = np.diag([20, 20, 0, 20])  # Penalizing (X, Y, Psi, Velocity)
        Qu = np.diag([5, 20])  # Penalizing (Steering, Throttle)
        
        # Terminal penalty matrix (Emphasizing position and velocity, reduce yaw angle)
        Q_Terminal = np.diag([20, 20, 0, 0])  # Emphasizing position (X, Y), no penalty on yaw angle
        
        Qx_casadi = ca.DM(Qx)
        Qu_casadi = ca.DM(Qu)
        Q_Terminal_casadi = ca.DM(Q_Terminal)

        # Penalty for steering rate of change
        lambda_steering_rate = 120  
        lambda_throttle_rate = 120
        
        # Objective function (Cost Function)
        cost = 0
        for k in range(N):
            state_error = self.X[:, k] - self.target  # State deviation from target
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
        cost += ca.mtimes([(self.X[:, -1] - self.target).T, Q_Terminal_casadi, (self.X[:, -1] - self.target)])

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

    def solve(self, state, target):
        """Solve the NMPC problem and return optimal control input"""
        self.opti.set_value(self.X_init, state)
        self.opti.set_value(self.target, target)

        sol = self.opti.solve()
        return sol.value(self.U[:, 0])  # Return first control action

# Generate a set of reference waypoints for the vehicle to follow
def generate_reference_track():
    waypoints = np.array([
        [0, 0], [5, 1], [10, 3], [15, 6], [20, 10], [25, 15], [30, 21]
    ])
    return waypoints

# Function to generate waypoints along a circular track
def generate_banana_path(length=75, num_points=15, curvature_factor=0.07):
    # Generate straight x values
    x = np.linspace(0, length, num_points)
    
    # Apply curvature as a sinusoidal function for the y values (gradual curvature)
    y = np.sin(curvature_factor * x)
    
    # Combine into a set of waypoints
    waypoints = np.column_stack((x, y))
    
    return waypoints
# Simulation and Testing
def simulate():
    model = KinematicBicycleModel()
    controller = NMPCController(model)
    waypoints = generate_banana_path() #generate_reference_track()  
    # waypoints = np.array([[15,1]]) # [3,-1]
    # Initial State
    state = np.array([0.0, 0.0, 0, 0])
    # target = np.array([15, 2, 0, 1])  # Target [X, Y, psi, V], example from data set 12.9373288434962	1.39597286603004
    target_index = 0 
    # Testing
    # state = np.array([106, -12.71, -89.61, 0])
    # target = np.array([107, -12, -89.61, 1])

    states = [state]
    controls = []
    targets = []
    T = 500  # Total simulation steps
    for t in range(T):

         # Set target to the next waypoint
        if target_index < len(waypoints):
            target = np.array([waypoints[target_index][0], waypoints[target_index][1], 0, 1])
            targets.append(target[:2])  # Store for plotting
        else:
            break  # Stop when all waypoints are reached

        control = controller.solve(state, target)
        state = model.step_rk4(state, control)  # Use RK4 version for simulation

        # Convert CasADi matrix (state) to NumPy array and extract values
        state_values = state.full().flatten()  # Use full() to convert DM to NumPy array

        #noise_std = np.array([0.01, 0.01, 0.001, 0.01])  # [X, Y, psi, V] noise std devs
        # state_values += np.random.normal(0, noise_std)


        states.append(state_values)
        controls.append(control)

        # Print time step and data
        print("----------------------------------------")
        print(f"Time step: {t}")
        print(f"Position: X={state_values[0]:.2f}, Y={state_values[1]:.2f}")
        print(f"Velocity: V={state_values[3]:.2f}")
        print(f"Steering: {control[0]:.2f}, Throttle: {control[1]:.2f}")
        print("----------------------------------------")

        # Stop if close to target
        if np.linalg.norm(state_values[:2] - target[:2]) < 0.25:
            print("Target Reached!")
            target_index += 1
            # break

    states = np.array(states)  # Ensure states is a numpy array
    controls = np.array(controls)  # Ensure controls is a numpy array
    targets = np.array(targets)  
    # Plot Reference Track and Vehicle Path
    plt.figure()
    plt.plot(waypoints[:, 0], waypoints[:, 1], 'bo-', label="Reference Track")  # Waypoints
    plt.plot(states[:, 0], states[:, 1], 'r-', label="Vehicle Path")  # Vehicle path
    plt.scatter(waypoints[:, 0], waypoints[:, 1], color='blue', marker='x', s=100, label="Targets")
    #plt.scatter(targets[:, 0], targets[:, 1], color='green', marker='o', s=50, label="Target Points")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("MPC Vehicle Tracking")
    plt.legend()
    plt.grid()
    
    # Plot Control Inputs
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(range(len(controls)), [c[0] for c in controls], label="Steering Angle (rad)")
    plt.xlabel("Time Step")
    plt.ylabel("Steering")
    plt.legend()
    plt.grid()
    
    plt.subplot(2, 1, 2)
    plt.plot(range(len(controls)), [c[1] for c in controls], label="Throttle")
    plt.xlabel("Time Step")
    plt.ylabel("Throttle")
    plt.legend()
    plt.grid()
    
    # Plot States
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(range(len(states)), states[:, 0], label="X Position")
    plt.xlabel("Time Step")
    plt.ylabel("X")
    plt.legend()
    plt.grid()
    
    plt.subplot(3, 1, 2)
    plt.plot(range(len(states)), states[:, 1], label="Y Position")
    plt.xlabel("Time Step")
    plt.ylabel("Y")
    plt.legend()
    plt.grid()
    
    plt.subplot(3, 1, 3)
    plt.plot(range(len(states)), states[:, 3], label="Velocity")
    plt.xlabel("Time Step")
    plt.ylabel("V")
    plt.legend()
    plt.grid()
    
    plt.show()

simulate()