import casadi as ca
import numpy as np

# Robot parameters
dt = 0.1  # Discretization time step
N = 20    # Prediction horizon
v_max = 1.0  # Max linear velocity
omega_max = 1.0  # Max angular velocity
d_safe = 0.5  # Safe distance from obstacles

# Define symbolic variables
x = ca.SX.sym('x')  # x position
y = ca.SX.sym('y')  # y position
theta = ca.SX.sym('theta')  # orientation
v = ca.SX.sym('v')  # linear velocity
omega = ca.SX.sym('omega')  # angular velocity
states = ca.vertcat(x, y, theta)
controls = ca.vertcat(v, omega)
n_states = states.size()[0]
n_controls = controls.size()[0]

# Define the system dynamics (differential drive model)
rhs = ca.vertcat(v * ca.cos(theta),  # dx/dt
                 v * ca.sin(theta),  # dy/dt
                 omega)              # dtheta/dt

# Define the CasADi function for dynamics
f = ca.Function('f', [states, controls], [rhs])

# Objective weights
Q_goal = np.diag([10, 10, 1])  # Weight for the goal (penalize x, y, theta deviations)
R_control = np.diag([0.5, 0.5])  # Weight for control effort

# Define the optimization variables for MPC
U = ca.SX.sym('U', n_controls, N)  # Control input over the horizon
X = ca.SX.sym('X', n_states, N + 1)  # State trajectory over the horizon

# Initialize the cost function
cost = 0

# Target position (goal)
x_goal = np.array([5, 5])

# Obstacle positions
obstacles = np.array([[2, 2], [4, 3]])

# Set up the optimization problem
g = []  # Constraints list
g.append(X[:, 0] - states)  # Initial state constraint

# Loop over the prediction horizon to compute the cost and apply dynamics
for k in range(N):
    # Update the cost function
    cost += ca.mtimes([(X[:2, k] - x_goal).T, Q_goal[:2, :2], X[:2, k] - x_goal])  # Goal reaching cost
    cost += ca.mtimes([U[:, k].T, R_control, U[:, k]])  # Control effort cost
    
    # Obstacle avoidance cost (soft constraints)
    for obs in obstacles:
        dist_to_obstacle = ca.sqrt((X[0, k] - obs[0])**2 + (X[1, k] - obs[1])**2)
        cost += 1000 * ca.if_else(dist_to_obstacle < d_safe, 1/dist_to_obstacle, 0)  # Penalize when near obstacles
    
    # System dynamics constraint (x_{k+1} = f(x_k, u_k))
    x_next = X[:, k] + dt * f(X[:, k], U[:, k])
    g.append(X[:, k + 1] - x_next)

# Terminal cost (distance to goal)
cost += ca.mtimes([(X[:2, N] - x_goal).T, Q_goal[:2, :2], X[:2, N] - x_goal])

# Control constraints
for k in range(N):
    g.append(U[0, k] - v_max)  # Velocity limit
    g.append(-U[0, k])         # Minimum velocity
    g.append(U[1, k] - omega_max)  # Angular velocity limit
    g.append(-U[1, k])         # Minimum angular velocity

# Define decision variables
OPT_variables = ca.vertcat(ca.reshape(U, -1, 1), ca.reshape(X, -1, 1))

# Define the problem (minimize cost subject to constraints)
nlp_prob = {'f': cost, 'x': OPT_variables, 'g': ca.vertcat(*g)}

# Define solver options
opts = {'ipopt.print_level': 0, 'ipopt.max_iter': 100, 'ipopt.tol': 1e-4, 'print_time': 0}
solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

# Initial conditions
x_init = np.array([0, 0, 0])  # Start at the origin
x_goal = np.array([5, 5])  # Goal position

# Set initial guess for optimization variables
U0 = np.zeros((n_controls, N))
X0 = np.tile(x_init, (N+1, 1)).T

# Decision variable initial guess
initial_guess = np.concatenate([U0.reshape((-1,), order='F'), X0.reshape((-1,), order='F')])

# Solve the problem
sol = solver(x0=initial_guess, lbx=-np.inf, ubx=np.inf, lbg=0, ubg=0)

# Extract the optimal solution
u_opt = sol['x'].full().flatten()[:N * n_controls].reshape((n_controls, N))
x_opt = sol['x'].full().flatten()[N * n_controls:].reshape((n_states, N + 1))

print("Optimal control inputs: ", u_opt)
print("Optimal state trajectory: ", x_opt)
