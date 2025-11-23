import gymnasium as gym
from stable_baselines3 import TD3
import numpy as np
import torch
import matplotlib.pyplot as plt

# Load model and env
model = TD3.load("td3_pendulum")
env = gym.make("Pendulum-v1", render_mode="rgb_array")

# Function to normalize pendulum state
def normalize_pendulum_state(x):
    """
    Takes raw [theta, theta_dot] and returns [cos(theta), sin(theta), theta_dot] tensor
    """
    x = x.float()  # ensure float32
    return torch.stack([torch.cos(x[0]), torch.sin(x[0]), x[1]])

# Create a grid over theta and theta_dot
n_theta = 200
n_theta_dot = 100
theta_grid = np.linspace(-2*np.pi, 2*np.pi, n_theta)
theta_dot_grid = np.linspace(-8, 8, n_theta_dot)

# Create meshgrid
Theta, Theta_dot = np.meshgrid(theta_grid, theta_dot_grid)
grid_points = np.stack([Theta.ravel(), Theta_dot.ravel()], axis=1)

# Normalize all states at once
states_tensor = torch.stack(
    [normalize_pendulum_state(torch.tensor(x, dtype=torch.float32)) for x in grid_points]
)
# Shape: (n_theta * n_theta_dot, 3)

# Get actions from the actor for all states
with torch.no_grad():
    actions_tensor = model.actor(states_tensor)  # shape: (N, action_dim)
    # Compute Q-values from critic (use qf1)
    q_values = model.critic.qf1(torch.cat([states_tensor, actions_tensor], dim=1))

# Reshape Q-values to grid
Q_grid = q_values.numpy().reshape(n_theta_dot, n_theta)

# Plot heatmap
plt.figure(figsize=(10, 6))
plt.imshow(
    Q_grid,
    extent=[-2*np.pi, 2*np.pi, -8, 8],
    origin="lower",
    aspect="auto",
    cmap="viridis",
)
plt.colorbar(label="Q-value")
plt.xlabel("Theta (rad)")
plt.ylabel("Theta dot (rad/s)")
plt.title("TD3 Critic Q-value Heatmap for Pendulum-v1")
plt.show()