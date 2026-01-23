# Combined figure with trajectory and information rate over same timespan

import numpy as np
import matplotlib.pyplot as plt
from jax import numpy as jnp

def double_pendulum_positions(theta0, theta1, l0=1.0, l1=1.0):
    x0 = l0 * np.sin(theta0)
    y0 = -l0 * np.cos(theta0)
    x1 = x0 + l1 * np.sin(theta0 + theta1)
    y1 = y0 - l1 * np.cos(theta0 + theta1)
    return x0, y0, x1, y1

def plot_combined_trajectory_info(
    theta0,
    theta1,
    hist,
    dt=0.01,
    sample=100,
    l0=1.0,
    l1=1.0,
    link_color="#c49a6c",
    joint_color="#8b5a2b",
    ground_color="#7fbf7f",
    sampling_strategy="uniform",
    filename=None
):
    """
    Combined figure showing:
    - Top: Trajectory visualization over time
    - Bottom: Information rate over time
    Both with aligned time axes.
    """
    x0, y0, x1, y1 = double_pendulum_positions(theta0, theta1, l0, l1)
    
    # Time array
    t = np.arange(len(theta0)) * dt
    total_time = t[-1]
    
    # Choose sampling strategy for trajectory
    if sampling_strategy == "sqrt":
        t_normalized = np.linspace(0, 1, sample) ** 2
        idx = (t_normalized * (len(theta0) - 1)).astype(int)
    elif sampling_strategy == "log":
        t_normalized = np.logspace(0, 1, sample) - 1
        t_normalized = t_normalized / t_normalized[-1]
        idx = (t_normalized * (len(theta0) - 1)).astype(int)
    else:
        idx = np.linspace(0, len(theta0) - 1, sample).astype(int)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), 
                                     gridspec_kw={'height_ratios': [1, 1]})
    
    # Top subplot: trajectory
    # Collect all y positions to determine appropriate ground level
    all_y = []
    for i in idx:
        all_y.extend([0, y0[i], y1[i]])
    ground_y = min(all_y) - 0.3

    # Use total_time as the width (this is the key fix!)
    total_width = total_time
    
    for k, i in enumerate(idx):
        # Map k to horizontal position (same as original code)
        x_offset = (k / (sample - 1)) * total_width
        
        # first link
        ax1.plot(
            [x_offset, x_offset + x0[i]],
            [0, y0[i]],
            lw=1.5,
            color=link_color,
            alpha=0.9,
        )
        
        # second link
        ax1.plot(
            [x_offset + x0[i], x_offset + x1[i]],
            [y0[i], y1[i]],
            lw=1.5,
            color=link_color,
            alpha=0.9,
        )
        
        # joints
        ax1.scatter(
            [x_offset, x_offset + x0[i], x_offset + x1[i]],
            [0, y0[i], y1[i]],
            s=10,
            color=joint_color,
            zorder=3,
        )
    
    # ground line
    ax1.plot(
        [-0.3, total_width + 0.3],
        [ground_y, ground_y],
        color=ground_color,
        lw=4,
    )
    
    ax1.set_aspect("equal")
    ax1.set_xlim(-0.3, total_width + 0.3)
    ax1.set_ylim(ground_y - 0.2, 2.2)
    ax1.set_ylabel('Position')
    ax1.set_xticks([])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    
    # Bottom subplot: information rate
    ax2.plot(t, hist, linewidth=1.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('bits/s')
    ax2.set_xlim(-0.3, total_time + 0.3)
    ax2.grid(True, alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()


# ---- Load and process data ----
X = jnp.load('DOUBLE_PENDULUM-h=512-gamma=1.0-shots=512-iter=10-elite=0.1-smooth=0.1-alpha=1.0-dt=0.01-traj.npy')

# Focus on swing-up stage (before timestep 800)
T_swing_up = 600
theta0 = X[:T_swing_up, 0]
theta1 = X[:T_swing_up, 1]

hist = jnp.load('DOUBLE_PENDULUM-h=512-gamma=1.0-shots=512-iter=10-elite=0.1-smooth=0.1-alpha=1.0-dt=0.01-hist.npy')[:T_swing_up]

# Create combined visualization
plot_combined_trajectory_info(
    theta0,
    theta1,
    hist,
    dt=0.01,
    sample=100,
    sampling_strategy="uniform",
    filename="combined_swingup_trajectory_info.png",
)