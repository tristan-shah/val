# Condensed, publication-safe version:
# Time is mapped to a fixed horizontal span (single-column friendly)

import numpy as np
import matplotlib.pyplot as plt
from jax import numpy as jnp

def double_pendulum_positions(theta0, theta1, l0=1.0, l1=1.0):
    x0 = l0 * np.sin(theta0)
    y0 = -l0 * np.cos(theta0)
    x1 = x0 + l1 * np.sin(theta0 + theta1)
    y1 = y0 - l1 * np.cos(theta0 + theta1)
    return x0, y0, x1, y1

def plot_trajectory_time_condensed(
    theta0,
    theta1,
    sample=20,
    l0=1.0,
    l1=1.0,
    total_width=10.0,
    link_color="#c49a6c",
    joint_color="#8b5a2b",
    ground_color="#7fbf7f",
    sampling_strategy="uniform",  # "uniform", "sqrt", or "log"
    filename=None
):
    """
    DIAYN-style condensed trajectory:
    time mapped to a fixed horizontal span.
    
    sampling_strategy:
        - "uniform": evenly spaced samples
        - "sqrt": denser at beginning (square root spacing)
        - "log": very dense at beginning (logarithmic spacing)
    """
    x0, y0, x1, y1 = double_pendulum_positions(theta0, theta1, l0, l1)
    
    # Choose sampling strategy
    if sampling_strategy == "sqrt":
        # Square root spacing - denser at beginning
        t_normalized = np.linspace(0, 1, sample) ** 2
        idx = (t_normalized * (len(theta0) - 1)).astype(int)
    elif sampling_strategy == "log":
        # Logarithmic spacing - very dense at beginning
        t_normalized = np.logspace(0, 1, sample) - 1
        t_normalized = t_normalized / t_normalized[-1]
        idx = (t_normalized * (len(theta0) - 1)).astype(int)
    else:
        # Uniform spacing (original)
        idx = np.linspace(0, len(theta0) - 1, sample).astype(int)

    fig, ax = plt.subplots(figsize=(6.8, 2.2))

    # Collect all y positions to determine appropriate ground level
    all_y = []
    for i in idx:
        all_y.extend([0, y0[i], y1[i]])
    
    ground_y = min(all_y) - 0.3

    for k, i in enumerate(idx):
        x_offset = (k / (sample - 1)) * total_width

        # first link
        ax.plot(
            [x_offset, x_offset + x0[i]],
            [0, y0[i]],
            lw=1.5,
            color=link_color,
            alpha=0.9,
        )

        # second link
        ax.plot(
            [x_offset + x0[i], x_offset + x1[i]],
            [y0[i], y1[i]],
            lw=1.5,
            color=link_color,
            alpha=0.9,
        )

        # joints
        ax.scatter(
            [x_offset, x_offset + x0[i], x_offset + x1[i]],
            [0, y0[i], y1[i]],
            s=10,
            color=joint_color,
            zorder=3,
        )

    # ground line
    ax.plot(
        [-0.3, total_width + 0.3],
        [ground_y, ground_y],
        color=ground_color,
        lw=4,
    )

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-0.3, total_width + 0.3)
    ax.set_ylim(ground_y - 0.2, 2.2)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()


# ---- Load and process data ----
X = jnp.load('DOUBLE_PENDULUM-h=1024-gamma=1.0-shots=512-iter=10-elite=0.1-smooth=0.1-alpha=1.0-dt=0.01-traj.npy')

# Focus on swing-up stage (before timestep 800)
T_swing_up = 800
theta0 = X[:T_swing_up, 0]
theta1 = X[:T_swing_up, 1]

# Visualize the trajectory with denser sampling at the beginning
plot_trajectory_time_condensed(
    theta0,
    theta1,
    sample=100,
    total_width=15,#6.5,
    sampling_strategy="uniform",  # Try "sqrt" or "log" for denser early sampling
    filename="swingup_trajectory.png",
)

# # Condensed, publication-safe version:
# # Time is mapped to a fixed horizontal span (single-column friendly)

# import numpy as np
# import matplotlib.pyplot as plt
# from jax import numpy as jnp

# def double_pendulum_positions(theta0, theta1, l0=1.0, l1=1.0):
#     x0 = l0 * np.sin(theta0)
#     y0 = -l0 * np.cos(theta0)
#     x1 = x0 + l1 * np.sin(theta0 + theta1)
#     y1 = y0 - l1 * np.cos(theta0 + theta1)
#     return x0, y0, x1, y1

# def plot_trajectory_time_condensed(
#     theta0,
#     theta1,
#     sample=20,
#     l0=1.0,
#     l1=1.0,
#     total_width=10.0,
#     link_color="#c49a6c",
#     joint_color="#8b5a2b",
#     ground_color="#7fbf7f",
#     sampling_strategy="uniform",  # "uniform", "sqrt", or "log"
#     filename=None
# ):
#     """
#     DIAYN-style condensed trajectory:
#     time mapped to a fixed horizontal span.
    
#     sampling_strategy:
#         - "uniform": evenly spaced samples
#         - "sqrt": denser at beginning (square root spacing)
#         - "log": very dense at beginning (logarithmic spacing)
#     """
#     x0, y0, x1, y1 = double_pendulum_positions(theta0, theta1, l0, l1)
    
#     # Choose sampling strategy
#     if sampling_strategy == "sqrt":
#         # Square root spacing - denser at beginning
#         t_normalized = np.linspace(0, 1, sample) ** 2
#         idx = (t_normalized * (len(theta0) - 1)).astype(int)
#     elif sampling_strategy == "log":
#         # Logarithmic spacing - very dense at beginning
#         t_normalized = np.logspace(0, 1, sample) - 1
#         t_normalized = t_normalized / t_normalized[-1]
#         idx = (t_normalized * (len(theta0) - 1)).astype(int)
#     else:
#         # Uniform spacing (original)
#         idx = np.linspace(0, len(theta0) - 1, sample).astype(int)

#     fig, ax = plt.subplots(figsize=(6.8, 2.2))

#     # Collect all y positions to determine appropriate ground level
#     all_y = []
#     for i in idx:
#         all_y.extend([0, y0[i], y1[i]])
    
#     ground_y = min(all_y) - 0.3

#     for k, i in enumerate(idx):
#         x_offset = (k / (sample - 1)) * total_width

#         # first link
#         ax.plot(
#             [x_offset, x_offset + x0[i]],
#             [0, y0[i]],
#             lw=3,
#             color=link_color,
#             alpha=0.9,
#         )

#         # second link
#         ax.plot(
#             [x_offset + x0[i], x_offset + x1[i]],
#             [y0[i], y1[i]],
#             lw=3,
#             color=link_color,
#             alpha=0.9,
#         )

#         # joints
#         ax.scatter(
#             [x_offset, x_offset + x0[i], x_offset + x1[i]],
#             [0, y0[i], y1[i]],
#             s=10,
#             color=joint_color,
#             zorder=3,
#         )

#     # ground line
#     ax.plot(
#         [-0.3, total_width + 0.3],
#         [ground_y, ground_y],
#         color=ground_color,
#         lw=4,
#     )

#     ax.set_aspect("equal")
#     ax.axis("off")
#     ax.set_xlim(-0.3, total_width + 0.3)
#     ax.set_ylim(ground_y - 0.2, 2.2)

#     plt.tight_layout()
#     if filename:
#         plt.savefig(filename, dpi=300, bbox_inches="tight")
#     # plt.show()


# # ---- Load and process data ----
# X = jnp.load('DOUBLE_PENDULUM-h=1024-gamma=1.0-shots=512-iter=10-elite=0.1-smooth=0.1-alpha=1.0-dt=0.01-traj.npy')

# # Focus on swing-up stage (before timestep 800)
# T_swing_up = 800
# theta0 = X[:T_swing_up, 0]
# theta1 = X[:T_swing_up, 1]

# # Visualize the trajectory
# plot_trajectory_time_condensed(
#     theta0,
#     theta1,
#     sample=100,
#     total_width=6.5,
#     filename="swingup_trajectory.png",
# )