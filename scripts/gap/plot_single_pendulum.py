import numpy as np
import matplotlib.pyplot as plt
from jax import numpy as jnp

# ----------------------------
# Single Pendulum Positions
# ----------------------------
def single_pendulum_positions(theta, l=1.0):
    x = l * np.sin(theta)
    y = -l * np.cos(theta)
    return x, y

# ----------------------------
# Condensed trajectory plot
# ----------------------------
def plot_single_pendulum_time_condensed(
    theta,
    timespan=None,          # Pass actual time array
    sample=20,
    l=1.0,
    total_width=None,       # Optional: if None, use real timespan
    link_color='#c49a6c',
    joint_color='#8b5a2b',
    ground_color='#7fbf7f',
    sampling_strategy='uniform',  # 'uniform', 'sqrt', 'log'
    filename=None
):
    x, y = single_pendulum_positions(theta, l)

    # Determine total width in x-axis
    if total_width is None:
        total_width = timespan[-1] if timespan is not None else 10.0

    # Choose sampling strategy
    if sampling_strategy == 'sqrt':
        t_normalized = np.linspace(0, 1, sample) ** 2
        idx = (t_normalized * (len(theta) - 1)).astype(int)
    elif sampling_strategy == 'log':
        t_normalized = np.logspace(0, 1, sample) - 1
        t_normalized = t_normalized / t_normalized[-1]
        idx = (t_normalized * (len(theta) - 1)).astype(int)
    else:
        idx = np.linspace(0, len(theta) - 1, sample).astype(int)

    fig, ax = plt.subplots(figsize=(6.8, 2.2))

    all_y = [0]
    all_y.extend(y[idx])
    ground_y = min(all_y) - 0.3

    for k, i in enumerate(idx):
        # Map sampled index to horizontal position proportional to real time
        if timespan is not None:
            x_offset = (timespan[i] / timespan[idx[-1]]) * total_width
        else:
            x_offset = (k / (sample - 1)) * total_width

        # Pendulum link
        ax.plot([x_offset, x_offset + x[i]],
                [0, y[i]],
                lw=1.5,
                color=link_color,
                alpha=0.9)

        # Joint
        ax.scatter([x_offset, x_offset + x[i]],
                   [0, y[i]],
                   s=10,
                   color=joint_color,
                   zorder=3)

    # Ground line
    ax.plot([-0.3, total_width + 0.3],
            [ground_y, ground_y],
            color=ground_color,
            lw=4)

    ax.set_aspect('equal')
    ax.yaxis.set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if timespan is not None:
        n_ticks = 11
        tick_positions = np.linspace(0, total_width, n_ticks)
        # tick_labels = [f"{t:.2f}" for t in np.linspace(0, timespan[idx[-1]], n_ticks)]
        tick_labels = [str(i) for i in range(0, n_ticks)]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_xlim(-0.3, total_width + 0.3)
        ax.set_ylim(ground_y - 0.2, l + 0.5)
        ax.set_xlabel('Time (s)')

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


# ----------------------------
# Example usage
# ----------------------------
name = 'SINGLE_PENDULUM-h=600-gamma=1.0-shots=512-iter=1-elite=0.1-smooth=0.1-alpha=1.0-dt=0.01'

# Load trajectory data
X = jnp.load(name + '-traj.npy')
T_swing_up = 1000  # <-- can change freely
dt = 0.01
t = jnp.arange(T_swing_up) * dt
theta = X[:T_swing_up, 0]  # single angle

hist = jnp.load(name + '-hist.npy')[:T_swing_up]

# Plot trajectory synced to real time
plot_single_pendulum_time_condensed(
    theta,
    timespan=t,
    sample=100,
    total_width=None,  # will automatically match real time
    sampling_strategy='uniform',
    filename='single_pendulum_traj.png'
)

# Plot bits/s over time (automatically matches trajectory)
fig, ax = plt.subplots(1, 1, figsize=(6, 3))
ax.set_xlabel('Time (s)')
ax.set_ylabel('nats/s')
ax.plot(t, hist)
fig.tight_layout()
fig.savefig('single_pendulum_hist.png', dpi=300)
plt.show()
