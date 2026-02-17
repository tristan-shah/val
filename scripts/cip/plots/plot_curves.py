from jax import numpy as jnp
import matplotlib.pyplot as plt

# dt = 0.01


# name = 'DOUBLE_PENDULUM-gear=6.0-h=512-gamma=1.0-shots=512-iter=10-elite=0.1-smooth=0.1-alpha=1.0-dt=0.01'
# dp_hist = jnp.load(name + '-hist.npy')[:1200]

# name = 'SINGLE_PENDULUM-h=600-gamma=1.0-shots=512-iter=1-elite=0.1-smooth=0.1-alpha=1.0-dt=0.01'
# sp_hist = jnp.load(name + '-hist.npy')[:1200]

# name = 'CART_POLE-h=400-gamma=1.0-shots=512-iter=1-elite=0.1-smooth=0.1-alpha=1.0-dt=0.01'
# cp_hist = jnp.load(name + '-hist.npy')[:1200]


# T = jnp.arange(0.0, 1200) * dt


# fig, ax = plt.subplots(1, 1)
# fig.suptitle('Effectiveness of CIP for Intrinsic Control')
# ax.plot(T, dp_hist, label = 'Double Pendulum')
# ax.plot(T, sp_hist, label = 'Single Pendulum')
# ax.plot(T, cp_hist, label = 'Cart Pole')

# ax.set_xlabel('Time (s)')
# ax.set_ylabel('nats/s')

# leg = ax.legend(title = 'Environment')

# leg.get_title().set_fontweight('bold')
# fig.tight_layout()
# fig.savefig('cip.png', dpi = 300)
# plt.show()




dt = 0.05

T = jnp.arange(0.0, 600) * dt

fig, ax = plt.subplots(1, 1)
fig.suptitle('Testing CIP Surrogate on Single Pendulum With Horizon = 300')

ax.set_xlabel('Time (s)')
ax.set_ylabel('CIP Approximation (nats/s)')

for i in range(1, 3 + 1):
    hist = jnp.load(f'SINGLE_PENDULUM-h=200-shots=512-iter={i}-elite=0.1-smooth=0.1-rho=0.9-dt={dt}-hist.npy')
    ax.plot(T, hist, label = i)

ax.legend(title = 'MPC Iterations')
fig.savefig('surrogate.png', dpi = 300)
plt.show()