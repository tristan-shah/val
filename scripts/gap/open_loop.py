import jax
from jax import Array
from jax import numpy as jnp
from einops import einsum
import matplotlib.pyplot as plt

from val import Dynamics, make_step, make_unroll

@jax.jit
def compute_vol(fx: Array, fu: Array):
    
    dx = fx.shape[-1]
    du = fu.shape[-1]

    Q = jnp.eye(dx)
    R = jnp.eye(du) * 1.0

    def scan_fn(carry: tuple[Array, Array], inputs: tuple[Array, Array]):
        
        Y_t, V_t = carry
        fx_t, fu_t = inputs

        Y_t = Q + fx_t.T @ Y_t @ fx_t
        V_t = Q + fx_t.T @ V_t @ fx_t - fx_t.T @ V_t @ fu_t @ jnp.linalg.inv(R + fu_t.T @ V_t @ fu_t) @ fu_t.T @ V_t @ fx_t

        carry = (Y_t, V_t)

        return carry, carry
    
    _, (Y, V) = jax.lax.scan(scan_fn, init = (Q, Q), xs = (fx, fu), reverse = True)
    return Y, V

def make_compute_entropy(step: callable):
    
    traj_linerize = jax.jit(jax.vmap(jax.jacfwd(step, argnums = (0, 1))))
    unroll = make_unroll(step)

    def compute_entropy(x0: Array, U: Array):

        X = unroll(x0, U)
        fx, fu = traj_linerize(X[:-1], U)
        Y, V = compute_vol(fx, fu)

        ol_entropy = jnp.linalg.slogdet(Y).logabsdet
        cl_entropy = jnp.linalg.slogdet(V).logabsdet

        return ol_entropy, cl_entropy
    
    return jax.jit(compute_entropy)

if __name__ == '__main__':
    seed = 0
    key = jax.random.PRNGKey(seed)
    dt = 0.05
    horizon = 500
    shots = 128
    eps = 0.05

    dyn = Dynamics('xml/pendulum.xml', dt = dt)
    # dyn = Dynamics('xml/double_pendulum.xml', dt = dt)

    low = dyn.mjx_model.actuator_ctrlrange[:, 0]
    high = dyn.mjx_model.actuator_ctrlrange[:, 1]

    step = make_step(dyn)
    unroll = make_unroll(step)
    compute_entropy = make_compute_entropy(step)
    batch_compute_entropy = jax.jit(jax.vmap(compute_entropy, in_axes = (None, 0)))

    theta = 0.0
    x0 = jnp.array([theta, 0.0])

    T = jnp.arange(0, horizon)
    t = T * dt
    # time_average = 1 / (jnp.flip(t) + dt)

    U_bar = jnp.zeros((horizon, dyn.control_dim))

    hist = []
    for i in range(300):
        key, subkey = jax.random.split(key)
        U_batch = U_bar[None, :, :] + (jax.random.normal(subkey, (shots, horizon, dyn.control_dim)) * eps)
        U_batch = U_batch.clip(low, high)

        ol_entropy, cl_entropy = batch_compute_entropy(x0, U_batch)

        gap = ol_entropy - cl_entropy
        ## average over time
        total_gap = jnp.mean(gap, axis = 1)
        # total_gap = jnp.mean(ol_entropy, axis = 1)

        idx = jnp.argmax(total_gap)
        U_bar = U_batch[idx]

        print(i, total_gap.mean(), total_gap[idx])
        hist.append(total_gap.mean())

    fig, ax = plt.subplots(2, 1, figsize = (10, 5))
    fig.suptitle('Open Loop Predictive Sampling')

    ax[0].set_xlabel('Horizon Time (s)')
    ax[0].set_ylabel('Control')

    ax[0].set_ylim(low.item() * 1.2, high.item() * 1.2)

    for k in range(shots):
        ax[0].plot(t, U_batch[k])

    ax[1].set_xlabel('PS Iteration')
    ax[1].set_ylabel('Average Chaos Eating?')
    # ax[1].set_ylabel('Open Loop Entropy')
    ax[1].plot(hist)
    fig.tight_layout()
    fig.savefig(f'ent_only_h={horizon}_open_loop.png', dpi = 300)
    plt.show()

    X = unroll(x0, U_bar)
    dyn.render(X, path = f'ent_only_h={horizon}_open_loop.mp4', skip = 1)


    # print(ol_entropy)

    # fig, ax = plt.subplots(1, 1)
    # for i in range(shots):
    #     ax.plot(ol_entropy[i] - cl_entropy[i], alpha = 0.2)
    # plt.show()







    # ol_entropy, cl_entropy = compute_entropy(x0, U)
    

    # ol_entropy_rate = time_average * ol_entropy
    # cl_entropy_rate = time_average * cl_entropy

    # fig, ax = plt.subplots(3, 1, figsize = (9, 7))
    # fig.suptitle(f'Initial Angle = {theta}')
    
    # ax[0].plot(t, ol_entropy, label = 'Open Loop')
    # ax[0].plot(t, cl_entropy, label = 'Closed Loop')
    # ax[0].set_title('Total Entropy (Not Averaged)')
    # ax[0].set_ylabel('Entropy (nats)')
    # ax[0].legend()

    # ax[1].set_title('Averaged Entropy (Finite Time KSE)')
    # ax[1].plot(t, ol_entropy_rate, label = 'Open Loop')
    # ax[1].plot(t, cl_entropy_rate, label = 'Closed Loop')
    # ax[1].set_ylabel('Entropy (nats)')
    # ax[1].legend()

    # ax[2].set_title('Total Difference In Entropy')
    # ax[2].plot(t, ol_entropy - cl_entropy)
    # ax[2].set_xlabel('Time (s)')
    # ax[2].set_ylabel('Entropy (nats)')

    # fig.tight_layout()
    # fig.savefig(f'theta={theta}.png', dpi = 300)
    # plt.show()