import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from typing import Callable
from pathlib import Path

import jax
print(jax.devices())
from jax import Array
from jax import numpy as jnp
from einops import einsum
from flax import nnx
import flashbax as fbx
import optax
import orbax.checkpoint as ocp
import matplotlib.pyplot as plt

from val import Dynamics, make_step, make_unroll_policy
from val.pendulum import init_pendulum_state, normalize_pendulum_state#, pendulum_cost
from val.ddpg import SART, DDPG, Policy, build_hidden_layers, soft_update
from val.utils import make_compute_value

class Value(nnx.Module):
    def __init__(
            self, 
            rngs: nnx.Rngs,
            input_dim: int, 
            hidden_dim: int, 
            num_layers:int, 
            activation: Callable,
        ):
        
        layers = []
        ## input layer
        layers.append(nnx.Linear(input_dim, hidden_dim, rngs = rngs))
        layers.append(activation)

        ## hidden layers
        layers.extend(build_hidden_layers(hidden_dim, num_layers, activation, rngs))

        ## output layers
        layers.append(nnx.Linear(hidden_dim, 1, rngs = rngs))

        self.layers = nnx.Sequential(*layers)
    
    def __call__(self, x: Array):
        return self.layers(x)

# def make_update_value(step: Callable, pi: Callable, cost: Callable, normalize: Callable, gamma: float, grad_penalty: float, hess_penalty: float):

#     batch_normalize = jax.jit(jax.vmap(normalize))

#     @nnx.jit
#     def update_value(value: Value, target_value: Value, value_opt: nnx.Optimizer, batch: tuple):

#         x, u, c, t, x_next = batch

#         z = batch_normalize(x)
#         z_next = batch_normalize(x_next)

#         v_ = jax.lax.stop_gradient(target_value(z_next))
#         y = c + (1.0 - t) * gamma * v_

#         def loss_fn(value):
#             v = value(z)
#             return jnp.mean(optax.huber_loss(v, y))
        
#         loss, grad = nnx.value_and_grad(loss_fn)(value)
#         value_opt.update(grad)
#         return loss, 0.0, 0.0
    
#     return update_value

def make_update_value(step: Callable, pi: Callable, cost: Callable, normalize: Callable, gamma: float, grad_penalty: float, hess_penalty: float):


    pi_x = jax.jacfwd(pi)
    pi_xx = jax.jacfwd(pi_x)

    linearize_step = jax.jacfwd(step, argnums = (0, 1))
    quadraticize_step = jax.jacfwd(linearize_step, argnums = (0, 1))

    linearize_cost = jax.jacrev(cost, argnums = (0, 1))
    quadraticize_cost = jax.jacrev(linearize_cost, argnums = (0, 1))

    batch_pi_x = jax.vmap(pi_x)
    batch_pi_xx = jax.vmap(pi_xx)

    batch_linearize_step = jax.vmap(linearize_step)
    batch_quadraticize_step = jax.vmap(quadraticize_step)

    batch_linearize_cost = jax.vmap(linearize_cost)
    batch_quadraticize_cost = jax.vmap(quadraticize_cost)

    def make_V(value: Value):

        graphdef = nnx.graphdef(value)

        def V(x: Array, state: nnx.State):
            return nnx.merge(graphdef, state)(normalize(x)).squeeze()
        
        Vx = jax.jacrev(V)
        Vxx = jax.jacfwd(Vx)

        ## batchwise versions of value and gradient of value
        V = jax.vmap(V, in_axes = (0, None))
        Vx = jax.vmap(Vx, in_axes = (0, None))
        Vxx = jax.vmap(Vxx, in_axes = (0, None))
        return V, Vx, Vxx
    
    @nnx.jit
    def update_value(value: Value, target_value: Value, value_opt: nnx.Optimizer, batch: tuple):

        V, Vx, Vxx = make_V(value)

        x, u, c, t, x_next = batch

        '''
        computing value gradient update
        '''
        # ## compute policy gradient
        K = batch_pi_x(x)
        ## compute dynamics gradients
        fx, fu = batch_linearize_step(x, u)
        ## closed loop derivative
        D = fx + einsum(fu, K, 't x1 u, t u x2 -> t x1 x2')
        ## derivatives of cost
        cx, cu = batch_linearize_cost(x, u)
        cz = cx + einsum(cu, K, 't u, t u x -> t x')

        '''
        computing value hessian update
        '''
        # KK = batch_pi_xx(x)
        # (fxx, fxu), (fux, fuu) = batch_quadraticize_step(x, u)
        # (cxx, cxu), (cux, cuu) = batch_quadraticize_cost(x, u)
        # ## second order closed loop (total) derivative of dynamics
        # H = fxx \
        #     + einsum(fxu, K, 't x x1 u, t u x2 -> t x x1 x2') \
        #     + einsum(K, fux, 't u x1, t x u x2 -> t x x1 x2') \
        #     + einsum(K, fuu, K, 't u1 x1, t x u1 u2, t u2 x2 -> t x x1 x2') \
        #     + einsum(fu, KK, 't x u, t u x1 x2 -> t x x1 x2')
        # ## hessian of instantanious cost
        # czz = cxx \
        #     + einsum(cxu, K, 't x1 u, t u x2 -> t x1 x2') \
        #     + einsum(K, cux, 't u x1, t u x2 -> t x1 x2') \
        #     + einsum(K, cuu, K, 't u1 x1, t u1 u2, t u2 x2 -> t x1 x2') \
        #     + einsum(cu, KK, 't u, t u x1 x2 -> t x1 x2')

        ## compute next value and gradient
        v_next = jax.lax.stop_gradient(V(x_next, nnx.state(target_value)))
        vx_next = jax.lax.stop_gradient(Vx(x_next, nnx.state(target_value)))
        # vxx_next = jax.lax.stop_gradient(Vxx(x_next, nnx.state(target_value)))

        ## compute regression targets
        v_target = c[:, 0] + (1.0 - t[:, 0]) * gamma * v_next
        vx_target = cz + (1.0 - t) * gamma * einsum(D, vx_next, 'b x1 x2, b x1 -> b x2')

        # pullback = einsum(D, vxx_next, D, 'b x1 x3, b x1 x2, b x2 x4 -> b x3 x4')
        # pushforward = einsum(vx_next, H, 'b x, b x x1 x2 -> b x1 x2')
        # vxx_target = czz + (1.0 - t[:, :, None]) * gamma * (pullback + pushforward)

        def loss_fn(value: Value):

            value_state = nnx.state(value)

            v = V(x, value_state)
            # vx = Vx(x, value_state)
            # vxx = Vxx(x, value_state)
            
            v_loss = jnp.mean(optax.huber_loss(v, v_target))
            # vx_loss = jnp.mean(optax.huber_loss(vx, vx_target))
            vx_loss = 0.0
            # vxx_loss = jnp.mean(optax.huber_loss(vxx, vxx_target))
            vxx_loss = 0.0

            loss = v_loss + vx_loss * grad_penalty + vxx_loss * hess_penalty
            
            aux = (v_loss, vx_loss, vxx_loss)
            return loss, aux
        
        (loss, (v_loss, vx_loss, vxx_loss)), grad = nnx.value_and_grad(loss_fn, has_aux = True)(value)

        value_opt.update(grad)

        return v_loss, vx_loss, vxx_loss

    return update_value

def load_ddpg():

    ## hyperparameters of ddpg
    hidden_dim = 128
    num_layers_critic = 4
    num_layers_policy = 1
    activation = nnx.gelu
    critic_lr = 1e-3
    policy_lr = 1e-4
    gamma = 0.99
    # tau = 0.005

    ## load in pretrained ddpg
    manager = ocp.CheckpointManager(
        directory = Path(f'results/checkpoints/hidden_dim={hidden_dim}-num_layers_critic={num_layers_critic}-grad_penalty=0.0-hess_penalty=0.0').resolve()
    )
    ddpg = DDPG(rngs, normalize_pendulum_state, input_dim, ctrl_dim, hidden_dim, num_layers_critic, num_layers_policy, activation, critic_lr, policy_lr, gamma)
    ddpg_state = manager.restore(39900, args = ocp.args.StandardRestore(nnx.state(ddpg)))
    nnx.update(ddpg, ddpg_state)
    return ddpg


@jax.jit
def pendulum_cost(xt: Array, ut: Array):
    theta = xt[0]
    theta_dot = xt[1]
    angle_cost = jnp.cos(theta)
    vel_cost = 0.1 * theta_dot ** 2
    act_cost = 1.0 * ut ** 2
    return (angle_cost + vel_cost + act_cost).squeeze()

if __name__ == '__main__':

    seed = 1
    rngs = nnx.Rngs(seed)
    key = jax.random.PRNGKey(seed)

    ## load in xml
    xml_path = 'xml/pendulum.xml'
    dt = 0.05
    dyn = Dynamics(path = xml_path, dt = dt)
    step = make_step(dyn)
    batch_init_state = jax.vmap(init_pendulum_state)
    input_dim = len(normalize_pendulum_state(init_pendulum_state(key))) ## get the shape of the normalized state
    ctrl_dim = dyn.control_dim


    ## buffer parameters
    iterations = 100
    num_env = 1000
    episode_len = 400
    buffer_len = iterations * num_env * episode_len
    print(f'Buffer Length = {buffer_len}')


    ## hyperparameters of value
    batch_size = 1024
    value_hidden_dim = 1024
    value_num_layers = 5
    value_activation = nnx.gelu
    value_gamma = 0.9
    ## loss weights
    # grad_penalty = 0.0
    # hess_penalty = 0.0005
    grad_penalty = 0.0
    hess_penalty = 0.0
    tau = 0.005

    ## instantiating the buffer
    buffer = fbx.make_flat_buffer(
        max_length = buffer_len,
        min_length = batch_size,
        sample_batch_size = batch_size,
        add_batch_size = num_env
    )

    ## jit compiling buffer functions
    buffer = buffer.replace(
        init = jax.jit(buffer.init),
        add = jax.jit(buffer.add, donate_argnums = 0),
        sample = jax.jit(buffer.sample),
        can_sample = jax.jit(buffer.can_sample),
    )

    ## create dummy transition
    sart = SART(
        s = jnp.zeros(dyn.state_dim), 
        a = jnp.zeros(dyn.control_dim),
        r = jnp.zeros(1),
        t = jnp.zeros(1, dtype = bool)
    )

    ## initialize replay buffer with dummy transition
    buffer_state = buffer.init(sart)

    # === RESTORE THE BUFFER ===
    checkpointer = ocp.CheckpointManager(
        directory = Path('results/buffers/pendulum_ddpg_dataset').resolve()
    )
    buffer_state = checkpointer.restore(0, args = ocp.args.StandardRestore(buffer_state))


    ddpg = load_ddpg()
    policy = ddpg.policy
    ## make a policy
    pi = jax.jit(lambda _x: policy(normalize_pendulum_state(_x)))

    ## instantiate value function
    value = Value(rngs, input_dim, hidden_dim = value_hidden_dim, num_layers = value_num_layers, activation = value_activation)
    target_value = nnx.clone(value)

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(1e-3)
    )

    value_opt = nnx.Optimizer(value, tx)
    ## make the value update function
    update_value = make_update_value(step, pi, pendulum_cost, normalize_pendulum_state, value_gamma, grad_penalty, hess_penalty)

    '''
    testing
    '''
    # key, subkey = jax.random.split(key)
    # # sample a batch from the buffer
    # data = buffer.sample(buffer_state, subkey)
    # # extract experience
    # batch = (data.experience.first.s, data.experience.first.a, data.experience.first.r, data.experience.first.t, data.experience.second.s)
    # v_loss, vx_loss, vxx_loss = update_value(value, target_value, value_opt, batch)
    # print(v_loss, vx_loss, vxx_loss)


    ## build functions
    unroll_policy = make_unroll_policy(step, pi, episode_len)
    compute_value = make_compute_value(pendulum_cost)

    @jax.jit
    def compute_true_value(xt: Array):
        X, U = unroll_policy(xt)
        return compute_value(X, U, value_gamma)
    
    ## evaluate true value over grid points
    num_test_points = 10000
    test_points = batch_init_state(jax.random.split(key, num_test_points))
    normalized_test_points = jax.vmap(normalize_pendulum_state)(test_points)
    true_value = jax.vmap(compute_true_value)(test_points)[:, 0]

    test_every = 500

    v_loss_hist = []
    vx_loss_hist = []
    vxx_loss_hist = []
    test_loss_hist = []

    train_steps = 100000
    path = Path(f'results/offline/train_steps={train_steps}-gamma={value_gamma}-grad_p={grad_penalty}-hess_p={hess_penalty}-hidden_dim={value_hidden_dim}-num_layers={value_num_layers}').resolve()

    ## Create a checkpointer
    options = ocp.CheckpointManagerOptions(
        preservation_policy = ocp.checkpoint_managers.preservation_policy.LatestN(4)
    )
    checkpointer = ocp.CheckpointManager(directory = path, options = options)

    for i in range(train_steps):

        key, subkey = jax.random.split(key)

        ## sample a batch from the buffer
        data = buffer.sample(buffer_state, subkey)
        ## extract experience
        batch = (data.experience.first.s, data.experience.first.a, data.experience.first.r, data.experience.first.t, data.experience.second.s)

        ## update value function
        v_loss, vx_loss, vxx_loss = update_value(value, target_value, value_opt, batch)
        ## update target network
        target_value = soft_update(value, target_value, tau)

        print(f'Iteration = {i}, V Loss = {v_loss}, Vx Loss = {vx_loss}, Vxx Loss = {vxx_loss}')
        v_loss_hist.append(v_loss)
        vx_loss_hist.append(vx_loss)
        vxx_loss_hist.append(vxx_loss)

        if i % test_every == 0:
            pred_value = value(normalized_test_points).squeeze()
            test_loss = jnp.mean(optax.squared_error(pred_value, true_value))
            test_loss_hist.append(test_loss)
            print(f'Test Loss = {test_loss}')
            checkpointer.save(i, args = ocp.args.StandardSave(nnx.state(value)))

            jnp.save(path / 'v_loss_hist.npy', jnp.array(v_loss_hist))
            jnp.save(path / 'vx_loss_hist.npy', jnp.array(vx_loss_hist))
            jnp.save(path / 'test_loss_hist.npy', jnp.array(test_loss_hist))

            fig, ax = plt.subplots(1, 4, figsize = (20, 5))
            ax[0].set_title('Value Loss')
            ax[0].set_xlabel('Train Iteration')
            ax[0].set_ylabel('Value Loss')
            ax[0].plot(v_loss_hist)

            ax[1].set_title('Value Gradient Loss')
            ax[1].set_xlabel('Train Iteration')
            ax[1].set_ylabel('Value Gradient Loss')
            ax[1].plot(vx_loss_hist)

            ax[2].set_title('Value Hessian Loss')
            ax[2].set_xlabel('Train Iteration')
            ax[2].set_ylabel('Value Hessian Loss')
            ax[2].plot(vxx_loss_hist)

            ax[3].set_title('Test Error')
            ax[3].set_xlabel('Test Iteration')
            ax[3].set_ylabel('Log Mean Squared Error')
            ax[3].plot(jnp.log(jnp.array(test_loss_hist)))

            fig.tight_layout()
            fig.savefig(path / 'test_loss_hist.png', dpi = 300)
            plt.close(fig)