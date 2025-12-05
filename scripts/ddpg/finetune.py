from typing import Callable
from pathlib import Path

import jax
from jax import Array
from jax import numpy as jnp
from einops import einsum
from flax import nnx
import flashbax as fbx
import optax
import orbax.checkpoint as ocp
import matplotlib.pyplot as plt

from val import Dynamics, make_step, make_unroll_policy
from val.pendulum import init_pendulum_state, normalize_pendulum_state, pendulum_cost
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
    
def make_update_value(step: Callable, pi: Callable, cost: Callable, normalize: Callable, gamma: float, grad_penalty: float):

    batch_normalize = jax.jit(jax.vmap(normalize))

    @nnx.jit
    def update_value(value: Value, target_value: Value, value_opt: nnx.Optimizer, batch: tuple):

        x, u, c, t, x_next = batch

        z = batch_normalize(x)
        z_next = batch_normalize(x_next)

        v_ = jax.lax.stop_gradient(target_value(z_next))
        y = c + (1.0 - t) * gamma * v_

        def loss_fn(value):
            v = value(z)
            return jnp.mean(optax.huber_loss(v, y))
        
        loss, grad = nnx.value_and_grad(loss_fn)(value)
        value_opt.update(grad)
        return loss
    
    return update_value

# def make_update_value(step: Callable, pi: Callable, cost: Callable, normalize: Callable, gamma: float, grad_penalty: float):


#     batch_linearize_step = jax.vmap(jax.jacfwd(step, argnums = (0, 1)))
#     batch_linearize_cost = jax.vmap(jax.jacrev(cost, argnums = (0, 1)))
#     batch_pi_x = jax.vmap(jax.jacfwd(pi))

#     def make_V(value: Value):

#         graphdef = nnx.graphdef(value)

#         def V(x: Array, state: nnx.State):
#             return nnx.merge(graphdef, state)(normalize(x)).squeeze()
        
#         Vx = jax.jacrev(V)

#         V = jax.vmap(V, in_axes = (0, None))
#         Vx = jax.vmap(Vx, in_axes = (0, None))
#         return V, Vx
    
#     @nnx.jit
#     def update_value(value: Value, target_value: Value, value_opt: nnx.Optimizer, batch: tuple):

#         V, Vx = make_V(value)

#         x, u, c, t, x_next = batch

#         ## compute next value and gradient
#         v_next = jax.lax.stop_gradient(V(x_next, nnx.state(target_value)))
#         # vx_next = jax.lax.stop_gradient(Vx(x_next, nnx.state(target_value)))

#         # ## compute policy gradient
#         # K = batch_pi_x(x)
#         # ## compute dynamics gradients
#         # fx, fu = batch_linearize_step(x, u)
#         # ## closed loop derivative
#         # D = fx + einsum(fu, K, 't x1 u, t u x2 -> t x1 x2')
#         # ## derivatives of cost
#         # cx, cu = batch_linearize_cost(x, u)
#         # cz = cx + einsum(cu, K, 't u, t u x -> t x')

#         ## compute regression targets
#         v_target = c + (1.0 - t) * gamma * v_next
#         # vx_target = cz + (1.0 - t) * gamma * einsum(D, vx_next, 'b x1 x2, b x1 -> b x2')

#         def loss_fn(value: Value):

#             value_state = nnx.state(value)

#             v = V(x, value_state)
#             # vx = Vx(x, value_state)

#             v_loss = jnp.mean(optax.huber_loss(v, v_target))
#             # vx_loss = jnp.mean(optax.huber_loss(vx, vx_target))
#             vx_loss = 0.0

#             loss = v_loss #+ vx_loss * grad_penalty
            
#             aux = (v_loss, vx_loss)
#             return loss, aux
        
#         (loss, (v_loss, vx_loss)), grad = nnx.value_and_grad(loss_fn, has_aux = True)(value)

#         value_opt.update(grad)

#         return v_loss, vx_loss

#     return update_value

if __name__ == '__main__':

    seed = 0
    rngs = nnx.Rngs(seed)
    key = jax.random.PRNGKey(seed)

    episode_len = 400
    dt = 0.05
    ## load in xml
    xml_path = 'xml/pendulum.xml'
    dyn = Dynamics(path = xml_path, dt = dt)
    step = make_step(dyn)
    batch_init_state = jax.vmap(init_pendulum_state)
    input_dim = len(normalize_pendulum_state(init_pendulum_state(key))) ## get the shape of the normalized state
    ctrl_dim = dyn.control_dim
    
    grad_penalty = 0.0 #1e-12

    ## hyperparameters of ddpg
    hidden_dim = 128
    num_layers_critic = 4
    num_layers_policy = 1
    activation = nnx.gelu
    critic_lr = 1e-3
    policy_lr = 1e-4
    gamma = 0.99
    tau = 0.005

    ## buffer parameters
    iterations = 100
    num_env = 1000
    episode_len = 400
    buffer_len = iterations * num_env * episode_len
    print(f'Buffer Length = {buffer_len}')

    ## can change this sampling batch size
    batch_size = 1024

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
        directory = Path('buffers/pendulum_ddpg_dataset').resolve()
    )
    buffer_state = checkpointer.restore(0, args = ocp.args.StandardRestore(buffer_state))

    ## load in pretrained ddpg
    manager = ocp.CheckpointManager(
        directory = Path(f'checkpoints/hidden_dim={hidden_dim}-num_layers_critic={num_layers_critic}-grad_penalty=0.0-hess_penalty=0.0').resolve()
    )
    ddpg = DDPG(rngs, normalize_pendulum_state, input_dim, ctrl_dim, hidden_dim, num_layers_critic, num_layers_policy, activation, critic_lr, policy_lr, gamma)
    ddpg_state = manager.restore(39900, args = ocp.args.StandardRestore(nnx.state(ddpg)))
    nnx.update(ddpg, ddpg_state)
    policy = ddpg.policy
    ## make a policy
    pi = jax.jit(lambda _x: policy(normalize_pendulum_state(_x)))

    ## instantiate value function
    value_hidden_dim = 256
    value_num_layers = 10
    value_activation = nnx.gelu
    value = Value(rngs, input_dim, hidden_dim = value_hidden_dim, num_layers = value_num_layers, activation = value_activation)
    target_value = nnx.clone(value)
    value_opt = nnx.Optimizer(value, optax.adam(1e-3))
    ## make the value update function
    # update_value = make_update_value(normalize_pendulum_state, gamma)
    update_value = make_update_value(step, pi, pendulum_cost, normalize_pendulum_state, gamma, grad_penalty)

    num_test_points = 10000
    ## initialize a random batch of states
    test_points = batch_init_state(jax.random.split(key, num_test_points))
    normalized_test_points = jax.vmap(normalize_pendulum_state)(test_points)

    ## build functions
    unroll_policy = make_unroll_policy(step, pi, episode_len)
    compute_value = make_compute_value(pendulum_cost)

    @jax.jit
    def compute_true_value(xt: Array):
        X, U = unroll_policy(xt)
        return compute_value(X, U, gamma)
    
    ## evaluate true value over grid points
    true_value = jax.vmap(compute_true_value)(test_points)[:, 0]

    test_every = 500

    v_loss_hist = []
    vx_loss_hist = []
    test_loss_hist = []

    path = Path(f'offline/value_grad-grad_penalty={grad_penalty}-value_hidden_dim={value_hidden_dim}-value_num_layers={value_num_layers}').resolve()

    ## Create a checkpointer
    options = ocp.CheckpointManagerOptions(
        preservation_policy = ocp.checkpoint_managers.preservation_policy.LatestN(4)
    )
    checkpointer = ocp.CheckpointManager(directory = path, options = options)

    '''
    testing
    '''
    # key, subkey = jax.random.split(key)
    ## sample a batch from the buffer
    # data = buffer.sample(buffer_state, subkey)
    ## extract experience
    # batch = (data.experience.first.s, data.experience.first.a, data.experience.first.r, data.experience.first.t, data.experience.second.s)
    # v_loss, vx_loss = update_value(value, target_value, value_opt, batch)


    for i in range(50000):

        key, subkey = jax.random.split(key)

        ## sample a batch from the buffer
        data = buffer.sample(buffer_state, subkey)
        ## extract experience
        batch = (data.experience.first.s, data.experience.first.a, data.experience.first.r, data.experience.first.t, data.experience.second.s)

        ## update value function
        v_loss = update_value(value, target_value, value_opt, batch)
        vx_loss = 0.0
        # v_loss, vx_loss = update_value(value, target_value, value_opt, batch)
        ## update target network
        target_value = soft_update(value, target_value, tau)

        print(f'Iteration = {i}, V Loss = {v_loss}, Vx Loss = {vx_loss}')
        v_loss_hist.append(v_loss)
        vx_loss_hist.append(vx_loss)

        if i % test_every == 0:
            pred_value = value(normalized_test_points).squeeze()
            test_loss = jnp.mean(optax.squared_error(pred_value, true_value))
            test_loss_hist.append(test_loss)
            print(f'Test Loss = {test_loss}')
            checkpointer.save(i, args = ocp.args.StandardSave(nnx.state(value)))

            jnp.save(path / 'v_loss_hist.npy', jnp.array(v_loss_hist))
            jnp.save(path / 'vx_loss_hist.npy', jnp.array(vx_loss_hist))
            jnp.save(path / 'test_loss_hist.npy', jnp.array(test_loss_hist))

            fig, ax = plt.subplots(1, 3, figsize = (15, 5))
            ax[0].set_title('Value Loss')
            ax[0].set_xlabel('Train Iteration')
            ax[0].set_ylabel('Value Loss')
            ax[0].plot(v_loss_hist)

            ax[1].set_title('Value Gradient Loss')
            ax[1].set_xlabel('Train Iteration')
            ax[1].set_ylabel('Value Gradient Loss')
            ax[1].plot(vx_loss_hist)

            ax[2].set_title('Test Error')
            ax[2].set_xlabel('Test Iteration')
            ax[2].set_ylabel('Log Mean Squared Error')
            ax[2].plot(jnp.log(jnp.array(test_loss_hist)))

            fig.tight_layout()
            fig.savefig(path / 'test_loss_hist.png', dpi = 300)
            plt.close(fig)