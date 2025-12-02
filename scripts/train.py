from pathlib import Path
from copy import deepcopy

import jax
from jax import Array
from jax import numpy as jnp
from flax import nnx
import flashbax as fbx
from flashbax.buffers.flat_buffer import TransitionSample
import chex
import optax
import orbax.checkpoint as ocp

from val import Dynamics, make_step
from val.pendulum import init_pendulum_state, normalize_pendulum_state, pendulum_reward

def build_pendulum_critic(state_dim: int, act_dim:int, h_dim: int, rngs: nnx.Rngs):

    critic = nnx.Sequential(
        nnx.Linear(state_dim + act_dim, h_dim, rngs = rngs),
        nnx.gelu,
        nnx.Linear(h_dim, h_dim, rngs = rngs),
        nnx.gelu,
        nnx.Linear(h_dim, h_dim, rngs = rngs),
        nnx.gelu,
        nnx.Linear(h_dim, h_dim, rngs = rngs),
        nnx.gelu,
        nnx.Linear(h_dim, 1, rngs = rngs)
    )

    return critic

def build_pendulum_policy(state_dim: int, act_dim:int, h_dim: int, rngs: nnx.Rngs):

    policy = nnx.Sequential(
        nnx.Linear(state_dim, h_dim, rngs = rngs),
        nnx.gelu,
        nnx.Linear(h_dim, h_dim, rngs = rngs),
        nnx.gelu,
        nnx.Linear(h_dim, h_dim, rngs = rngs),
        nnx.gelu,
        nnx.Linear(h_dim, act_dim, rngs = rngs),
        nnx.tanh
    )

    return policy

@nnx.jit
def soft_update(model, target_model, tau=0.005):
    new_state = jax.tree_util.tree_map(
        lambda p, tp: tau * p + (1 - tau) * tp,
        nnx.state(model),
        nnx.state(target_model)
    )

    nnx.update(target_model, new_state)
    return target_model

@chex.dataclass(frozen = True)
class SART:
    s: chex.Array
    a: chex.Array
    r: chex.Array
    t: chex.Array

@nnx.jit
def update_critic(critic, target_critic, target_policy, critic_opt, experience: TransitionSample):

    ## extract experience
    x = experience.first.s
    u = experience.first.a
    r = experience.first.r
    t = experience.first.t
    x_ = experience.second.s

    def loss_fn(critic):

        ## normalize states
        z = jax.vmap(normalize_pendulum_state)(x)
        z_ = jax.vmap(normalize_pendulum_state)(x_)
        ## compute action in next state
        u_ = target_policy(z_)

        ## compute values
        q = critic(jnp.concatenate([z, u], axis = -1))
        q_ = target_critic(jnp.concatenate([z_, u_], axis = -1))

        ## compute Bellman TD target
        y = r + (1.0 - t) * gamma * q_
        return jnp.mean(optax.huber_loss(q, y))
    
    loss, grads = nnx.value_and_grad(loss_fn)(critic)
    critic_opt.update(grads)

    return loss

@nnx.jit
def update_policy(critic, policy, policy_opt, experience: TransitionSample):

    ## extract experience
    x = experience.first.s

    def loss_fn(policy):
        ## normalize states
        z = jax.vmap(normalize_pendulum_state)(x)
        q = critic(jnp.concatenate([z, policy(z)], axis = -1))

        return - jnp.mean(q)
    
    loss, grads = nnx.value_and_grad(loss_fn)(policy)
    policy_opt.update(grads)

    return loss

def make_value(policy: callable, critic: callable, normalize: callable):

    def value(_x: Array):
        z = normalize(_x)
        return -critic(jnp.concatenate([z, policy(z)], axis = -1)).squeeze()
    
    return jax.jit(value)

if __name__ == '__main__':
    ## hyperparameters
    seed = 105
    key = jax.random.PRNGKey(seed)
    rngs = nnx.Rngs(seed)
    buffer_len = 1_000_000
    num_episodes = 100
    episode_len = 400
    gamma = 0.99
    batch_size = 128
    exploration_noise = 0.1
    tau = 0.005
    h_dim = 128
    critic_lr = 1e-3
    policy_lr = 1e-4
    dt = 0.05
    num_env = 100

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

    ## load in xml
    xml_path = 'xml/pendulum.xml'
    dyn = Dynamics(path = xml_path, dt = dt)

    step = make_step(dyn)

    low = dyn.mjx_model.actuator_ctrlrange[:, 0]
    high = dyn.mjx_model.actuator_ctrlrange[:, 1]
    
    ## build networks
    critic = build_pendulum_critic(3, dyn.control_dim, h_dim, rngs)
    policy = build_pendulum_policy(3, dyn.control_dim, h_dim, rngs)
    
    target_critic = deepcopy(critic)
    target_policy = deepcopy(policy)
    pi = lambda _x: policy(normalize_pendulum_state(_x))

    # path = Path(f'checkpoints/num_env={num_env}_vec_ddpg').resolve()
    path = Path(f'checkpoints/TEST').resolve()

    ## Create a checkpointer
    options = ocp.CheckpointManagerOptions(
        save_decision_policy = ocp.checkpoint_managers.save_decision_policy.FixedIntervalPolicy(100),
        preservation_policy = ocp.checkpoint_managers.preservation_policy.LatestN(4)
    )

    manager = ocp.CheckpointManager(path, options = options)

    critic_opt = nnx.Optimizer(critic, optax.adam(critic_lr))
    policy_opt = nnx.Optimizer(policy, optax.adam(policy_lr))

    ## create dummy transition
    sart = SART(
        s = jnp.zeros(dyn.state_dim), 
        a = jnp.zeros(dyn.control_dim),
        r = jnp.zeros(1),
        t = jnp.zeros(1, dtype = bool)
    )

    ## initialize replay buffer with dummy transition
    buffer_state = buffer.init(sart)

    critic_loss_hist = []
    policy_loss_hist = []
    reward_hist = []
    step = 0

    for episode in range(num_episodes):
        key, subkey = jax.random.split(key)

        ## initialize a random batch of states
        batch_key = jax.random.split(key, num_env)
        x = jax.vmap(init_pendulum_state)(batch_key)

        done = jnp.zeros((num_env, 1), dtype = jnp.bool)
        total_reward = jnp.zeros((num_env, 1))

        ## run an episode
        for i in range(episode_len):
            key, subkey = jax.random.split(key)

            ## select a batch of random actions
            u = jax.vmap(pi)(x)
            key, subkey = jax.random.split(key)
            u = u + exploration_noise * jax.random.normal(subkey, shape = (num_env, dyn.control_dim))
            u = jnp.clip(u, low, high)

            ## step the dynamics
            x_next = jax.vmap(dyn.step)(x, u)

            ## compute the reward
            r = jax.vmap(pendulum_reward)(x, u)

            ## accumulate total reward
            total_reward += r

            ## check if episode is done
            done = done.at[:].set(jnp.bool(i == episode_len - 1))

            ## store transition
            sart = SART(s = x,  a = u, r = r, t = done)
            buffer_state = buffer.add(buffer_state, sart)

            ## overwrite previous state
            x = x_next

            if buffer.can_sample(buffer_state):

                data = buffer.sample(buffer_state, subkey)

                critic_loss = update_critic(critic, target_critic, target_policy, critic_opt, data.experience)
                policy_loss = update_policy(critic, policy, policy_opt, data.experience)

                target_critic = soft_update(critic, target_critic, tau)
                target_policy = soft_update(policy, target_policy, tau)

                critic_loss_hist.append(critic_loss)
                policy_loss_hist.append(policy_loss)

                manager.save(step, 
                    args = ocp.args.Composite(
                        critic_state = ocp.args.StandardSave(nnx.state(critic)),
                        policy_state = ocp.args.StandardSave(nnx.state(policy))
                    )
                )
                step += 1

        reward_hist.append(total_reward)
        print(episode, total_reward.mean(), total_reward.std())


    reward_hist = jnp.concatenate(reward_hist, axis = 1)
    print(reward_hist.shape)

    manager.wait_until_finished()


    reward_mean = reward_hist.mean(axis = 0)
    reward_std = reward_hist.std(axis = 0)
    

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize = (15, 5))
    ax[0].set_title('Critic Loss')
    ax[0].plot(critic_loss_hist)
    
    ax[1].set_title('Policy Loss')
    ax[1].plot(policy_loss_hist)
    
    ax[2].set_title('Reward History')
    ax[2].plot(reward_mean)
    fig.savefig('stats.png', dpi = 300)