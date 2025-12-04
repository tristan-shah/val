from pathlib import Path

import jax
from jax import numpy as jnp
from flax import nnx
import flashbax as fbx
import orbax.checkpoint as ocp
import matplotlib.pyplot as plt

from val import Dynamics, make_step
from val.pendulum import init_pendulum_state, pendulum_reward, pendulum_cost, normalize_pendulum_state
from val.ddpg import DDPG, SART, update_policy, soft_update

from val.ddpg import make_update_critic, update_critic

if __name__ == '__main__':
    seed = 0
    key = jax.random.PRNGKey(seed)
    rngs = nnx.Rngs(seed)

    dt = 0.05
    num_env = 100
    episode_len = 400
    buffer_len = 1_000_000
    num_episodes = 100

    gamma = 0.99
    hidden_dim = 128
    num_layers_critic = 4
    num_layers_policy = 1
    activation = nnx.gelu
    critic_lr = 1e-3
    policy_lr = 1e-4
    batch_size = 128
    tau = 0.005
    exploration_noise = 0.1

    # grad_penalty = 0.0001
    # hess_penalty = 0.0001
    grad_penalty = 0.0
    hess_penalty = 0.0

    ## load in dynamics
    dyn = Dynamics(path = 'xml/pendulum.xml', dt = dt)
    step = make_step(dyn)
    batch_step = jax.vmap(step)
    low = dyn.mjx_model.actuator_ctrlrange[:, 0]
    high = dyn.mjx_model.actuator_ctrlrange[:, 1]
    input_dim = len(normalize_pendulum_state(init_pendulum_state(key))) ## get the shape of the normalized state
    ctrl_dim = dyn.control_dim

    ddpg = DDPG(rngs, normalize_pendulum_state, input_dim, ctrl_dim, hidden_dim, num_layers_critic, num_layers_policy, activation, critic_lr, policy_lr, gamma)

    ## fake batch
    x = jax.random.normal(key, (batch_size, dyn.state_dim))
    u = jax.random.normal(key, (batch_size, dyn.control_dim))
    r = jax.random.normal(key, (batch_size,))
    t = jnp.zeros((batch_size, 1))
    x_ = jax.random.normal(key, (batch_size, dyn.state_dim))
    batch = (x, u, r, t, x_)

    # ## make the critic update function
    # update_critic = make_update_critic(
    #     step, 
    #     lambda _x, _u: jnp.squeeze(pendulum_reward(_x, _u)), 
    #     grad_penalty,
    #     hess_penalty
    # )


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

    critic_loss_hist = []
    critic_grad_loss_hist = []
    critic_hess_loss_hist = []
    policy_loss_hist = []
    reward_hist = []
    step = 0

    path = Path(f'checkpoints/hidden_dim={hidden_dim}-num_layers_critic={num_layers_critic}-grad_penalty={grad_penalty}-hess_penalty={hess_penalty}').resolve()
    path = Path(f'checkpoints/hidden_dim={hidden_dim}-num_layers_critic={num_layers_critic}-grad_penalty={grad_penalty}-hess_penalty={hess_penalty}').resolve()

    ## Create a checkpointer
    options = ocp.CheckpointManagerOptions(
        save_decision_policy = ocp.checkpoint_managers.save_decision_policy.FixedIntervalPolicy(100),
        preservation_policy = ocp.checkpoint_managers.preservation_policy.LatestN(4)
    )

    manager = ocp.CheckpointManager(path, options = options)

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
            u = ddpg.batch_act(x)
            key, subkey = jax.random.split(key)
            u = u + exploration_noise * jax.random.normal(subkey, shape = (num_env, dyn.control_dim))
            u = jnp.clip(u, low, high)

            ## step the dynamics
            x_next = batch_step(x, u)

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

                ## extract experience
                batch = (
                    data.experience.first.s,
                    data.experience.first.a,
                    data.experience.first.r,
                    data.experience.first.t,
                    data.experience.second.s
                )

                # critic_loss, critic_grad_loss, critic_hess_loss = update_critic(ddpg, batch)
                critic_loss = update_critic(ddpg, batch)
                critic_grad_loss, critic_hess_loss = 0.0, 0.0

                policy_loss = update_policy(ddpg, batch)

                ## update targets
                target_critic = soft_update(ddpg.critic, ddpg.target_critic, tau)
                target_policy = soft_update(ddpg.policy, ddpg.target_policy, tau)

                critic_loss_hist.append(critic_loss)
                critic_grad_loss_hist.append(critic_grad_loss)
                critic_hess_loss_hist.append(critic_hess_loss)
                policy_loss_hist.append(policy_loss)

                manager.save(step, 
                    args = ocp.args.StandardSave(nnx.state(ddpg))
                )

                step += 1
        
        reward_hist.append(total_reward)
        print(episode, total_reward.mean(), total_reward.std())

    
    reward_hist = jnp.concatenate(reward_hist, axis = 1)
    manager.wait_until_finished()

    ## compute stats over parallel environments
    reward_mean = reward_hist.mean(axis = 0)
    reward_std = reward_hist.std(axis = 0)
    
    fig, ax = plt.subplots(1, 5, figsize = (15, 5))
    ax[0].set_title('Critic Loss')
    ax[0].plot(critic_loss_hist)
    
    ax[1].set_title('Policy Loss')
    ax[1].plot(policy_loss_hist)

    ax[2].set_title('Critic Grad Loss')
    ax[2].plot(critic_grad_loss_hist)

    ax[3].set_title('Critic Hess Loss')
    ax[3].plot(critic_hess_loss_hist)
    
    ax[4].set_title('Reward History')
    ax[4].set_xlabel('Episode')
    ax[4].plot(reward_mean)
    fig.savefig('stats.png', dpi = 300)