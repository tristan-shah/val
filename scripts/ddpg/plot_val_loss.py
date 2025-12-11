from jax import numpy as jnp
import matplotlib.pyplot as plt

if __name__ == '__main__':


    without_penalty_test_loss_hist = jnp.load('offline/train_steps=100000-grad_penalty=0.0-hess_penalty=0.0-value_hidden_dim=128-value_num_layers=5/test_loss_hist.npy')
    grad_penalty_test_loss_hist = jnp.load('offline/train_steps=100000-grad_penalty=0.05-hess_penalty=0.0-value_hidden_dim=128-value_num_layers=5/test_loss_hist.npy')
    large_hess_penalty_test_loss_hist = jnp.load('offline/train_steps=100000-grad_penalty=0.0-hess_penalty=0.05-value_hidden_dim=128-value_num_layers=5/test_loss_hist.npy')
    small_hess_penalty_test_loss_hist = jnp.load('offline/train_steps=100000-grad_penalty=0.0-hess_penalty=5e-05-value_hidden_dim=128-value_num_layers=5/test_loss_hist.npy')

    # without_penalty_test_loss_hist = jnp.load('old_offline/train_steps=100000-value_grad-grad_penalty=0.0-value_hidden_dim=256-value_num_layers=10/test_loss_hist.npy')
    # grad_penalty_test_loss_hist = jnp.load('old_offline/train_steps=100000-value_grad-grad_penalty=0.05-value_hidden_dim=256-value_num_layers=10/test_loss_hist.npy')

    fig, ax = plt.subplots(1, 1)
    ax.set_title('Error From True Value Function')
    ax.set_xlabel('Test Iteration')
    ax.set_ylabel('Log MSE Error')

    ax.plot(jnp.log(without_penalty_test_loss_hist), label = 'Vanilla')
    ax.plot(jnp.log(grad_penalty_test_loss_hist), label = 'Gradient Penalty = 0.05')
    ax.plot(jnp.log(large_hess_penalty_test_loss_hist), label = 'Hessian Penalty = 0.05')
    ax.plot(jnp.log(small_hess_penalty_test_loss_hist), label = 'Hessian Penalty = 5e-05')
    ax.legend()
    fig.tight_layout()
    fig.savefig('gradient_penalty.png', dpi = 300)
    plt.show()