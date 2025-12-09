from jax import numpy as jnp
import matplotlib.pyplot as plt

if __name__ == '__main__':


    without_penalty_test_loss_hist = jnp.load('offline/train_steps=100000-grad_penalty=0.0-hess_penalty=0.0-value_hidden_dim=128-value_num_layers=5/test_loss_hist.npy')
    grad_penalty_test_loss_hist = jnp.load('offline/train_steps=100000-grad_penalty=0.05-hess_penalty=0.0-value_hidden_dim=128-value_num_layers=5/test_loss_hist.npy')
    # hess_penalty_test_loss_hist = jnp.load('offline/train_steps=100000-grad_penalty=0.05-hess_penalty=5e-05-value_hidden_dim=128-value_num_layers=5/test_loss_hist.npy')


    fig, ax = plt.subplots(1, 1)
    ax.set_title('Error From True Value Function')
    ax.set_xlabel('Test Iteration')
    ax.set_ylabel('Log MSE Error')

    ax.plot(jnp.log(without_penalty_test_loss_hist), label = 'Vanilla')
    ax.plot(jnp.log(grad_penalty_test_loss_hist), label = 'Gradient Penalty')
    # ax.plot(jnp.log(hess_penalty_test_loss_hist), label = 'Gradient and Hessian')
    ax.legend()
    fig.tight_layout()
    # fig.savefig('gradient_penalty.png', dpi = 300)
    plt.show()
