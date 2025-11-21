import jax
from jax import numpy as jnp
from flax import nnx

class MLP(nnx.Module):
  def __init__(self, din: int, dmid: int, dout: int, *, rngs: nnx.Rngs):
    self.linear1 = nnx.Linear(din, dmid, rngs=rngs)
    self.dropout = nnx.Dropout(rate=0.1)
    self.bn = nnx.BatchNorm(dmid, rngs=rngs)
    self.linear2 = nnx.Linear(dmid, dout, rngs=rngs)

  def __call__(self, x: jax.Array, rngs: nnx.Rngs):
    x = nnx.gelu(self.dropout(self.bn(self.linear1(x)), rngs=rngs))
    return self.linear2(x)

model = MLP(2, 16, 5, rngs=nnx.Rngs(0))

y = model(x=jnp.ones((3, 2)), rngs=nnx.Rngs(1))

import optax

rngs = nnx.Rngs(0)


# An MLP containing 2 custom `Linear` layers, 1 `nnx.Dropout` layer, 1 `nnx.BatchNorm` layer.
model = MLP(2, 16, 10, rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

@nnx.jit  # Automatic state management
def train_step(model, optimizer, x, y, rngs):
  def loss_fn(model: MLP, rngs: nnx.Rngs):
    y_pred = model(x, rngs)
    return jnp.mean((y_pred - y) ** 2)

  loss, grads = nnx.value_and_grad(loss_fn)(model, rngs)
  optimizer.update(grads)  # In place updates.

  return loss

x, y = jnp.ones((5, 2)), jnp.ones((5, 10))
loss = train_step(model, optimizer, x, y, rngs)

print(f'{loss = }')
print(f'{optimizer.step.value = }')