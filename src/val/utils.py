import jax
from jax import Array
from jax import numpy as jnp
from einops import einsum
from mujoco.mjx import Data

def smooth_angle_wrap(theta: float):
    return jax.lax.atan2(jax.lax.sin(theta), jax.lax.cos(theta))

def compute_pendulum_error(X: Array):

    e = jnp.stack([
        smooth_angle_wrap(X[:, 0] - jnp.pi),
        X[:, 1]
    ], axis = -1)

    return e

def split_state(xt: Array, nq: int):
    return xt[:nq], xt[nq:]

def get_state(data: Data):
    return jnp.concatenate([data.qpos, data.qvel])

def select_output(f: callable, index: int):
    return lambda *args, **kwargs: f(*args, **kwargs)[index]

def projection(u, v):
    '''Projection of v onto u.'''
    return (jnp.dot(v, u) / jnp.dot(u, u)) * u

@jax.jit
def modified_gram_schmidt(V: Array) -> Array:
    d = V.shape[1]
    U = V.at[:, 0].set(V[:, 0])

    def gs_step(i, U):
        v = V[:, i]

        def body(j, v):
            mask = (j < i)
            proj = projection(U[:, j], v) * mask
            return v - proj

        v_i = jax.lax.fori_loop(0, d, body, v)
        U = U.at[:, i].set(v_i)
        return U

    U = jax.lax.fori_loop(1, d, gs_step, U)
    return U

@jax.jit
def estimate_lyapunov_hist(A: Array, dt: float):
    '''
    Computes Lyapunov exponents and returns the time history LCE(t)
    shape: (T, d)
    '''
    T, d, _ = A.shape
    L = jnp.eye(d)
    LCE = jnp.zeros(d)

    def step(carry, A_t):
        L, LCE = carry

        # advance
        L = A_t @ L

        # orthogonalize
        Q = modified_gram_schmidt(L)
        norms = jnp.linalg.norm(Q, axis = 0)

        # renormalize
        L = Q / norms

        # accumulate logs
        LCE = LCE + jnp.log(norms)

        # history value for this step
        LCE_t = LCE / dt

        return (L, LCE), LCE_t

    (_, _), hist = jax.lax.scan(step, (L, LCE), A)

    # divide by time index: hist[t] /= (t+1)
    t_idx = jnp.arange(1, T+1).reshape(-1, 1)
    hist = hist / t_idx

    return hist


def make_compute_value(cost: callable):

    def compute_value(X: Array, U: Array, gamma: float):
        '''
        computes discounted cost to go along a trajectory
        '''
    
        c = jax.vmap(cost)(X[:-1], U)

        def body_fun(V_t: Array, c_t: Array):
            V_t = c_t + gamma * V_t
            return V_t, V_t

        V_T = jnp.array(0.0)
        _, V = jax.lax.scan(body_fun, init = V_T, xs = c, reverse = True)
        return jnp.concatenate([V, V_T[None]])
    
    return jax.jit(compute_value)

def make_compute_value_grad(step: callable, pi: callable, cost: callable):

    linearize_step = jax.jacfwd(step, argnums = (0, 1))
    linearize_cost = jax.jacrev(cost, argnums = (0, 1))
    pi_x = jax.jacfwd(pi)

    def compute_value_grad(X: Array, U: Array, gamma: float):
        '''
        computes the gradient of the discounted cost to go along a trajectory
        '''

        ## compute policy gradient
        K = jax.vmap(pi_x)(X[:-1])
        ## compute dynamics gradients
        fx, fu = jax.vmap(linearize_step)(X[:-1], U)
        ## closed loop derivative
        D = fx + einsum(fu, K, 't x1 u, t u x2 -> t x1 x2')
        ## derivatives of cost
        cx, cu = jax.vmap(linearize_cost)(X[:-1], U)
        Cz = cx + einsum(cu, K, 't u, t u x -> t x')

        ## backwards recursion for value gradient
        def body_fn(Vx_t: Array, inputs: tuple):
            D_t, Cz_t = inputs
            Vx_t = gamma * D_t.T @ Vx_t + Cz_t
            return Vx_t, Vx_t
        
        ## call backwards recursion starting from zero gradient
        dx = X.shape[-1]
        Vx_T = jnp.zeros(dx)
        _, Vx = jax.lax.scan(body_fn, init = Vx_T, xs = (D, Cz), reverse = True)
        return jnp.concatenate([Vx, Vx_T[None]])
    
    return jax.jit(compute_value_grad)

def make_compute_value_taylor(step: callable, pi: callable, cost: callable):

    ## dynamics gradients
    linearize_step = jax.jacfwd(step, argnums = (0, 1))
    quadraticize_step = jax.jacfwd(linearize_step, argnums = (0, 1))
    ## cost gradients
    linearize_cost = jax.jacobian(cost, argnums = (0, 1))
    quadraticize_cost = jax.jacfwd(linearize_cost, argnums = (0, 1))
    ## policy gradients
    pi_x = jax.jit(jax.jacfwd(pi))
    pi_xx = jax.jit(jax.jacfwd(pi_x))

    def compute_value_taylor(X: Array, U: Array, gamma: float):

        ## precompute relevant tensors
        fx, fu = jax.vmap(linearize_step)(X[:-1], U)
        (fxx, fxu), (fux, fuu) = jax.vmap(quadraticize_step)(X[:-1], U)

        ## policy gradients
        K = jax.vmap(pi_x)(X[:-1])
        KK = jax.vmap(pi_xx)(X[:-1])

        ## cost gradients
        cx, cu = jax.vmap(linearize_cost)(X[:-1], U)
        (cxx, cxu), (cux, cuu) = jax.vmap(quadraticize_cost)(X[:-1], U)

        ## first order closed loop (total) derivative of dynamics
        D = fx + einsum(fu, K, 't x1 u, t u x2 -> t x1 x2')

        ## second order closed loop (total) derivative of dynamics
        H = fxx \
            + einsum(fxu, K, 't x x1 u, t u x2 -> t x x1 x2') \
            + einsum(K, fux, 't u x1, t x u x2 -> t x x1 x2') \
            + einsum(K, fuu, K, 't u1 x1, t x u1 u2, t u2 x2 -> t x x1 x2') \
            + einsum(fu, KK, 't x u, t u x1 x2 -> t x x1 x2')

        ## first derivative of instantanious cost
        Cz = cx + einsum(cu, K, 't u, t u x -> t x')

        ## hessian of instantanious cost
        Czz = cxx \
            + einsum(cxu, K, 't x1 u, t u x2 -> t x1 x2') \
            + einsum(K, cux, 't u x1, t u x2 -> t x1 x2') \
            + einsum(K, cuu, K, 't u1 x1, t u1 u2, t u2 x2 -> t x1 x2') \
            + einsum(cu, KK, 't u, t u x1 x2 -> t x1 x2')

        def body_fn(carry: tuple, input: tuple):
            
            Vx_t, Vxx_t = carry
            D_t, H_t, Cz_t, Czz_t = input

            ## step hessian back using gradient of value from the previous step
            pullback = D_t.T @ Vxx_t @ D_t
            pushforward = einsum(Vx_t, H_t, 'x, x x1 x2 -> x1 x2')
            Vxx_t = gamma * (pullback + pushforward) + Czz_t
            ## step gradient back
            Vx_t = gamma * D_t.T @ Vx_t + Cz_t

            ## package back into tuple
            carry = (Vx_t, Vxx_t)

            return carry, carry
            
        dx = X.shape[-1]
        Vx_T = jnp.zeros(dx)
        Vxx_T = jnp.zeros((dx, dx))

        _, (Vx, Vxx) = jax.lax.scan(
            body_fn, 
            init = (Vx_T, Vxx_T), 
            xs = (D, H, Cz, Czz),
            reverse = True)
        
        Vx = jnp.concatenate([Vx, Vx_T[None]])
        Vxx = jnp.concatenate([Vxx, Vxx_T[None]])
        return Vx, Vxx

    return jax.jit(compute_value_taylor)

def make_compute_ddp_hessian(step: callable, pi: callable, cost: callable):

    ## dynamics gradients
    linearize_step = jax.jacfwd(step, argnums = (0, 1))
    quadraticize_step = jax.jacfwd(linearize_step, argnums = (0, 1))
    ## cost gradients
    linearize_cost = jax.jacobian(cost, argnums = (0, 1))
    quadraticize_cost = jax.jacfwd(linearize_cost, argnums = (0, 1))
    ## policy gradients
    pi_x = jax.jit(jax.jacfwd(pi))

    def compute_ddp_hessian(X: Array, U: Array, gamma: float):

        ## precompute relevant tensors
        fx, fu = jax.vmap(linearize_step)(X[:-1], U)
        (fxx, fxu), (fux, fuu) = jax.vmap(quadraticize_step)(X[:-1], U)

        ## policy gradients
        K = jax.vmap(pi_x)(X[:-1])

        ## cost gradients
        cx, cu = jax.vmap(linearize_cost)(X[:-1], U)
        (cxx, cxu), (cux, cuu) = jax.vmap(quadraticize_cost)(X[:-1], U)

        ## first order closed loop (total) derivative of dynamics
        D = fx + einsum(fu, K, 't x1 u, t u x2 -> t x1 x2')

        ## second order closed loop (total) derivative of dynamics
        H = fxx \
            + einsum(fxu, K, 't x x1 u, t u x2 -> t x x1 x2') \
            + einsum(K, fux, 't u x1, t x u x2 -> t x x1 x2') \
            + einsum(K, fuu, K, 't u1 x1, t x u1 u2, t u2 x2 -> t x x1 x2') \

        ## first derivative of instantanious cost
        Cz = cx + einsum(cu, K, 't u, t u x -> t x')

        ## hessian of instantanious cost
        Czz = cxx \
            + einsum(cxu, K, 't x1 u, t u x2 -> t x1 x2') \
            + einsum(K, cux, 't u x1, t u x2 -> t x1 x2') \
            + einsum(K, cuu, K, 't u1 x1, t u1 u2, t u2 x2 -> t x1 x2') \

        def body_fn(carry: tuple, input: tuple):
            
            Vx_t, Vxx_t = carry
            D_t, H_t, Cz_t, Czz_t = input

            ## step hessian back using gradient of value from the previous step
            pullback = D_t.T @ Vxx_t @ D_t
            pushforward = einsum(Vx_t, H_t, 'x, x x1 x2 -> x1 x2')
            Vxx_t = gamma * (pullback + pushforward) + Czz_t
            ## step gradient back
            Vx_t = gamma * D_t.T @ Vx_t + Cz_t

            ## package back into tuple
            carry = (Vx_t, Vxx_t)

            return carry, Vxx_t
            
        dx = X.shape[-1]
        Vx_T = jnp.zeros(dx)
        Vxx_T = jnp.zeros((dx, dx))

        _, Vxx = jax.lax.scan(
            body_fn, 
            init = (Vx_T, Vxx_T), 
            xs = (D, H, Cz, Czz),
            reverse = True)
        
        return jnp.concatenate([Vxx, Vxx_T[None]])
    
    return jax.jit(compute_ddp_hessian)

def make_compute_ilqr_hessian(step: callable, pi: callable, cost: callable):

    ## dynamics gradients
    linearize_step = jax.jacfwd(step, argnums = (0, 1))
    ## cost gradients
    linearize_cost = jax.jacobian(cost, argnums = (0, 1))
    quadraticize_cost = jax.jacfwd(linearize_cost, argnums = (0, 1))
    ## policy gradients
    pi_x = jax.jit(jax.jacfwd(pi))

    def compute_ilqr_hessian(X: Array, U: Array, gamma: float):

        ## linearize dynamics
        fx, fu = jax.vmap(linearize_step)(X[:-1], U)
        ## policy gradients
        K = jax.vmap(pi_x)(X[:-1])
        (cxx, cxu), (cux, cuu) = jax.vmap(quadraticize_cost)(X[:-1], U)
        ## first order closed loop (total) derivative of dynamics
        D = fx + einsum(fu, K, 't x1 u, t u x2 -> t x1 x2')
        ## hessian of instantanious cost
        Czz = cxx \
            + einsum(cxu, K, 't x1 u, t u x2 -> t x1 x2') \
            + einsum(K, cux, 't u x1, t u x2 -> t x1 x2') \
            + einsum(K, cuu, K, 't u1 x1, t u1 u2, t u2 x2 -> t x1 x2') \

        def body_fn(Vxx_t: Array, input: tuple):
            
            D_t, Czz_t = input
            ## step hessian back using gradient of value from the previous step
            Vxx_t = gamma * D_t.T @ Vxx_t @ D_t + Czz_t

            return Vxx_t, Vxx_t
            
        dx = X.shape[-1]
        Vxx_T = jnp.zeros((dx, dx))

        _, Vxx = jax.lax.scan(
            body_fn, 
            init = Vxx_T, 
            xs = (D, Czz),
            reverse = True)
        
        return jnp.concatenate([Vxx, Vxx_T[None]])
    
    return jax.jit(compute_ilqr_hessian)