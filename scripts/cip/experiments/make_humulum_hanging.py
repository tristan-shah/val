
    '''
    Inspecting Jacobians
    '''
    # unroll = make_unroll(step)
    # # xt = jnp.zeros(dyn.state_dim)
    # xt = jnp.array([
    #     4.37842171e-02, -7.06415649e-01,  4.58404688e-03,  2.38883152e+00,
    #     1.77039702e-01, -3.31996537e+00,  7.49812204e-01,  1.58174276e+01,
    #     3.25496310e+00, -1.90802900e+01,  2.19792342e+01,  3.25215461e+00,
    #     -2.50979743e+01,  1.06648931e+00, -5.34004770e-04, -2.54689731e+00,
    #     5.49031977e+00,  2.93148233e+00,  9.35608825e-01, -9.03307414e-01,
    #     -1.07136141e+01,  2.11564302e+00,  9.04544857e+00,  1.96704250e+00,
    #     3.19275699e+00, -5.27687866e+00,
    #     ])
    # U = jnp.zeros((horizon, dyn.control_dim))
    # X = unroll(xt, U)

    # compute_entropy_efficient = make_compute_entropy_efficient(step)
    # logdet_Y, logdet_W = compute_entropy_efficient(X[:-1], U)

    # compute_entropy = make_compute_entropy(dyn.nq)
    # fx, fu = jax.vmap(jax.jacfwd(step, argnums = (0, 1)))(X[:-1], U)

    # logdet_Y_og, _ = compute_entropy(fx, fu)
    # print(fx)

    # fig, ax = plt.subplots(1, 1)

    # ax.plot(logdet_Y)
    # ax.plot(logdet_Y_og)
    # # ax.plot(logdet_W)
    # plt.show()



    # dyn.render(X, path = 'vid.mp4', skip = 1, distance = 5, lookat = jnp.array([0.0, 0.0, 0.0]))
    # jnp.save('hanging.npy', X[-1])