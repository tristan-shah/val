import time
from argparse import ArgumentParser
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ['MUJOCO_GL'] = 'egl'
from pathlib import Path

import jax
from jax import numpy as jnp
import matplotlib.pyplot as plt

from val import Dynamics, make_step, make_unroll
from val.cem import CEM
from val.cip import make_compute_cip




import mujoco
from mujoco import MjModel, MjData
def live_viewer(model: MjModel, data: MjData):

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():

            step_start = time.time()

            mujoco.mj_step(model, data)
            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)

            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    return None

if __name__ == '__main__':


    model = mujoco.MjModel.from_xml_path('xml/humanoid.xml')

    data = mujoco.MjData(model)
    # data.qpos = mjx_data.qpos
    # data.qvel = mjx_data.qvel
    mujoco.mj_forward(model, data)

    live_viewer(model, data)





    # horizon = 1000
    # dt = 0.003

    # dyn = Dynamics('xml/humulum.xml', dt = dt)

    # low = dyn.mjx_model.actuator_ctrlrange[:, 0]
    # high = dyn.mjx_model.actuator_ctrlrange[:, 1]

    # print(dyn.state_dim, dyn.control_dim)

    # step = make_step(dyn)
    # unroll = make_unroll(step)


    # qpos = dyn.mjx_model.key_qpos[0]
    # qvel = jnp.zeros(dyn.mjx_model.nv)
    # xt = jnp.concatenate([qpos, qvel])

    # # U = jnp.zeros((horizon, dyn.control_dim))
    # U = jax.random.uniform(jax.random.key(0), (horizon, dyn.control_dim), minval = low, maxval = high)
    # X = unroll(xt, U)

    # dyn.render(X, path = 'vid.mp4', skip = 3, distance = 4, lookat = jnp.array([0, 0, 1]))



