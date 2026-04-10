import time

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


    model = mujoco.MjModel.from_xml_path('xml/humulum.xml')

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    live_viewer(model, data)

