import mujoco
from jax import numpy as jnp
import matplotlib.pyplot as plt

from val import Dynamics
from val.utils import split_state

if __name__ == '__main__':

    # xt = jnp.array([-0.0401412, -0.14850562, -0.82079887, -0.70289018, -0.04919794, -0.49061505, -0.64328963, 1.06724115, -9.56907105, -7.28927573, -1.62271432, -5.5033596])
    # xt = jnp.array([-0.26284324, -0.17516752, -2.46493592, -1.32693578, -1.17184795, -0.51558514, -1.69930846, -2.36238637, -13.34939317, -2.95025004, -11.9435652, 6.96031027])
    # xt = jnp.array([-0.05601106, -0.26164009, 0.20310174, 0.06055344, 0.39528471, -0.26941497, -0.33254936, -0.44359306, 0.58103429, -1.96301134, 5.67718671, -3.44963713])
    # xt = jnp.array([-0.50648103, -0.76866602, -8.17837775, -2.18260624, -1.93828801, 0.78831487, -1.01119459, 0.30036342, 4.56329178, 6.7200204, -3.88158844, 0.07601326])
    xt = jnp.array([-0.5711419, -0.92552205, -3.9428642, -1.85808544, -1.75181354, -0.20106605, -1.30346027, -0.836341, -4.25710386, -4.66392829, 3.47574359, -7.07349787])

    dyn = Dynamics(path = 'xml/unrestricted_hopper.xml', dt = 0.01)
    
    renderer = mujoco.Renderer(dyn.model, height = 1080, width = 1920)
    # Create a free camera
    camera = mujoco.MjvCamera()
    camera.lookat = jnp.array([0.0, 0.0, 1.0])  # Point the camera is looking at (x, y, z)
    camera.distance = 3  # Distance from the lookat point
    camera.azimuth = 90  # Horizontal angle (degrees, 0 = looking along +x)
    camera.elevation = 0  # Vertical angle (degrees, -90 = straight down)

    data = mujoco.MjData(dyn.model)
    data.qpos, data.qvel = split_state(xt, dyn.nq)
    mujoco.mj_forward(dyn.model, data)
        
    renderer.update_scene(data, camera = camera)

    img = renderer.render()

    plt.imshow(img)
    plt.show()

    renderer.close()
