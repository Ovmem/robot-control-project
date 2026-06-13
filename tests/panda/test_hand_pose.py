import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path(
    "models/panda/panda.xml"
)

data = mujoco.MjData(model)

# 给一个非零姿态
data.qpos[:7] = np.array([
    0.3,
    -0.5,
    0.2,
    -1.0,
    0.4,
    1.2,
    0.1
])

mujoco.mj_forward(
    model,
    data
)

body_id = model.body(
    "hand"
).id

print()

print("position =")
print(
    data.xpos[body_id]
)

print()

print("rotation matrix =")
print(
    data.xmat[body_id].reshape(3,3)
)