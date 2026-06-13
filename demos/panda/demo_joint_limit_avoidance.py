import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path(
    r"models/panda/panda.xml"
)

data = mujoco.MjData(model)

# 故意放一个很别扭的姿态
data.qpos[:7] = np.array([
    1.5,
    -1.5,
    1.5,
    -2.5,
    2.0,
    1.5,
    2.0
])

mujoco.mj_forward(model, data)

body_id = model.body("hand").id

dt = 0.01

q_mid = np.zeros(7)

k_sec = 0.5

with mujoco.viewer.launch_passive(
    model,
    data
) as viewer:

    while viewer.is_running():

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))

        mujoco.mj_jacBody(
            model,
            data,
            jacp,
            jacr,
            body_id
        )

        J = np.vstack([
            jacp[:, :7],
            jacr[:, :7]
        ])

        N = (
            np.eye(7)
            -
            np.linalg.pinv(J) @ J
        )

        q = data.qpos[:7].copy()

        q_sec = (
            -k_sec
            *
            (q - q_mid)
        )

        qdot = N @ q_sec

        data.qpos[:7] += qdot * dt

        mujoco.mj_forward(
            model,
            data
        )

        if int(data.time * 100) % 100 == 0:

            print()

            print("joint norm =")

            print(
                np.linalg.norm(q)
            )

        viewer.sync()