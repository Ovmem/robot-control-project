import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path(
    "models/panda/panda.xml"
)

data = mujoco.MjData(model)

# 故意离舒适位比较远
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

target_pos = np.array([
    0.5,
    0.2,
    0.6
])

q_mid = np.zeros(7)

Kp = 2.0
Ksec = 0.5

dt = 0.01

step = 0

with mujoco.viewer.launch_passive(
    model,
    data
) as viewer:

    while viewer.is_running():

        # 当前关节
        q = data.qpos[:7].copy()

        # 当前末端位置
        x = data.xpos[body_id].copy()

        # 位置误差
        error = target_pos - x

        # 主任务速度
        xdot = Kp * error

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))

        mujoco.mj_jacBody(
            model,
            data,
            jacp,
            jacr,
            body_id
        )

        J = jacp[:, :7]

        # 主任务
        qdot_primary = (
            np.linalg.pinv(J)
            @ xdot
        )

        # Null Space
        N = (
            np.eye(7)
            -
            np.linalg.pinv(J)
            @ J
        )

        # Secondary Task
        qdot_sec = (
            -Ksec
            *
            (q - q_mid)
        )

        qdot = (
            qdot_primary
            +
            N @ qdot_sec
        )

        data.qpos[:7] += qdot * dt

        mujoco.mj_forward(
            model,
            data
        )

        if step % 100 == 0:

            print()

            print("step =", step)

            print()

            print(
                "position error =",
                np.linalg.norm(error)
            )

            print()

            print(
                "joint norm =",
                np.linalg.norm(q)
            )

        step += 1

        viewer.sync()