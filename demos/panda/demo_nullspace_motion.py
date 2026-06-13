import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path(
    r"models\panda\panda.xml"
)

data = mujoco.MjData(model)

# 初始姿态
data.qpos[:7] = np.array([
    0.0,
    -0.5,
    0.0,
    -2.0,
    0.0,
    1.5,
    0.7
])

mujoco.mj_forward(model, data)

body_id = model.body("hand").id

dt = 0.01

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

        # 故意给一个二级任务
        eigvals, eigvecs = np.linalg.eig(N)

        print("eigenvalues =")
        print(eigvals)

        print()

        idx = np.argmax(np.real(eigvals))

        null_dir = np.real(
            eigvecs[:, idx]
        )

        print("null direction =")
        print(null_dir)

        q_secondary = 0.5 * null_dir

        qdot = N @ q_secondary

        data.qpos[:7] += qdot * dt

        mujoco.mj_forward(
            model,
            data
        )

        viewer.sync()

        if int(data.time*100) % 100 == 0:

            print()

            print("qdot =")

            print(qdot)

        pos = data.xpos[body_id]

        print("ee =")
        print(pos)