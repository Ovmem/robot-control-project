import numpy as np

L1 = 0.5
L2 = 0.5


def fk(q):

    q1 = q[0]
    q2 = q[1]

    x = (
        L1 * np.cos(q1)
        + L2 * np.cos(q1 + q2)
    )

    y = (
        L1 * np.sin(q1)
        + L2 * np.sin(q1 + q2)
    )

    return np.array([x, y])


def ik(x, y):

    c2 = (
        x * x
        + y * y
        - L1 * L1
        - L2 * L2
    ) / (2 * L1 * L2)

    # 防止数值误差
    c2 = np.clip(c2, -1.0, 1.0)

    q2 = np.arccos(c2)

    q1 = (
        np.arctan2(y, x)
        - np.arctan2(
            L2 * np.sin(q2),
            L1 + L2 * np.cos(q2)
        )
    )

    return np.array([q1, q2])

def jacobian(q):

    q1 = q[0]
    q2 = q[1]

    J = np.array([

        [
            -L1*np.sin(q1)
            -L2*np.sin(q1+q2),

            -L2*np.sin(q1+q2)
        ],

        [
            L1*np.cos(q1)
            +L2*np.cos(q1+q2),

            L2*np.cos(q1+q2)
        ]
    ])

    return J