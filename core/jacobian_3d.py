import numpy as np

from core.kinematics_3d import fk_3d_arm


def numerical_jacobian(q, eps=1e-6):

    q = np.array(q)

    p0 = fk_3d_arm(q)[-1]

    J = np.zeros((3, len(q)))

    for i in range(len(q)):

        q2 = q.copy()

        q2[i] += eps

        p1 = fk_3d_arm(q2)[-1]

        J[:, i] = (p1 - p0) / eps

    return J