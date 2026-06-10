import numpy as np

def damped_pinv(J, lam=0.05):

    m = J.shape[0]

    return (
        J.T
        @ np.linalg.inv(
            J @ J.T
            + lam**2 * np.eye(m)
        )
    )