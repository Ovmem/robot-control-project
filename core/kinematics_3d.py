import numpy as np

def rotz(theta):

    c = np.cos(theta)
    s = np.sin(theta)

    return np.array([
        [c,-s,0],
        [s, c,0],
        [0, 0,1]
    ])

def roty(theta):

    c = np.cos(theta)
    s = np.sin(theta)

    return np.array([
        [ c,0,s],
        [ 0,1,0],
        [-s,0,c]
    ])

def fk_3d_arm(q):

    q1,q2,q3 = q

    l1 = 0.5
    l2 = 0.4
    l3 = 0.3

    #基座
    p0 = np.zeros(3)

    #第一关节
    R1 = rotz(q1)

    p1 = p0 + R1 @ np.array([
        l1,
        0,
        0
    ])

    #第二关节
    R2 = R1 @ roty(q2)

    p2 = p1 + R2 @ np.array([
        l2,
        0,
        0
    ])

    #第三关节
    R3 = R2 @ roty(q3)

    p3 = p2 + R3 @ np.array([
        l3,
        0,
        0
    ])

    return p0,p1,p2,p3
