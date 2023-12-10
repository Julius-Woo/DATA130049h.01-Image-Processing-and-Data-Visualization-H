import numpy as np


# point control: translation
def get_T(x_s, y_s, x_d, y_d):
    '''
    get translation matrix T (3D, for each point)
    :param x_s, y_s: source points' coordinates, each is an array.
    :param x_d, y_d: destination points' coordinates, each is an array.
    '''
    num = np.array(x_s).shape[0]
    T = np.zeros((num, 3, 3))
    for i in range(num):
        T[i] = np.array(
            [[1, 0, 0], [0, 1, 0], [x_d[i] - x_s[i], y_d[i] - y_s[i], 1]])
    return T

def get_dist(u, v, x, y):
    dist = np.sqrt((u - x) ** 2 + (v - y) ** 2)
    eps = 1e-8
    if dist < eps:  # avoid zero division
        return eps
    else:
        return dist

def get_weight(u, v, x_c, y_c, e):
    '''
    get weight matrix (an array, weights from point(x,y) to all control points)
    :param u, v: point in consideration
    :param x_c, y_c: control points' coordinates, each is an array.
    :param e: exponent in weight calculation
    '''
    num = np.array(x_c).shape[0]
    weight = np.zeros((num, 1))
    for i in range(num):
        x, y = x_c[i], y_c[i]
        dist = get_dist(u, v, x, y)
        weight[i] = dist ** (-e)
    weight = weight / np.sum(weight)
    return weight

def get_transcorr(u, v, x_c, y_c, T, e):
    '''
    get transformed coordinates
    :param u, v: point in consideration
    :param x_c, y_c: control points' coordinates, each is an array.
    :param T: transformation matrix
    :param e: exponent in weight calculation
    '''
    num = np.array(x_c).shape[0]
    w = get_weight(u, v, x_c, y_c, e)
    des = np.zeros((num, 2))  # destination points for every T[i] (2D)
    for i in range(num):
        s = np.array([u, v, 1])  # source point (homogeneous coordinates)
        d = np.matmul(s, T[i])  # destination point (homogeneous coordinates)
        x, y, _ = d
        des[i] = np.array([x, y])
    tmp = w * des
    new_u = np.sum(tmp[:, 0])
    new_v = np.sum(tmp[:, 1])
    return new_u, new_v