import numpy as np


def quat_rotate_inverse_numpy(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    shape = v.shape
    assert q.shape[:-1] == v.shape[:-1], "q and v must have the same batch size"
    q = q.reshape(-1, 4)
    v = v.reshape(-1, 3)
    # q_w corresponds to the scalar part of the quaternion
    q_w = q[:, 0]
    # q_vec corresponds to the vector part of the quaternion
    q_vec = q[:, 1:]

    # Calculate a
    a = v * (2.0 * q_w**2 - 1.0)[:, np.newaxis]

    # Calculate b
    b = np.cross(q_vec, v) * q_w[:, np.newaxis] * 2.0

    # Calculate c
    dot_product = np.sum(q_vec * v, axis=1, keepdims=True)
    c = q_vec * dot_product * 2.0

    return (a - b + c).reshape(shape)

def quat_rotate_numpy(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    shape = v.shape
    assert q.shape[:-1] == v.shape[:-1], "q and v must have the same batch size"
    q = q.reshape(-1, 4)
    v = v.reshape(-1, 3)
    # q_w corresponds to the scalar part of the quaternion
    q_w = q[:, 0]
    # q_vec corresponds to the vector part of the quaternion
    q_vec = q[:, 1:]

    # Calculate a
    a = v * (2.0 * q_w**2 - 1.0)[:, np.newaxis]

    # Calculate b
    b = np.cross(q_vec, v) * q_w[:, np.newaxis] * 2.0

    # Calculate c
    dot_product = np.sum(q_vec * v, axis=1, keepdims=True)
    c = q_vec * dot_product * 2.0

    return (a + b + c).reshape(shape)
