"""
Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

import torch
import numpy as np
from torch import Tensor
from typing import Tuple, List, Optional

from pbfm.motion.library.motion_utils.rotation_conversions import (
    axis_angle_to_quaternion as _axis_angle_to_quaternion,
    matrix_to_quaternion,
    quaternion_to_matrix as _quaternion_to_matrix,
)


def to_torch(x, dtype=torch.float, device='cuda:0', requires_grad=False):
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)


@torch.jit.script
def quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([x, y, z, w], dim=-1).view(shape)

    return quat


@torch.jit.script
def normalize(x, eps: float = 1e-9):
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


@torch.jit.script
def quat_apply(a, b):
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, :3]
    t = xyz.cross(b, dim=-1) * 2
    return (b + a[:, 3:] * t + xyz.cross(t, dim=-1)).view(shape)


@torch.jit.script
def quat_rotate(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c


# Old quat_rotate_inverse removed - use the version with w_last parameter below


@torch.jit.script
def quat_conjugate(a):
    shape = a.shape
    a = a.reshape(-1, 4)
    return torch.cat((-a[:, :3], a[:, -1:]), dim=-1).view(shape)


@torch.jit.script
def quat_unit(a):
    return normalize(a)


@torch.jit.script
def quat_from_angle_axis(angle, axis):
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    return quat_unit(torch.cat([xyz, w], dim=-1))


@torch.jit.script
def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


@torch.jit.script
def get_basis_vector(q, v):
    return quat_rotate(q, v)


def get_axis_params(value, axis_idx, x_value=0., dtype=np.float64, n_dims=3):
    """construct arguments to `Vec` according to axis index.
    """
    zs = np.zeros((n_dims,))
    assert axis_idx < n_dims, "the axis dim should be within the vector dimensions"
    zs[axis_idx] = 1.
    params = np.where(zs == 1., value, zs)
    params[0] = x_value
    return list(params.astype(dtype))


@torch.jit.script
def copysign(a, b):
    # type: (float, Tensor) -> Tensor
    a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a) * torch.sign(b)


@torch.jit.script
def get_euler_xyz(q):
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * \
        q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(torch.abs(sinp) >= 1, copysign(
        np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * \
        q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll % (2*np.pi), pitch % (2*np.pi), yaw % (2*np.pi)


@torch.jit.script
def quat_from_euler_xyz(roll, pitch, yaw):
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return torch.stack([qx, qy, qz, qw], dim=-1)


@torch.jit.script
def torch_rand_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    return (upper - lower) * torch.rand(*shape, device=device) + lower


@torch.jit.script
def torch_random_dir_2(shape, device):
    # type: (Tuple[int, int], str) -> Tensor
    angle = torch_rand_float(-np.pi, np.pi, shape, device).squeeze(-1)
    return torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)


@torch.jit.script
def tensor_clamp(t, min_t, max_t):
    return torch.max(torch.min(t, max_t), min_t)


@torch.jit.script
def scale(x, lower, upper):
    return (0.5 * (x + 1.0) * (upper - lower) + lower)


@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


def unscale_np(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


# ========== Quaternion utilities ==========
# These functions support w_last parameter for quaternion format flexibility

@torch.jit.script
def quat_apply(a: Tensor, b: Tensor, w_last: bool) -> Tensor:
    """Apply quaternion rotation to vector, supporting both w_last=True (xyzw) and w_last=False (wxyz) formats."""
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    if w_last:
        xyz = a[:, :3]
        w = a[:, 3:]
    else:
        xyz = a[:, 1:]
        w = a[:, :1]
    t = xyz.cross(b, dim=-1) * 2
    return (b + w * t + xyz.cross(t, dim=-1)).view(shape)

@torch.jit.script
def quat_apply_yaw(quat: Tensor, vec: Tensor, w_last: bool) -> Tensor:
    """Apply only yaw rotation from quaternion to vector."""
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.0
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec, w_last)

@torch.jit.script
def wrap_to_pi(angles):
    """Wrap angles to [-pi, pi] range."""
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles

@torch.jit.script
def quat_conjugate(a: Tensor, w_last: bool) -> Tensor:
    """Compute quaternion conjugate."""
    shape = a.shape
    a = a.reshape(-1, 4)
    if w_last:
        return torch.cat((-a[:, :3], a[:, -1:]), dim=-1).view(shape)
    else:
        return torch.cat((a[:, 0:1], -a[:, 1:]), dim=-1).view(shape)

@torch.jit.script
def quat_rotate(q: Tensor, v: Tensor, w_last: bool) -> Tensor:
    """Rotate vector by quaternion."""
    shape = q.shape
    if w_last:
        q_w = q[:, -1]
        q_vec = q[:, :3]
    else:
        q_w = q[:, 0]
        q_vec = q[:, 1:]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = (
        q_vec
        * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1)
        * 2.0
    )
    return a + b + c

@torch.jit.script
def quat_rotate_inverse(q: Tensor, v: Tensor, w_last: bool) -> Tensor:
    """Rotate vector by inverse quaternion."""
    shape = q.shape
    if w_last:
        q_w = q[:, -1]
        q_vec = q[:, :3]
    else:
        q_w = q[:, 0]
        q_vec = q[:, 1:]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = (
        q_vec
        * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1)
        * 2.0
    )
    return a - b + c

@torch.jit.script
def quat_angle_axis(x: Tensor, w_last: bool) -> Tuple[Tensor, Tensor]:
    """Convert quaternion to angle-axis representation."""
    if w_last:
        w = x[..., -1]
        axis = x[..., :3]
    else:
        w = x[..., 0]
        axis = x[..., 1:]
    s = 2 * (w**2) - 1
    angle = s.clamp(-1, 1).arccos()
    axis /= axis.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-9)
    return angle, axis

@torch.jit.script
def quat_from_angle_axis(angle: Tensor, axis: Tensor, w_last: bool) -> Tensor:
    """Convert angle-axis to quaternion."""
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    if w_last:
        return quat_unit(torch.cat([xyz, w], dim=-1))
    else:
        return quat_unit(torch.cat([w, xyz], dim=-1))

@torch.jit.script
def vec_to_heading(h_vec):
    """Convert vector to heading angle."""
    h_theta = torch.atan2(h_vec[..., 1], h_vec[..., 0])
    return h_theta

@torch.jit.script
def heading_to_quat(h_theta, w_last: bool):
    # type: (Tensor, bool) -> Tensor
    """Convert heading angle to quaternion."""
    axis = torch.zeros(
        h_theta.shape
        + [
            3,
        ],
        device=h_theta.device,
    )
    axis[..., 2] = 1
    heading_q = quat_from_angle_axis(h_theta, axis, w_last=w_last)
    return heading_q

@torch.jit.script
def quat_axis(q: Tensor, axis: int, w_last: bool) -> Tensor:
    """Get axis vector from quaternion."""
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec, w_last)

@torch.jit.script
def get_basis_vector(q: Tensor, v: Tensor, w_last: bool) -> Tensor:
    """Get basis vector rotated by quaternion."""
    return quat_rotate(q, v, w_last)

@torch.jit.script
def quat_to_angle_axis(q: Tensor, w_last: bool) -> Tuple[Tensor, Tensor]:
    """Convert quaternion to angle-axis representation."""
    min_theta = 1e-5
    if w_last:
        qx, qy, qz, qw = 0, 1, 2, 3
    else:
        qw, qx, qy, qz = 0, 1, 2, 3

    sin_theta = torch.sqrt(1 - q[..., qw] * q[..., qw])
    angle = 2 * torch.acos(q[..., qw])
    angle = normalize_angle(angle)
    sin_theta_expand = sin_theta.unsqueeze(-1)
    axis = q[..., qx:qw] / sin_theta_expand

    mask = torch.abs(sin_theta) > min_theta
    default_axis = torch.zeros_like(axis)
    default_axis[..., -1] = 1

    angle = torch.where(mask, angle, torch.zeros_like(angle))
    mask_expand = mask.unsqueeze(-1)
    axis = torch.where(mask_expand, axis, default_axis)
    return angle, axis

@torch.jit.script
def slerp(q0, q1, t):
    """Spherical linear interpolation between two quaternions."""
    cos_half_theta = torch.sum(q0 * q1, dim=-1)

    neg_mask = cos_half_theta < 0
    q1 = q1.clone()
    q1[neg_mask] = -q1[neg_mask]
    cos_half_theta = torch.abs(cos_half_theta)
    cos_half_theta = torch.unsqueeze(cos_half_theta, dim=-1)

    half_theta = torch.acos(cos_half_theta)
    sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta)

    ratioA = torch.sin((1 - t) * half_theta) / sin_half_theta
    ratioB = torch.sin(t * half_theta) / sin_half_theta

    new_q = ratioA * q0 + ratioB * q1

    new_q = torch.where(torch.abs(sin_half_theta) < 0.001, 0.5 * q0 + 0.5 * q1, new_q)
    new_q = torch.where(torch.abs(cos_half_theta) >= 1, q0, new_q)

    return new_q

@torch.jit.script
def angle_axis_to_exp_map(angle, axis):
    """Convert angle-axis to exponential map."""
    angle_expand = angle.unsqueeze(-1)
    exp_map = angle_expand * axis
    return exp_map

@torch.jit.script
def my_quat_rotate(q, v):
    """Rotate vector by quaternion (assumes w_last=True format)."""
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c

@torch.jit.script
def quat_to_tan_norm(q: Tensor, w_last: bool) -> Tensor:
    """Represent rotation using tangent and normal vectors."""
    ref_tan = torch.zeros_like(q[..., 0:3])
    ref_tan[..., 0] = 1
    if w_last:
        tan = my_quat_rotate(q, ref_tan)
    else:
        raise NotImplementedError
        
    ref_norm = torch.zeros_like(q[..., 0:3])
    ref_norm[..., -1] = 1
    if w_last:
        norm = my_quat_rotate(q, ref_norm)
    else:
        raise NotImplementedError

    norm_tan = torch.cat([tan, norm], dim=len(tan.shape) - 1)
    return norm_tan

@torch.jit.script
def calc_heading(q):
    """Calculate heading direction from quaternion."""
    ref_dir = torch.zeros_like(q[..., 0:3])
    ref_dir[..., 0] = 1
    rot_dir = my_quat_rotate(q, ref_dir)

    heading = torch.atan2(rot_dir[..., 1], rot_dir[..., 0])
    return heading

@torch.jit.script
def quat_to_exp_map(q: Tensor, w_last: bool) -> Tensor:
    """Convert quaternion to exponential map."""
    angle, axis = quat_to_angle_axis(q, w_last)
    exp_map = angle_axis_to_exp_map(angle, axis)
    return exp_map

@torch.jit.script
def calc_heading_quat(q, w_last: bool):
    # type: (Tensor, bool) -> Tensor
    """Calculate heading rotation from quaternion."""
    heading = calc_heading(q)
    axis = torch.zeros_like(q[..., 0:3])
    axis[..., 2] = 1

    heading_q = quat_from_angle_axis(heading, axis, w_last=w_last)
    return heading_q

@torch.jit.script
def calc_heading_quat_inv(q, w_last: bool):
    # type: (Tensor, bool) -> Tensor
    """Calculate inverse heading rotation from quaternion."""
    heading = calc_heading(q)
    axis = torch.zeros_like(q[..., 0:3])
    axis[..., 2] = 1

    heading_q = quat_from_angle_axis(-heading, axis, w_last=w_last)
    return heading_q

@torch.jit.script
def quat_inverse(x, w_last: bool):
    # type: (Tensor, bool) -> Tensor
    """Compute inverse quaternion."""
    return quat_conjugate(x, w_last=w_last)

@torch.jit.script
def get_euler_xyz(q: Tensor, w_last: bool) -> Tuple[Tensor, Tensor, Tensor]:
    """Convert quaternion to Euler angles (roll, pitch, yaw)."""
    if w_last:
        qx, qy, qz, qw = 0, 1, 2, 3
    else:
        qw, qx, qy, qz = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = (
        q[:, qw] * q[:, qw]
        - q[:, qx] * q[:, qx]
        - q[:, qy] * q[:, qy]
        + q[:, qz] * q[:, qz]
    )
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(
        torch.abs(sinp) >= 1, copysign(np.pi / 2.0, sinp), torch.asin(sinp)
    )

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = (
        q[:, qw] * q[:, qw]
        + q[:, qx] * q[:, qx]
        - q[:, qy] * q[:, qy]
        - q[:, qz] * q[:, qz]
    )
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll % (2 * np.pi), pitch % (2 * np.pi), yaw % (2 * np.pi)

# @torch.jit.script
def get_euler_xyz_in_tensor(q):
    """Convert quaternion to Euler angles as tensor (assumes w_last=True format)."""
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * \
                q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(
        torch.abs(sinp) >= 1, copysign(np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * \
                q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack((roll, pitch, yaw), dim=-1)

@torch.jit.script
def quat_pos(x):
    """Make all the real part of the quaternion positive."""
    q = x
    z = (q[..., 3:] < 0).float()
    q = (1 - 2 * z) * q
    return q

@torch.jit.script
def is_valid_quat(q):
    """Check if quaternion is valid."""
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return (w * w + x * x + y * y + z * z).allclose(torch.ones_like(w))

@torch.jit.script
def quat_normalize(q):
    """Normalize quaternion."""
    q = quat_unit(quat_pos(q))
    return q

@torch.jit.script
def quat_mul(a, b, w_last: bool):
    # type: (Tensor, Tensor, bool) -> Tensor
    """Multiply two quaternions."""
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    if w_last:
        x1, y1, z1, w1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
        x2, y2, z2, w2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    else:
        w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
        w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    if w_last:
        quat = torch.stack([x, y, z, w], dim=-1).view(shape)
    else:
        quat = torch.stack([w, x, y, z], dim=-1).view(shape)

    return quat

@torch.jit.script
def quat_mul_norm(x, y, w_last: bool):
    # type: (Tensor, Tensor, bool) -> Tensor
    """Multiply and normalize quaternions."""
    return quat_unit(quat_mul(x, y, w_last))

@torch.jit.script
def quat_identity(shape: List[int]):
    """Construct identity quaternion given shape."""
    w = torch.ones(shape + [1])
    xyz = torch.zeros(shape + [3])
    q = torch.cat([xyz, w], dim=-1)
    return quat_normalize(q)

@torch.jit.script
def quat_identity_like(x):
    """Construct identity quaternion with same shape."""
    return quat_identity(x.shape[:-1])

@torch.jit.script
def transform_from_rotation_translation(
    r: Optional[torch.Tensor] = None, t: Optional[torch.Tensor] = None
):
    """Construct transform from quaternion and translation."""
    assert r is not None or t is not None, "rotation and translation can't be all None"
    if r is None:
        assert t is not None
        r = quat_identity(list(t.shape))
    if t is None:
        t = torch.zeros(list(r.shape) + [3])
    return torch.cat([r, t], dim=-1)

@torch.jit.script
def transform_rotation(x):
    """Get rotation from transform."""
    return x[..., :4]

@torch.jit.script
def transform_translation(x):
    """Get translation from transform."""
    return x[..., 4:]

@torch.jit.script
def transform_mul(x, y):
    """Combine two transformations."""
    z = transform_from_rotation_translation(
        r=quat_mul_norm(transform_rotation(x), transform_rotation(y), w_last=True),
        t=quat_rotate(transform_rotation(x), transform_translation(y), w_last=True)
        + transform_translation(x),
    )
    return z

quaternion_to_matrix = torch.jit.script(_quaternion_to_matrix)
axis_angle_to_quaternion = torch.jit.script(_axis_angle_to_quaternion)

# @torch.jit.script
def wxyz_to_xyzw(quat):
    """Convert quaternion from wxyz to xyzw format."""
    return quat[..., [1, 2, 3, 0]]

# @torch.jit.script
def xyzw_to_wxyz(quat):
    """Convert quaternion from xyzw to wxyz format."""
    return quat[..., [3, 0, 1, 2]]

def quat_w_first(rot):
    """Convert quaternion to w-first format."""
    rot = torch.cat([rot[..., [-1]], rot[..., :-1]], -1)
    return rot
