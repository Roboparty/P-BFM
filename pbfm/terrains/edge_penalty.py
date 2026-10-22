"""Virtual foot/edge feedback; never adds a physical collision object.

Mechanism reference: project-instinct/InstinctMJ, commit 6edbc19,
tasks/parkour/mdp/rewards.py::volume_points_penetration (CC BY-NC-4.0).
This implementation independently extracts *exposed horizontal box-top edges*
from the compiled MuJoCo terrain. It is not Instinct's triangle-mesh Greedyconcat
extractor. Smooth heightfields/slopes are not converted to artificial edges.
Top-edge candidates are clipped by terrain-only ray tests: flat seams and buried
box tops are rejected. Clipping resolution is 5 cm; cylinder endpoints have flat
caps, matching the reference radial-penetration query (not capsules).
"""
from collections import defaultdict
import math

import numpy as np
import torch


def exposed_box_edges(model, *, group=5, sample_spacing=.05, min_drop=.04):
    """CPU, once per immutable terrain. Returns [E,2,3] world endpoints."""
    import mujoco
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    groups = np.zeros(6, dtype=np.uint8)
    requested_groups = (group,) if isinstance(group, int) else tuple(group)
    groups[list(requested_groups)] = 1
    hit = np.zeros(1, dtype=np.int32)
    down = np.array([0., 0., -1.])
    edges = []
    # Start above all terrain, including guard-ring boxes.
    selected = np.isin(model.geom_group, requested_groups) & (model.body_weldid[model.geom_bodyid] == 0)
    if (np.isin(model.geom_group, requested_groups) & (model.body_weldid[model.geom_bodyid] != 0)).any():
        raise ValueError('Terrain query groups include a moving robot body')
    ceiling = float(np.max(data.geom_xpos[selected, 2] + np.linalg.norm(model.geom_size[selected], axis=1))) + 2 if selected.any() else 10.

    def height(xy):
        ray = np.array([xy[0], xy[1], ceiling])
        distance = mujoco.mj_ray(model, data, ray, down, groups, True, -1, hit)
        return ceiling - distance if distance >= 0 else -np.inf

    for gid in np.flatnonzero(selected & (model.geom_type == mujoco.mjtGeom.mjGEOM_BOX)):
        rotation = data.geom_xmat[gid].reshape(3, 3)
        if not np.allclose(rotation[:, 2], [0, 0, 1], atol=1e-5):
            continue  # Deliberately not inventing step edges on sloped boxes.
        sx, sy, sz = model.geom_size[gid]
        corners = np.array([[-sx,-sy,sz], [sx,-sy,sz], [sx,sy,sz], [-sx,sy,sz]]) @ rotation.T + data.geom_xpos[gid]
        for a, b in zip(corners, np.roll(corners, -1, axis=0)):
            length = np.linalg.norm(b-a)
            if length < 1e-5:
                continue
            count = max(1, math.ceil(length/sample_spacing))
            fractions = np.linspace(0, 1, count+1)
            delta = (b-a)/length
            normal = np.array([-delta[1], delta[0]]) * .012
            admitted = []
            for left, right in zip(fractions[:-1], fractions[1:]):
                midpoint = a + .5*(left+right)*(b-a)
                h1, h2 = height(midpoint[:2]+normal), height(midpoint[:2]-normal)
                top = max(h1, h2)
                # One side must really be this exposed top and the other lower.
                admitted.append(abs(top-a[2]) < .003 and top-min(h1,h2) >= min_drop)
            start = None
            for i, ok in enumerate(admitted+[False]):
                if ok and start is None:
                    start = i
                elif not ok and start is not None:
                    edges.append([a+fractions[start]*(b-a), a+fractions[i]*(b-a)])
                    start = None
    if not edges:
        return np.empty((0, 2, 3), dtype=np.float32)
    # Duplicate boundaries can occur in stacked rings. Max depth also prevents
    # duplicate cylinders multiplying a point's cost.
    canonical = []
    for a, b in edges:
        if tuple(a) > tuple(b):
            a, b = b, a
        canonical.append(np.concatenate((a,b)))
    return np.unique(np.round(canonical, 5), axis=0).reshape(-1,2,3).astype(np.float32)


def cylinder_depth(points, segments, radius=.05):
    """Small CPU/Torch oracle for tests. No singularity on the cylinder axis."""
    if not len(segments):
        return points.new_zeros(points.shape[:-1])
    a, b = segments[:,0], segments[:,1]
    ab = b-a
    lengths = torch.linalg.vector_norm(ab, dim=-1)
    if (lengths <= 0).any():
        raise ValueError('Degenerate edge')
    axes = ab/lengths[:,None]
    ap = points[...,None,:]-a
    axial = (ap*axes).sum(-1)
    radial = torch.linalg.vector_norm(ap-axial[...,None]*axes, dim=-1)
    depth = (radius-radial).clamp_min(0)
    depth = torch.where((axial >= 0) & (axial <= lengths), depth, 0.)
    return depth.amax(-1)


class VirtualTerrainEdges:
    def __init__(self, segments, *, device, radius=.05, cell_size=.5):
        if not 0 < radius < cell_size:
            raise ValueError('Invalid virtual edge radius/grid size')
        self.segments_np = np.asarray(segments, dtype=np.float32).reshape(-1,2,3)
        self.radius = float(radius)
        self.device = torch.device(device)
        self.segments = torch.as_tensor(self.segments_np, device=device)
        self.cell_size = float(cell_size)
        if not len(segments):
            return
        # Insert into every cell intersected by the radius-expanded AABB.
        # Query only the containing cell; there is no unbounded all-edges scan.
        self.origin = self.segments_np[:,:,:2].min((0,1))-radius
        maximum = self.segments_np[:,:,:2].max((0,1))+radius
        self.res = np.floor((maximum-self.origin)/cell_size).astype(int)+1
        cells = defaultdict(list)
        for i, edge in enumerate(self.segments_np):
            low = np.floor((edge[:,:2].min(0)-radius-self.origin)/cell_size).astype(int)
            high = np.floor((edge[:,:2].max(0)+radius-self.origin)/cell_size).astype(int)
            for x in range(low[0],high[0]+1):
                for y in range(low[1],high[1]+1):
                    cells[x*self.res[1]+y].append(i)
        counts = np.array([len(cells[i]) for i in range(int(np.prod(self.res)))])
        offsets = np.concatenate(([0],counts.cumsum())).astype(np.int32)
        ids = np.array([j for i in range(len(counts)) for j in cells[i]], dtype=np.int32)
        self.offsets = torch.as_tensor(offsets,device=device)
        self.ids = torch.as_tensor(ids,device=device)
        self.starts = self.segments[:,0].contiguous()
        self.ends = self.segments[:,1].contiguous()

    @torch.no_grad()
    def depth(self, points):
        if not len(self.segments):
            return points.new_zeros(points.shape[:-1])
        if points.device.type != 'cuda':
            return cylinder_depth(points, self.segments, self.radius)
        import warp as wp
        from pbfm.terrains.edge_penalty_warp import query_edge_depth
        flat = points.reshape(-1,3).contiguous()
        result = torch.zeros(len(flat),device=points.device)
        wp.launch(query_edge_depth,dim=len(flat),inputs=[
            wp.from_torch(flat,dtype=wp.vec3), wp.from_torch(self.starts,dtype=wp.vec3),
            wp.from_torch(self.ends,dtype=wp.vec3), wp.from_torch(self.offsets,dtype=wp.int32),
            wp.from_torch(self.ids,dtype=wp.int32), float(self.origin[0]), float(self.origin[1]),
            int(self.res[0]), int(self.res[1]), self.cell_size, self.radius,
            wp.from_torch(result,dtype=wp.float32)],device=str(points.device),
            stream=wp.stream_from_torch(torch.cuda.current_stream(points.device)))
        return result.reshape(points.shape[:-1])


def foot_volume_points(device, bounds=((-0.025, 0.12), (-0.03, 0.03), (-0.04, 0.0))):
    """100 points/foot, ankle-roll local frame; G1 bare-foot bounds."""
    return torch.cartesian_prod(*(torch.linspace(*bound, count, device=device)
                                 for bound, count in zip(bounds, (10, 5, 2))))


def foot_points_world(position, quaternion_xyzw, linear_velocity, angular_velocity, points):
    # Rotation of the local lever arm, not just of foot-origin velocity.
    qv = quaternion_xyzw[...,:3].unsqueeze(-2)
    qw = quaternion_xyzw[...,3:].unsqueeze(-2)
    p = points.expand(*position.shape[:-1],len(points),3)
    uv = torch.cross(qv.expand_as(p),p,dim=-1)
    rotated = p + 2*(qw*uv + torch.cross(qv.expand_as(p),uv,dim=-1))
    world = position.unsqueeze(-2)+rotated
    velocity = linear_velocity.unsqueeze(-2)+torch.cross(angular_velocity.unsqueeze(-2).expand_as(rotated),rotated,dim=-1)
    return world, velocity


def penetration_cost(depth, velocity, tolerance=0.):
    return ((depth > tolerance)*depth*(torch.linalg.vector_norm(velocity,dim=-1)+1e-6)).sum((-1,-2))
