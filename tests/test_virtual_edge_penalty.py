import numpy as np
import pytest
import torch
from pbfm.terrains.edge_penalty import (
    cylinder_depth, exposed_box_edges, VirtualTerrainEdges,
    foot_points_world, foot_volume_points, penetration_cost,
)


def test_depth_axis_caps_and_no_duplicate_cost():
    edges = torch.tensor([[[0.,0.,.15],[1.,0.,.15]]])
    points = torch.tensor([[.5,0.,.15],[.5,.04,.15],[.5,.06,.15],[-.01,0.,.15]])
    expected = torch.tensor([.05,.01,0.,0.])
    torch.testing.assert_close(cylinder_depth(points,edges), expected)
    torch.testing.assert_close(cylinder_depth(points,edges.repeat(2,1,1)),expected)


def test_cost_is_speed_weighted_and_no_contact_required():
    d = torch.full((1,2,100),.01)
    v = torch.zeros(1,2,100,3)
    assert penetration_cost(d,v).item() == pytest.approx(2e-6)
    v[...,0] = 1
    assert penetration_cost(d,v).item() == pytest.approx(2.000002)
    assert penetration_cost(torch.zeros_like(d),v).item() == 0


def test_world_point_velocity_includes_angular_motion():
    p = torch.zeros(1,2,3)
    q = torch.tensor([0.,0.,0.,1.]).expand(1,2,4)
    omega = torch.tensor([0.,0.,2.]).expand_as(p)
    xyz, vel = foot_points_world(p,q,p,omega,torch.tensor([[.1,0.,0.]]))
    torch.testing.assert_close(vel,torch.tensor([0.,.2,0.]).expand(1,2,1,3))
    assert foot_volume_points('cpu').shape == (100,3)


def model(extra):
    import mujoco
    return mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="plane" group="5" size="0 0 .1"/>'+extra+'</worldbody></mujoco>')


def test_flat_seams_and_buried_edges_are_not_obstacles():
    flat = model('<geom type="box" group="5" pos="0 0 -.1" size="1 1 .1"/>')
    assert len(exposed_box_edges(flat)) == 0
    buried = model('<geom type="box" group="5" pos="0 0 .05" size=".2 .2 .05"/>'
                   '<geom type="box" group="5" pos="0 0 .075" size="1 1 .075"/>')
    edges = exposed_box_edges(buried)
    assert len(edges) == 4
    np.testing.assert_allclose(edges[:,:,2],.15)
    assert np.all(np.max(np.abs(edges[:,:,:2]),axis=(1,2)) == 1)


def test_fifteen_cm_step_only_has_exposed_top_edges():
    m = model('<geom type="box" group="5" pos="0 0 .075" size="1 1 .075"/>')
    edges = exposed_box_edges(m)
    assert len(edges) == 4
    field = VirtualTerrainEdges(edges,device='cpu')
    points = torch.tensor([[0.,-1.,.17],[0.,0.,.17],[0.,-1.,.25]])
    torch.testing.assert_close(field.depth(points),torch.tensor([.03,0.,0.]),atol=1e-6,rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU parity test')
def test_sparse_warp_query_matches_torch_and_bounds():
    import warp as wp
    wp.init()
    edges = np.array([[[0.,0.,.15],[2.,0.,.15]],[[1.,0.,.3],[1.,2.,.3]]],dtype=np.float32)
    field = VirtualTerrainEdges(edges,device='cuda:0')
    points = torch.rand(12000,3,device='cuda:0')*torch.tensor([4.,4.,.5],device='cuda:0')-torch.tensor([1.,1.,0.],device='cuda:0')
    points = torch.cat((points,torch.tensor([[0.,0.,.15],[1.,1.,.3]],device='cuda:0')))
    torch.testing.assert_close(field.depth(points),cylinder_depth(points,field.segments),atol=1e-6,rtol=1e-5)
