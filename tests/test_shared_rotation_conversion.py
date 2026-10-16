"""The shared conversion preserves the G1 FK quaternion convention."""
import io

import pytest
import torch

from pbfm.motion.library.motion_utils import rotation_conversions
from pbfm.motion.library.motion_utils.rotation_conversions import matrix_to_quaternion as shared_conversion
from pbfm.utils import torch_utils
from pbfm.utils.torch_utils import matrix_to_quaternion, quaternion_to_matrix


def test_shared_matrix_conversion_identity_half_turns_and_gradient():
    assert matrix_to_quaternion is shared_conversion
    rotations = torch.eye(4, dtype=torch.float64)  # identity and half turns in wxyz
    rotations = torch.cat((rotations, torch.tensor([[.7, .2, -.3, .1]], dtype=torch.float64)))
    rotations = rotations / rotations.norm(dim=-1, keepdim=True)
    matrices = quaternion_to_matrix(rotations).detach().requires_grad_(True)
    converted = matrix_to_quaternion(matrices)
    torch.testing.assert_close(quaternion_to_matrix(converted), matrices)
    converted.sum().backward()
    assert torch.isfinite(matrices.grad).all()
    with pytest.raises(ValueError, match="Invalid rotation matrix shape"):
        matrix_to_quaternion(torch.zeros(2, 2))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("name, values", [
    ("quaternion_to_matrix", [[1., 0., 0., 0.], [0., 1., 0., 0.],
                              [0., 0., 1., 0.], [0., 0., 0., 1.], [.7, .2, -.3, .1]]),
    ("axis_angle_to_quaternion", [[0., 0., 0.], [1e-8, -1e-8, 1e-8],
                                 [torch.pi, 0., 0.], [.2, -.3, .4]]),
])
def test_shared_scripted_conversions_preserve_outputs_gradients_and_serialization(name, values, dtype):
    shared = getattr(rotation_conversions, name)
    scripted = getattr(torch_utils, name)
    assert isinstance(scripted, torch.jit.ScriptFunction)
    expected_input = torch.tensor(values, dtype=dtype, requires_grad=True)
    actual_input = expected_input.detach().clone().requires_grad_(True)
    expected, actual = shared(expected_input), scripted(actual_input)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected.sum().backward()
    actual.sum().backward()
    assert torch.isfinite(actual_input.grad).all()
    # JIT backward arithmetic may round one ULP differently from eager execution.
    torch.testing.assert_close(actual_input.grad, expected_input.grad, rtol=0, atol=torch.finfo(dtype).eps)

    serialized = io.BytesIO()
    torch.jit.save(scripted, serialized)
    serialized.seek(0)
    restored = torch.jit.load(serialized)
    torch.testing.assert_close(restored(actual_input), expected, rtol=0, atol=0)
