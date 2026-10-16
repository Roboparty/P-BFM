from __future__ import annotations

import torch

from pbfm.agents.nn_models import EMA


def _calibrated_normalizer() -> EMA:
    normalizer = EMA(tau=0.9, translate=False, scale=True)
    normalizer(torch.tensor([[-2.0], [-1.0], [1.0], [2.0]]))
    normalizer(torch.tensor([[-4.0], [-2.0], [2.0], [4.0]]))
    return normalizer


def test_ema_freeze_stops_updates_and_uses_frozen_scale() -> None:
    normalizer = _calibrated_normalizer()
    normalizer.freeze(global_step=38_400_000)
    mean_before = normalizer.mean.clone()
    mean_square_before = normalizer.mean_square.clone()
    counter_before = normalizer.counter.clone()
    frozen_std = normalizer.frozen_std.clone()

    x = torch.tensor([[-100.0], [100.0]])
    output = normalizer(x)

    torch.testing.assert_close(output, x / frozen_std)
    torch.testing.assert_close(normalizer.mean, mean_before)
    torch.testing.assert_close(normalizer.mean_square, mean_square_before)
    torch.testing.assert_close(normalizer.counter, counter_before)
    assert normalizer.is_frozen
    assert normalizer.frozen_at_global_step.item() == 38_400_000


def test_ema_frozen_state_survives_checkpoint_roundtrip() -> None:
    source = _calibrated_normalizer()
    source.freeze(global_step=38_404_096)

    restored = EMA(tau=0.9, translate=False, scale=True)
    restored.load_state_dict(source.state_dict())

    x = torch.tensor([[-3.0], [7.0]])
    torch.testing.assert_close(restored(x), source(x))
    assert restored.is_frozen
    assert restored.frozen_at_global_step.item() == 38_404_096
    torch.testing.assert_close(restored.frozen_std, source.frozen_std)


def test_ema_refuses_to_freeze_without_observations() -> None:
    normalizer = EMA(translate=False, scale=True)
    try:
        normalizer.freeze(global_step=1)
    except RuntimeError as exc:
        assert "before observing" in str(exc)
    else:
        raise AssertionError("Expected freezing an uninitialized EMA to fail")
