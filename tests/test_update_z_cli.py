from __future__ import annotations

import io
import sys
import unittest
from unittest.mock import patch

import torch

from pbfm.agents.presets import build_fb_depth_agent
from pbfm.train import build_ufo_mjlab_config, parse_args
from pbfm.training.diagnostics import _accumulate_metrics
from pbfm.training.config import _trajectory_output_keys


class UpdateZCliTest(unittest.TestCase):
    def _parse(self, *args: str):
        with patch.object(sys, "argv", ["train.py", *args]), patch("sys.stderr", io.StringIO()):
            return parse_args()

    def test_depth_defaults_are_preserved(self) -> None:
        self.assertEqual(self._parse().update_z_every_step, 100)
        self.assertEqual(self._parse("--agent", "fb_depth").update_z_every_step, 100)

    def test_depth_cli_value_reaches_agent_config(self) -> None:
        args = self._parse("--agent", "fb_depth", "--update-z-every-step", "37")
        selected = build_fb_depth_agent(
            device="cpu",
            compile=False,
            update_z_every_step=args.update_z_every_step,
            lr_scale=1.0,
            clip_grad_norm=0.0,
            cartwheel_aux_safe=False,
        )
        self.assertEqual(selected.train.update_z_every_step, 37)

    def test_programmatic_depth_default_remains_hundred(self) -> None:
        cfg = build_ufo_mjlab_config(
            device="cpu",
            work_dir="/tmp/ufo_update_z_test",
            num_envs=1,
            num_env_steps=1,
            seed=1,
            use_wandb=False,
            wandb_run_name=None,
            smoke=True,
            agent="fb_depth",
        )
        self.assertEqual(cfg.agent.train.update_z_every_step, 100)

    def test_depth_trajectory_buffer_keeps_aux_rewards(self) -> None:
        selected = build_fb_depth_agent(
            device="cpu",
            compile=False,
            update_z_every_step=10,
            lr_scale=1.0,
            clip_grad_norm=0.0,
            cartwheel_aux_safe=False,
        )
        self.assertIn("aux_rewards", _trajectory_output_keys(selected))

    def test_metric_accumulation_accepts_changing_keys(self) -> None:
        totals, counts = _accumulate_metrics(
            None,
            {},
            {"fb_loss": torch.tensor(2.0)},
        )
        totals, counts = _accumulate_metrics(
            totals,
            counts,
            {
                "fb_loss": torch.tensor(4.0),
                "disc_wgan_gp_loss": torch.tensor(6.0),
            },
        )
        self.assertEqual(counts, {"fb_loss": 2, "disc_wgan_gp_loss": 1})
        self.assertEqual((totals["fb_loss"] / counts["fb_loss"]).item(), 3.0)
        self.assertEqual((totals["disc_wgan_gp_loss"] / counts["disc_wgan_gp_loss"]).item(), 6.0)


if __name__ == "__main__":
    unittest.main()
