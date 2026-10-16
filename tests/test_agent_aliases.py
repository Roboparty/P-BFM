from __future__ import annotations

import io
import sys
import unittest
from unittest.mock import patch

from pbfm.agents.presets import build_fb_depth_agent
from pbfm.train import (
    _default_update_z_every_step,
    _select_device_and_rank,
    build_ufo_mjlab_config,
    canonical_agent_name,
    parse_args,
)


class AgentAliasesTest(unittest.TestCase):
    def _parse(self, *args: str):
        stderr = io.StringIO()
        with patch.object(sys, "argv", ["train.py", *args]), patch("sys.stderr", stderr):
            parsed = parse_args()
        return parsed, stderr.getvalue()

    def test_canonical_agent_names(self) -> None:
        self.assertEqual(canonical_agent_name("fb_depth"), "fb_depth")
        for name in ("fb", "fb_terrain", "unsupported"):
            with self.subTest(agent=name), self.assertRaisesRegex(ValueError, "Unsupported agent preset"):
                canonical_agent_name(name)
            with self.subTest(agent=name), self.assertRaisesRegex(ValueError, "Unsupported agent preset"):
                build_ufo_mjlab_config(device="cpu", agent=name)
            with self.subTest(agent=name), self.assertRaises(SystemExit):
                self._parse("--agent", name)

    def test_agent_depth_parses(self) -> None:
        args, warning = self._parse("--agent", "fb_depth")
        self.assertEqual(args.agent, "fb_depth")
        self.assertEqual(args.update_z_every_step, 100)
        self.assertEqual(warning, "")

    def test_depth_default_update_z_every_step(self) -> None:
        self.assertEqual(_default_update_z_every_step("fb_depth"), 100)

    def test_depth_preset_builds_agent_config(self) -> None:
        selected = build_fb_depth_agent(
            device="cpu",
            compile=False,
            update_z_every_step=10,
            lr_scale=1.0,
            clip_grad_norm=0.0,
            cartwheel_aux_safe=False,
        )
        self.assertEqual(selected.name, "FBcprAuxAgent")
        self.assertEqual(selected.train.update_z_every_step, 10)

    def test_depth_build_config_uses_canonical_public_names(self) -> None:
        cfg = build_ufo_mjlab_config(
            device="cpu",
            work_dir="/tmp/ufo_agent_alias_test",
            num_envs=1,
            num_env_steps=1,
            seed=1,
            use_wandb=False,
            wandb_run_name=None,
            smoke=True,
            agent="fb_depth",
        )
        self.assertEqual(cfg.wandb_gname, "pbfm_direct_depth_v0")
        self.assertEqual(cfg.wandb_run_name, "pbfm_fb_depth")
        self.assertEqual(cfg.tags["agent"], "fb_depth")

    def test_wandb_project_cli_reaches_training_config(self) -> None:
        args, warning = self._parse("--agent", "fb_depth", "--wandb-project", "PBFM")
        self.assertEqual(warning, "")
        cfg = build_ufo_mjlab_config(
            device="cpu",
            work_dir="/tmp/ufo_wandb_project_test",
            num_envs=1,
            num_env_steps=1,
            seed=1,
            use_wandb=True,
            wandb_run_name="terrain-test",
            wandb_project=args.wandb_project,
            smoke=False,
            agent=args.agent,
        )
        self.assertEqual(cfg.wandb_pname, "PBFM")

    def test_distributed_rank_is_used_without_cuda_visible_devices(self) -> None:
        env = {"LOCAL_RANK": "3", "RANK": "3", "WORLD_SIZE": "8"}
        with patch.dict("os.environ", env, clear=True):
            self.assertEqual(_select_device_and_rank(seed=1), ("cuda:3", 3, 3, 8))


if __name__ == "__main__":
    unittest.main()
