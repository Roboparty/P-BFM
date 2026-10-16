"""Release scheduling must not silently change the historical agent or DR."""
from unittest.mock import patch

import pytest

from scripts import train_legacy_scratch as scratch


def test_only_curriculum_clock_and_tags_change(tmp_path):
    # Building a CUDA configuration does not allocate tensors or initialize CUDA.
    kwargs = dict(device="cuda", work_dir=str(tmp_path), num_env_steps=scratch.SCRATCH_STEPS)
    original = scratch.build_config(**kwargs).model_dump(mode="json")
    actual = scratch.build_scratch_config(**kwargs).model_dump(mode="json")
    original["env"]["terrain_curriculum_start_global_step"] = 0
    original["tags"].update(start_mode="scratch", curriculum_start_global_step=0)
    assert actual == original
    assert scratch.build_scratch_config(device="cuda").num_env_steps == 192020480


@pytest.mark.parametrize("smoke, steps", [(False, 192020480), (True, 32768)])
def test_cli_scratch_schedule(tmp_path, smoke, steps):
    argv = ["train_legacy_scratch", "--gpu-ids", "single", "--work-dir", str(tmp_path)]
    if smoke:
        argv.append("--smoke")
    with patch("sys.argv", argv), patch.object(scratch.train, "launch") as launch, \
            patch.object(scratch.train, "DEFAULT_NUM_ENV_STEPS"), \
            patch.object(scratch.train, "build_ufo_mjlab_config"):
        scratch.main()
    assert launch.call_args.args[0].num_env_steps == steps


def test_existing_checkpoint_cannot_be_overwritten(tmp_path):
    (tmp_path / "checkpoint").mkdir()
    argv = ["train_legacy_scratch", "--gpu-ids", "single", "--work-dir", str(tmp_path)]
    with patch("sys.argv", argv), patch.object(scratch.train, "launch") as launch, \
            patch.object(scratch.train, "DEFAULT_NUM_ENV_STEPS"), \
            patch.object(scratch.train, "build_ufo_mjlab_config"):
        with pytest.raises(FileExistsError):
            scratch.main()
    launch.assert_not_called()


def test_rejects_nested_gpu_launch(tmp_path):
    argv = ["train_legacy_scratch", "--gpu-ids", "0,1", "--work-dir", str(tmp_path)]
    with patch("sys.argv", argv), patch.object(scratch.train, "launch") as launch, \
            patch.object(scratch.train, "DEFAULT_NUM_ENV_STEPS"), \
            patch.object(scratch.train, "build_ufo_mjlab_config"):
        with pytest.raises(ValueError, match="torchrun"):
            scratch.main()
    launch.assert_not_called()
