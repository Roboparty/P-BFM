"""Historical large-network training with the 192M scratch schedule."""
from pathlib import Path

import pbfm.train as train

build_config = train.build_ufo_mjlab_config
SCRATCH_STEPS = 192020480


def build_scratch_config(**kwargs):
    kwargs.setdefault("num_env_steps", SCRATCH_STEPS)
    cfg = build_config(**kwargs)
    cfg = cfg.model_copy(update={
        "env": cfg.env.model_copy(update={"terrain_curriculum_start_global_step": 0}),
        "tags": {**cfg.tags, "start_mode": "scratch", "curriculum_start_global_step": 0},
    })
    assert cfg.agent.train.reg_coeff == 0.01
    assert cfg.agent.train.reg_coeff_velocity == 0.002
    assert cfg.agent.train.reg_coeff_heading == 0.002
    assert cfg.buffer_device == "cuda"
    return cfg


def main():
    train.DEFAULT_NUM_ENV_STEPS = SCRATCH_STEPS
    train.build_ufo_mjlab_config = build_scratch_config
    args = train.parse_args()
    # Eight ranks need more than 2048 global steps to perform optimizer updates.
    if args.smoke:
        args.num_env_steps = 32768
    if args.gpu_ids != "single":
        raise ValueError("Use --gpu-ids single; torchrun launches the eight workers")
    if (Path(args.work_dir).expanduser() / "checkpoint").exists():
        raise FileExistsError("Scratch entry point requires a new work directory")
    train.launch(args)


if __name__ == "__main__":
    main()
