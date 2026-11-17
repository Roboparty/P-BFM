# Code responsibilities

Start with `pbfm/train.py`, then `training/workspace.py`, then
`agents/fb_cpr_aux/agent.py`. They answer different questions: what is configured,
when data/updates are produced, and how the learning objectives are calculated.

```text
train → configs → training/workspace
                    ├─ envs + terrains + perception → replay
                    ├─ motion → expert samples and contexts
                    ├─ agents → FB, D, QD/QAux/QH/QV and Actor updates
                    └─ training/checkpoint + training/evaluation

inference/{tracking,goal,reward} → shared runtime → loaded policy
                                            └─ inference/export
evaluation/ → offline rollouts, quality metrics and diagnostics
```

## Configuration: one location

`pbfm/configs/legacy_large.json` is the serialized training template.
`simulation/` contains Hydra scene/DR/observation YAMLs;
`robots/` contains separate G1 and RP1 specifications; `depth_contract.json` records
the G1 default depth contract, while RP1 selects its own depth profile.
All are packaged resources. `utils/resource_paths.py`
resolves bundled assets referenced by checkpoints without substituting external
motion datasets.

## Robot boundary

FB, the Actor, D, QD/QAux/QH/QV, replay and the training loop are shared. Robot
specifications select joint/body ordering, control, contact geometry, reference
data validation, camera mounting, rewards and DR. Assets live under
`pbfm/data/robots/{g1,rp1}/`; configuration lives under `pbfm/configs/`.

The saved run configuration carries the selected robot calibration into tracking,
goal, reward and export. This is separate training per robot, not cross-robot
weight sharing or compatibility with the source RP1 Newton checkpoints.

## Keep these boundaries

- `agents/`: model construction, losses, replay and model/optimizer serialization.
- `training/`: collection, scheduling, distributed updates and per-rank replay saves.
- `envs/`: simulation step, observations, reset and rewards; no Actor-loss ownership.
- `motion/`: reference data, conversion and body-state construction.
- `inference/`: three prompt types and deployment export.
- `evaluation/`: offline comparisons and diagnostics. Training-time evaluation
  remains in `training/evaluation/` because it affects training priorities.
- `scripts/`: user-facing commands; no second implementation of the algorithm.

The two checkpoint modules are not duplicates: `agents/checkpoint.py` owns model
and optimizer state, while `training/checkpoint.py` also owns step/rank bookkeeping
and replay shards. Likewise, robot parsers in `robots/` are not the robot assets
under `data/robots/`.

For training schedules, checkpoint loading and replay storage requirements, see
[Training](TRAINING.md).
