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
`robots/` contains the curated G1 specification; `depth_contract.json` records
the depth input contract. All are packaged resources. `utils/resource_paths.py`
resolves bundled assets referenced by checkpoints without substituting external
motion datasets.

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
