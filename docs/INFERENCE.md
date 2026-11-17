# Inference and deployment export

G1 and RP1 use the same three policy inference entry points:

```bash
python -m pbfm.inference.tracking --help
python -m pbfm.inference.goal --help
python -m pbfm.inference.reward --help
```

Run from the repository root after installation. Supply the run directory with
`--model-folder`; it must contain `config.json` and the selected `checkpoint/`.
The saved configuration selects the robot and calibration; do not override it
with another robot's YAML. Motion data and weights are external.

Video export requires an `ffmpeg` executable on `PATH` (`ffmpeg -version`).
Installing the Python video packages alone does not provide that command.

| Operation | Additional inputs |
| --- | --- |
| Tracking | Corresponding robot's reference motion PKL; `--motion-list` indexes this dataset |
| Goal | Corresponding reference data and goal definitions; RP1 requires an explicit `--goal-json` |
| Reward | A compatible replay shard to infer a new latent; RP1 currently supports locomotion tasks |
| ONNX export | Model checkpoint and saved run configuration; no replay required |
| Execute an exported context | Matching policy, context and sensor/control interface; no replay or reference PKL at runtime |

This guide describes PBFM checkpoints. Original RP1 Newton `.pt` files cannot be
passed directly to these commands, and G1/RP1 weights are not interchangeable.

## Tracking

Tracking includes camera, terrain selection, terrain-relative reset
and motion-reference/policy comparison options; a second preview program is not
needed. For a training terrain tile, for example:

```bash
python -m pbfm.inference.tracking --model-folder /path/to/run \
  --data-path /path/to/robot_reference_motions.pkl --motion-list 0 --save-mp4 \
  --terrain-family low_stairs_down --difficulty-row 7
```

Terrain geometry is read from that checkpoint's configuration. The family name
alone does not specify the direction of travel: center-outward traversal of
`low_stairs_down` goes uphill. Check reset position, heading and actual geometry
before labelling a video.

## Goal and reward

RP1 goal definitions must refer to its own motion data, not G1 motion IDs:

```bash
python -m pbfm.inference.goal --model-folder /path/to/rp1-run \
  --data-path /path/to/rp1_reference_motions.pkl \
  --goal-json /path/to/rp1_goals.json --save-mp4
```

The goal file is a list of entries with `motion_id`, `motion_name`, and `frames`
(a list of frame indices). Use indices from the matching dataset; joint and qpos
fields, if supplied, must have the selected robot's dimensions. There is no
bundled RP1 goal list.

For a forward locomotion reward (supported by both robots):

```bash
python -m pbfm.inference.reward --model-folder /path/to/run \
  --data-path /path/to/robot_reference_motions.pkl \
  --tasks move-ego-0-0.8 --buffer-path /path/to/replay-shard --save-mp4
```

The replay must belong to the same robot and have the fields needed for reward
relabeling. A model-only download is insufficient for this step. Once a latent
and its commands are saved, executing that context no longer needs replay.
RP1 does not yet expose the G1-specific upper-body reward task set.

## Context

Tracking advances an eight-frame reference latent window (shortened at the end).
Heading and valid planar-velocity targets use the same aligned reference. A valid
zero-velocity target means stop; an invalid target means no velocity instruction.
Goal and reward latents do not automatically create reference velocity targets.
Use the CLI's velocity-context options deliberately for custom motion names.

Keep the exported latent, heading/velocity fields, validity flags, frame rate and
coordinate convention together. A latent alone is not the complete input contract
of a command-conditioned policy. The policy always uses its estimated current
velocity, not simulator velocity labels, during deployment.

## Export

```bash
python -m scripts.export_depth_policy_onnx \
  --checkpoint /path/to/run/checkpoint --output /path/to/export/policy.onnx
python -m scripts.export_backward_encoder_onnx \
  --model-folder /path/to/run --output /path/to/export/backward_encoder.onnx --device cpu
python -m scripts.verify_depth_policy_trace --help
```

Inference and export preserve the run's saved robot calibration; current YAML
files only supply missing/relocated asset paths. When exporting a copied snapshot,
pass `--model-folder /path/to/original/run` so `policy.meta.json` includes the
correct joint order, PD gains, action scale and camera mount.

The exported `policy.onnx` includes **depth encoding, the velocity
estimator, observation normalization and the Actor**. It is not a split actor-only
graph. Backward encoding is a separate export. No critic or simulator-only velocity
label is an input to the deployed policy. Inspect the generated metadata for the
precise names and shapes instead of reusing another checkpoint's interface.

The backward encoder still needs the reference body-state features used in
training, including privileged features. It can encode reference targets offline;
executing an already-exported context does not require the real robot to supply
those privileged features or run the backward encoder at every control step.

The default depth stack is K × H × W = 8 × 18 × 32 for G1 and 8 × 18 × 36 for
RP1. RP1 uses its own camera mount, crop and 0.1–2.5 m normalization range;
do not substitute G1 preprocessing. Exported metadata is authoritative for
the selected checkpoint, including joint order, default pose, action scaling,
PD gains, camera mount and depth contract.

ONNX receives normalized float depth in [0,1]. Environment uint8 depth must be
converted to float32 and divided by 255 once; already-normalized float depth must
not be divided again. Real camera preprocessing, temporal history, context frame
alignment, action scaling and PD control remain outside the graph. ONNX parity
tests do not validate those real-robot integration steps.
