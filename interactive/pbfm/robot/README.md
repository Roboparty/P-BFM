# PBFM browser robot and terrain assets

These local prototype assets come from the archived UFO project, not from the
upstream viewer's G1 model. The source repository license is retained verbatim in
`LICENSE.source.txt`. The robot is a Unitree G1; no separate asset license was
present in the copied robot folder, so this file does not grant additional rights.

`g1_29dof.original.xml` is unchanged, SHA256
`579c8d3d98afa207cc816f1f6bdbe2594c6153cd00d6cdf644a102a4794b5920`.
The same XML was verified in four local archived project checkouts. The original
364822528 snapshot's recorded source pathname is absent locally. This is an
archived matching project asset, not a claim of an independently verified hash
from that missing pathname. Meshes, body inertias, joint limits, collision shapes
and exclusions are preserved.

`g1_29dof.xml` adds the exact checkpoint's logged mode15 armature, dry friction
and actuator effort limits. JavaScript applies explicit PD and MJLab's DC motor
torque/speed clipping at 200 Hz, with four physics steps per 50 Hz policy step.
Nominal gains are used; training noise and domain randomization are disabled.
MuJoCo WASM 3.3.8 is a CPU/browser adaptation of the native MuJoCo-Warp training
environment, not a claim of numerical equivalence.

`scene.xml` is one connected RP1 map with the archived generator's seven
families, 10 difficulty rows × 7 columns, 5 m tiles, 50 × 35 m core, 10 m guard
ring and 2 m flat safety ring. `terrain.json` contains all 70 spawn origins.
The generator explicitly fixes map seed 0; environment seed 4831 is separate.
Stairs use the native 0.10–0.20 m curriculum range. This is not the modified
fixed 12 cm video terrain or a new benchmark result. The names `low_stairs_up`
and `low_stairs_down` describe their height toward the center: center resets
initially descend and ascend, respectively.

`build_assets.py` reproduces the export with local MuJoCo/MJLab and the archived
UFO checkout. It checks model dimensions, actuator armature/friction and equality
of exported binary heightfields with the native generator. Terrain textures were
replaced with neutral gray because buffer textures cannot be serialized by MjSpec;
collision geometry and height samples are preserved.

Default browser reset is deterministic: flat family, row 4, root position
`[-2.5, -15, 0.8]`, identity quaternion, checkpoint default joint pose, zero
velocities. It does not reproduce randomized native episode initialization.
