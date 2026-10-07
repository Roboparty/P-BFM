# Import your own G1 motion

Files are read and encoded locally in your browser; they are not uploaded.
Use a motion already retargeted to the G1's 29 joints. Human BVH/FBX/SMPL data,
video, and pickle files are not accepted.

## JSON (recommended)

Download `motion-example.json`. Its required fields are:

- `schema`: `pbfm-g1-motion-v1`
- `fps`: the actual source sampling rate, from 1 to 240 Hz
- `jointNames`: all 29 G1 joint names, in the order used by `jointPosition`
- `frames`: an array of objects with `rootPosition` (3 numbers in meters),
  `rootQuaternionWxyz` (unit quaternion, scalar first), and `jointPosition`
  (29 angles in radians).

Use a right-handed, Z-up frame with the ground at Z=0. Root X/Y origin is
automatically removed. Root height and all relative motion are preserved.

## CSV

Select the real sampling rate and quaternion order before importing.
Each row contains 36 comma-separated numbers:

`root_x,root_y,root_z,quaternion_4_values,29_joint_angles`

Joint order is the order in the JSON example. A header is optional; when
present it must use `root_x,root_y,root_z`, then `root_qx,root_qy,root_qz,root_qw`
for XYZW (or `root_qw,root_qx,root_qy,root_qz` for WXYZ), then the 29 joint names.
No timestamps, extra columns, or quoted text are supported.

## Processing and limits

The importer accepts up to 120 seconds and 50 MB. It linearly resamples positions
and joint angles, and spherically interpolates root orientations to 50 Hz.
Root/whole-body velocities are differentiated from these poses and smoothed with
a sigma=2-frame Gaussian, following the reference feature convention. Joint
velocities use finite differences. A scratch MuJoCo state constructs 30 link
poses plus the training virtual head; the actual checkpoint backward encoder
produces behavioral latents. Each control step uses the normalized mean of the
next eight available embeddings. The reference pauses at its end.

This is a visualization and policy experiment, not an official benchmark.
Importing a correctly formatted motion does not guarantee successful tracking.
