# Third-party notices — P-BFM browser demo

This workspace began from [Axellwppr/humanoid-policy-viewer](https://github.com/Axellwppr/humanoid-policy-viewer), copyright 2026 Qingzhou Lu, BSD-3-Clause. Its original license is preserved in [LICENSE](LICENSE). Original notices are retained in `THIRD_PARTY_NOTICES.upstream.md`. The P-BFM runtime, observation adapter, live depth, native-coordinate renderer and UI are implemented in `src/pbfm/`.

## Browser libraries

- [MuJoCo](https://github.com/google-deepmind/mujoco), including `mujoco-js` 0.0.7 / MuJoCo WASM 3.3.8: [Apache-2.0](LICENSES/Apache-2.0.txt).
- [Three.js](https://github.com/mrdoob/three.js), 0.151.3: [MIT](LICENSES/Three.js-MIT.txt).
- [ONNX Runtime Web](https://github.com/microsoft/onnxruntime), 1.30.0: MIT; [upstream license](https://github.com/microsoft/onnxruntime/blob/v1.30.0/LICENSE) and [third-party notices](https://github.com/microsoft/onnxruntime/blob/v1.30.0/ThirdPartyNotices.txt).
- Vite 6.2.2 is used as a development/build tool (MIT).

Legacy Vue/Vuetify/example policy assets are retained only in source history or the local upstream archive. They are not part of this browser bundle. In particular, the upstream viewer's G1 policy is not the policy used here.

## Robot, terrain and motion assets

The G1 description/meshes and terrain generator are from the locally archived UFO project. The archive's source license and exact provenance are retained in [pbfm/robot/README.md](pbfm/robot/README.md) and [LICENSE.source.txt](pbfm/robot/LICENSE.source.txt). [Unitree's asset notice](LICENSES/Unitree-BSD-3-Clause.txt) is also retained. Motion sources retain their respective original terms.

## P-BFM inference weights

This demo uses the author's fixed 364,822,528-step P-BFM export. See `pbfm/config.json` for the model SHA256 and observation contract. The upstream viewer's license is not an additional license grant for P-BFM weights, archived assets, or motion data. No training checkpoint, optimizer, replay buffer, or critic weights are included.
