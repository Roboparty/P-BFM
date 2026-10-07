# P-BFM

Project website for **Perceptive Behavioral Foundation Models for Humanoid Control via Unsupervised Reinforcement Learning**.

- Website: https://roboparty.github.io/P-BFM/
- Affiliation: RoboPartyLab
- Contact: xuewangusst@gmail.com

This branch contains the project webpage, figures, robot demonstration videos, and browser policy demo. The paper will be made available after its arXiv release.

## Update the website

Edit `index.html`, `style.css`, or `main.js`, and update assets in `paper/`, `posters/`, or `pbfm_site_media/`. Push website changes to `pages`; GitHub Pages publishes from the root of this branch. The `main` branch is reserved for project code.

The original anonymous review website is hosted separately.

## Visitor counter

The public footer displays page-level unique visitors using [soxft/busuanzi](https://github.com/soxft/busuanzi), starting on 2026-10-04. It uses `page_uv`, so other projects on `roboparty.github.io` do not share this count. The provider estimates visitors using browser/IP information and a local-storage identity; it is not an exact count of people. Earlier traffic is not backfilled.

The counter loads only on the public P-BFM site, not local previews. A dash remains if the external service is unavailable or blocked. The public provider does not guarantee uptime or data retention.

## Interactive policy

Open https://roboparty.github.io/P-BFM/interactive/ or use **Try the policy** on
the homepage. The model is downloaded only after **Load interactive policy**.
Tracking includes one complete LaFAN recording from each of eight motion families,
lasting approximately 2–4.5 minutes. Lossless compressed schedules load only when
selected and pause at their real end. Users can
import retargeted G1 29-joint CSV/JSON motions; encoding stays in the browser.
The import panel links to a format guide and example. Goal and reward modes
share the same live policy and connected terrain map.

The policy binary is stored in four same-origin parts below GitHub's file-size
limit. The browser reassembles it and checks SHA256 before loading. Do not edit
individual parts or mix them between builds. Publish the complete static build
to `interactive/` together with its `pbfm/config.json`. Paper and source-code
release buttons remain disabled pending their respective releases.
