# Sample assets

These assets exist so the example scripts are runnable out-of-the-box.

| File | Size | Provenance |
|------|------|------------|
| `wind-turbines.png` | ~1.9 MB | AI-generated with ChatGPT (synthetic image, not a real photograph). No attribution required. Default image for `image_query.py`. |
| `1_pass_2_pass_3_pass_B.mp4` | ~7.8 MB | Worker-assembly inspection clip (o-ring → cap → manifold → wrench). Default video for `video_query.py`. Ground truth: all three steps PASS. Recorded by Archetype AI as a demo asset — cleared for distribution. |
| `assembly_before.png` / `assembly_after.png` | ~400 KB each | Single frames extracted from the clip above (ffmpeg, at ~0.5s and ~28.5s): parts laid out vs. wrench tightening the cap. Used by the `multi_image` before/after demo in `image_query.py`. |

All assets are clear to distribute (the wind-turbine image is AI-generated;
the clip and the frames extracted from it are an Archetype AI demo recording).
Replace any with your own data for more interesting demos — every example
script accepts a positional path argument.
