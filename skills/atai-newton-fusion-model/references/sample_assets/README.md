# Sample assets

These assets exist so the example scripts are runnable out-of-the-box.

| File | Size | Provenance |
|------|------|------------|
| `wind-turbines.png` | ~1.9 MB | AI-generated with ChatGPT (synthetic image, not a real photograph). No attribution required. Default image for the upload example in `image_query.py`. |
| `wind-turbines-small.jpg` | ~77 KB | The image above, scaled to 1024 px (ffmpeg). Used by the inline examples, which must stay under ~1.5 MB as base64. |
| `1_pass_2_pass_3_pass_B.mp4` | ~7.8 MB | Worker-assembly inspection clip (o-ring → cap → manifold → wrench). Default video for the upload example in `video_query.py`. Ground truth: all three steps PASS. Recorded by Archetype AI as a demo asset — cleared for distribution. |
| `1_pass_2_pass_3_pass_B_small.mp4` | ~470 KB | The clip above re-encoded at 480 px, no audio (ffmpeg, libx264 CRF 32). Used by the inline video examples. |
| `assembly_before.png` / `assembly_after.png` | ~400 KB each | Single frames extracted from the clip above (ffmpeg, at ~0.5s and ~28.5s): parts laid out vs. wrench tightening the cap. Used by the before/after example in `image_query.py`. |

All assets are clear to distribute (the wind-turbine image is AI-generated;
the clip and the frames extracted from it are an Archetype AI demo recording).
Replace any with your own data for more interesting demos — every example
script accepts a positional path argument.
