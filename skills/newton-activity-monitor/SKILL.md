---
name: newton-activity-monitor
description: >
  Build vision-based analysis and natural language Q&A using Newton's
  Activity Monitor Lens. Use when the user wants to analyze visual data
  (charts, dashboards, camera feeds, screenshots), get natural language
  explanations of sensor visualizations, or build a chat interface that
  can "see" and interpret graphical data. This skill covers the
  screenshot-to-video pipeline, focus text configuration, and SSE
  response handling.
  Do NOT use for time-series classification (use newton-machine-state).
  Do NOT use for initial API setup (use newton-setup).
---

# Newton Activity Monitor Lens

Analyze visual data and answer natural language questions using Newton's 2B-parameter vision model. The Activity Monitor accepts video input and returns natural language analysis based on configurable focus context and user instructions.

## When to Apply

- User wants to analyze dashboard screenshots or chart images
- User wants natural language explanations of sensor visualizations
- User is building a chat interface that interprets visual data
- User wants to monitor a camera feed or visual stream
- User references the "Newton Chat" or "Ask Newton" feature pattern

## Lens IDs

Two Activity Monitor lenses are mounted on prod, each pinned to a different Newton C model checkpoint. Pick by model version — the orchestration and parameter schema are identical between them.

| Lens ID | Pinned `model_version` | Use when |
|---|---|---|
| **`lns-1286e5d1d1b84a77-af311d579cc14869`** | `Newton::c2_5_8b_260413b723a9ab` (C 2.5, 8B) | **Default for new projects.** Newer, larger Newton C — substantially better structured-output compliance and richer descriptions (see the same model's improvement on text reasoning in [`newton-query-prompting`](../newton-query-prompting/SKILL.md)). |
| `lns-fd669361822b07e2-bc608aa3fdf8b4f9` | `Newton::c2_4_7b_251215a172f6d7` (C 2.4, 4.7B) | Older lens. Faster per-call but lower-quality output. Keep for backwards compatibility with existing apps that hard-coded this lens_id. |

Both lenses pin `lens_camera_processor`, `temporal_focus: 5`, `camera_buffer_size: 5`, `max_new_tokens: 512` by default. You override `focus` and `instruction` per session — see [`newton-models`](../newton-models/SKILL.md) for the full mounted-config dump and how lenses fit into the wider model catalog.

## Core Concept

The Activity Monitor Lens accepts video input and a natural language instruction, then returns a text response. It uses a **focus text** to understand domain context (what the charts represent, what thresholds mean, etc.).

## Workflow

### Step 1: Capture Visual Data

Capture the visualization as a PNG image. In a browser context, use `html2canvas`:

```typescript
import html2canvas from "html2canvas";

async function captureCharts(elementIds: string[]): Promise<Blob> {
  const canvases = await Promise.all(
    elementIds.map((id) => html2canvas(document.getElementById(id)!))
  );

  // Composite into a grid
  const composite = document.createElement("canvas");
  const cols = 2;
  const rows = Math.ceil(canvases.length / cols);
  composite.width = canvases[0].width * cols;
  composite.height = canvases[0].height * rows;
  const ctx = composite.getContext("2d")!;

  canvases.forEach((canvas, i) => {
    const x = (i % cols) * canvas.width;
    const y = Math.floor(i / cols) * canvas.height;
    ctx.drawImage(canvas, x, y);
  });

  return new Promise((resolve) => composite.toBlob(resolve!, "image/png"));
}
```

### Step 2: Convert PNG to MP4

Newton's Activity Monitor requires video input. Convert the static image to a short video using ffmpeg:

```typescript
import { execSync } from "child_process";
import { writeFileSync, readFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

function pngToMp4(pngBuffer: Buffer): Buffer {
  const tmpDir = tmpdir();
  const pngPath = join(tmpDir, `capture-${Date.now()}.png`);
  const mp4Path = join(tmpDir, `capture-${Date.now()}.mp4`);

  writeFileSync(pngPath, pngBuffer);

  execSync(
    `ffmpeg -y -loop 1 -i ${pngPath} -c:v libx264 -t 10 -pix_fmt yuv420p -r 2 ${mp4Path}`,
    { stdio: "pipe" }
  );

  return readFileSync(mp4Path);
}
```

**Parameters explained:**
- `-loop 1` — loop the single image
- `-t 10` — 10-second duration
- `-pix_fmt yuv420p` — compatibility pixel format
- `-r 2` — 2 fps (minimal, since it's a static image)

### Step 3: Upload Video

```typescript
async function uploadVideo(mp4Buffer: Buffer): Promise<string> {
  const formData = new FormData();
  formData.append(
    "file",
    new Blob([mp4Buffer], { type: "video/mp4" }),
    "capture.mp4"
  );

  const response = await newtonFetch("/files", {
    method: "POST",
    body: formData,
  });
  const data = await response.json();
  return data.file_id;
}
```

### Step 4: Create Session and Configure

```typescript
// Create session — use the C 2.5 lens for new work. Swap to the C 2.4
// lens_id (lns-fd669361822b07e2-bc608aa3fdf8b4f9) only if you need
// backwards compat with an app that hard-coded it.
const createResponse = await newtonFetch("/lens/sessions/create", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    lens_id: "lns-1286e5d1d1b84a77-af311d579cc14869",
  }),
});
const { session_id } = await createResponse.json();

// Configure with focus text and instruction
await newtonFetch("/lens/sessions/events/process", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    session_id,
    events: [{
      kind: "session.modify",
      payload: {
        focus: domainContextText,
        instruction: userQuestion,
      },
    }],
  }),
});
```

### Step 5: Connect SSE Consumer (Before Input!)

**Critical:** Set output stream before input stream to avoid race conditions.

```typescript
// Set output stream
await newtonFetch("/lens/sessions/events/process", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    session_id,
    events: [{
      kind: "output_stream.set",
      payload: {
        streams: [{
          id: "output-0",
          sink: { type: "sse" },
        }],
      },
    }],
  }),
});

// Connect SSE consumer
const sseResponse = await newtonFetch(
  `/lens/sessions/consumer/${session_id}`,
  { headers: { Accept: "text/event-stream" } }
);
```

### Step 6: Set Input and Read Response

```typescript
// Set video input
await newtonFetch("/lens/sessions/events/process", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    session_id,
    events: [{
      kind: "input_stream.set",
      payload: {
        streams: [{
          id: "input-0",
          source: { type: "video_file_reader", file_id: videoFileId },
        }],
      },
    }],
  }),
});

// Read single inference.result from SSE stream
const reader = sseResponse.body!.getReader();
const decoder = new TextDecoder();
let result = "";

while (true) {
  const { done, value } = await reader.read();
  if (done) break;

  const text = decoder.decode(value);
  // Parse SSE events, look for inference.result
  const match = text.match(/"kind"\s*:\s*"inference\.result"/);
  if (match) {
    const payload = JSON.parse(/* extract event data */);
    result = payload.result;
    reader.cancel();
    break;
  }
}
```

## Writing Effective Focus Text

The focus text gives Newton domain context. It should describe:

1. **What the visualization shows** — chart types, data meaning
2. **Normal ranges and thresholds** — what values are healthy/concerning
3. **Domain-specific knowledge** — quirks the model wouldn't know

### Example: HRV Dashboard Focus

```
You are analyzing a heart rate variability monitoring dashboard with 4 charts:
- Top-left: Stress probability over time (0-100%, threshold at 50%)
- Top-right: Heart rate in BPM (normal resting: 60-100)
- Bottom-left: RR intervals in milliseconds (normal: 600-1200ms)
- Bottom-right: RMSSD trend (higher = more relaxed, >40ms is good)

Key indicators: Rising stress probability + falling RMSSD suggests acute stress.
Short RR intervals correlate with elevated heart rate.
```

### Example: OBD2 Dashboard Focus

```
You are analyzing a vehicle diagnostics dashboard showing OBD2 sensor data:
- RPM gauge: Engine speed (idle ~700-800, redline varies by vehicle)
- Coolant temperature: Normal operating 85-105°C, overheating >110°C
- Engine load: Percentage of maximum power being used
- Throttle position: Driver pedal input percentage

Note: For hybrid vehicles (Toyota/Lexus), RPM=0 is normal when running
on electric motor. This is NOT an engine stall.
```

## Session Lifecycle

Activity Monitor sessions are typically **one-shot** — create, query, destroy. For chat-style interfaces, you can reuse a session by sending new `session.modify` events with updated instructions.

```
Capture Screenshot → Convert to MP4 → Upload → Create Session → Configure → Set Output → Set Input → Read SSE → Destroy Session
```

## Best Practices

- **Focus text is critical** — Newton's accuracy depends heavily on understanding what it's looking at
- **PNG → MP4 conversion is required** — the Activity Monitor Lens accepts video, not images
- **Keep videos short** — 10 seconds at 2 fps is sufficient for static screenshots
- **One result per query** — unlike Machine State, Activity Monitor returns a single text response
- **Set output before input** — avoids a race condition where results are lost
- **Clean up files** — delete uploaded videos after use to avoid storage accumulation
- **ffmpeg dependency** — ensure ffmpeg is available in the server environment

## Building a frontend on top of this

If you're wrapping the Activity Monitor Lens in a React/Svelte/etc. UI — vision-grounded chat, chart-reading dashboard, camera/screenshot Q&A panel — **read [`DESIGN.md`](../../DESIGN.md) at the root of this repo before writing any CSS**. The Archetype design system (Tailwind v4 + `@archetypeai/ds-lib-tokens` + Geist sans/mono + OKLCH palette + dark-first) is the expected visual language for these demos. [`newton-traffic-demo`](https://github.com/archetypeai/newton-traffic-demo) is the closest reference for vision-on-stream UIs; [`newton-swat-demo`](https://github.com/archetypeai/newton-swat-demo) and [`newton-wifi-demo`](https://github.com/archetypeai/newton-wifi-demo) show the canonical Menubar / Card / Badge (`good` / `warning` / `critical`) layout with mono numeric readouts and sharp 2px radii. Setting this up at the start is much cheaper than retrofitting later.
