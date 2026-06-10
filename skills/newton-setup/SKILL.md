---
name: newton-setup
description: >
  Help set up and configure Archetype AI Newton API access for a project.
  Use when the user wants to connect to Newton, configure API keys,
  initialize the archetypeai SDK, or set up environment variables for
  Archetype AI. Also use when the user encounters authentication errors
  or connection issues with Newton.
  Do NOT use for building sensor data pipelines (use newton-sensor-streaming),
  classification workflows (use newton-machine-state), or vision analysis
  (use newton-activity-monitor).
---

# Newton Setup

## When to Apply

- User wants to set up a new project with Archetype AI Newton
- User needs to configure API authentication
- User encounters `401 Unauthorized` or connection errors
- User asks how to install the `archetypeai` Python SDK
- User wants to verify their Newton API access

## API Configuration

### Base URL

```
https://api.u1.archetypeai.app/v0.5
```

### Authentication

All requests require a Bearer token via the `Authorization` header:

```
Authorization: Bearer ${ATAI_API_KEY}
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ATAI_API_KEY` | Yes | Archetype AI API key for Newton access |

### Setup Steps

1. **Obtain API key** from [Archetype AI](https://www.archetypeai.io/)

2. **Set environment variable:**
   ```bash
   # .env.local (Next.js) or .env
   ATAI_API_KEY=your_api_key_here
   ```

3. **Verify access:**
   ```bash
   curl -s -o /dev/null -w "%{http_code}" \
     -H "Authorization: Bearer $ATAI_API_KEY" \
     https://api.u1.archetypeai.app/v0.5/files
   ```
   A `200` response confirms valid credentials.

## Python SDK

### Installation

```bash
pip install archetypeai
```

### Initialization

```python
from archetypeai import ArchetypeAI

client = ArchetypeAI(api_key="your_api_key")
# Or reads from ATAI_API_KEY env var automatically
client = ArchetypeAI()
```

### SDK Capabilities

| API | Method | Use Case |
|-----|--------|----------|
| Summarize | `client.capabilities.summarize()` | Zero-shot interpretation of sensor data |
| Lens | `client.lens.create_and_run_lens()` | Create and run lens sessions |

## REST API (TypeScript / Direct HTTP)

For TypeScript or direct HTTP integrations, use the REST endpoints:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/files` | POST (multipart) | Upload CSV or video files |
| `/files/delete/{file_id}` | DELETE | Clean up uploaded files |
| `/lens/sessions/create` | POST | Create a new lens session |
| `/lens/sessions/events/process` | POST | Send events to a session |
| `/lens/sessions/consumer/{session_id}` | GET (SSE) | Read inference results |
| `/lens/sessions/destroy` | POST | Tear down a session |

### Helper Pattern

```typescript
const NEWTON_API_BASE = "https://api.u1.archetypeai.app/v0.5";

async function newtonFetch(path: string, options: RequestInit = {}) {
  const response = await fetch(`${NEWTON_API_BASE}${path}`, {
    ...options,
    headers: {
      Authorization: `Bearer ${process.env.ATAI_API_KEY}`,
      ...options.headers,
    },
  });
  if (!response.ok) {
    throw new Error(`Newton API error: ${response.status} ${response.statusText}`);
  }
  return response;
}
```

## Project Structure Recommendation

```
your-project/
├── .env.local              # ATAI_API_KEY (gitignored)
├── app/api/newton/          # Newton API routes
│   ├── status/route.ts      # Health check / availability
│   ├── query/route.ts       # Activity Monitor queries
│   └── stream/
│       ├── data/route.ts    # Receive sensor data
│       ├── start/route.ts   # Start streaming session
│       └── stop/route.ts    # Stop streaming session
└── lib/
    └── newton-stream.ts     # NewtonStreamManager singleton
```

## Status Endpoint Pattern

Always implement a status check so the frontend can gracefully degrade when Newton is unavailable:

```typescript
// app/api/newton/status/route.ts
export async function GET() {
  const available = !!process.env.ATAI_API_KEY;
  return Response.json({ available });
}
```

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `401 Unauthorized` | Invalid or missing API key | Verify `ATAI_API_KEY` is set and valid |
| `ECONNREFUSED` | Network issue | Check internet connectivity and API base URL |
| `timeout` | Long-running inference | Increase timeout; SSE streams can take 30-60s |
| Session not producing results | SSE consumer not connected before input | Always call `output_stream.set` before `input_stream.set` |
