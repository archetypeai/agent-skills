# Archetype AI Agent Skills

Agent skills for building applications with [Archetype AI's Newton](https://www.archetypeai.io/) — a real-time sensor intelligence platform that understands physical world data through foundation models.

Inspired by [mongodb/agent-skills](https://github.com/mongodb/agent-skills).

## Skills

| Skill | Description |
|-------|-------------|
| [atai-newton-fusion-model](skills/atai-newton-fusion-model/) | Call the Newton C 2.6 fusion model on `/query` with text, image, or video in one stateless POST — the first C checkpoint to reason over video frames via `/query` |

More skills are in review and will be added to this table as they land.

## Quick Start

### Claude Code

```bash
# Add as global skills (available in all projects)
cp -r skills/* ~/.claude/skills/

# Or add to a specific project
cp -r skills/* your-project/.claude/skills/
```

### Invoke a Skill

```
/atai-newton-fusion-model # Multimodal (text/image/video) queries on the C 2.6 fusion model
```

## Architecture

The skills here target Newton's **Direct Query API** — one stateless POST to `/query` per request, no session lifecycle, no batch jobs, no SSE plumbing:

```
Text:          POST /query ──────────────────────────────────► response
Image(s):      Upload file(s) → POST /query (file_ids) ───────► response
Video (.mp4):  Upload file → POST /query + max_frames ────────► response
Video frames:  POST /query (frames + query_metadata) ─────────► response
```

The reference scripts are built on the [official Archetype AI python client](https://github.com/archetypeai/python-client) (`pip install archetypeai`).

## API Base URL

```
https://api.u1.archetypeai.app/v0.5
```

## License

Apache-2.0
