---
name: newton-batch-upload
description: >
  Upload large files to the Archetype AI Newton platform using multipart
  presigned URLs. Use when the user needs to upload files larger than 255 MB
  (the simple upload limit), such as large CSV datasets for batch inference
  or fine-tuning. This skill covers the 3-step presigned URL flow: initiate,
  upload parts to S3, and complete.
  Do NOT use for files under 255 MB (use simple POST /files instead).
  Do NOT use for creating batch jobs (use newton-machine-state-batch).
---

# Newton Batch Upload

Upload large files (> 255 MB) to the Newton platform using multipart presigned URLs. The server automatically splits uploads into ~400 MB parts with direct-to-S3 transfers.

## When to Apply

- User needs to upload a large CSV, JSONL, or data file (> 255 MB)
- User encounters `413 Request Entity Too Large` on simple upload
- User wants to upload training data for fine-tuning
- User wants to upload large sensor datasets for batch inference

## Simple Upload (< 255 MB)

For small files, use the standard upload:

```bash
curl -X POST "$BASE_URL/files" \
  -H "Authorization: Bearer $ATAI_API_KEY" \
  -F "file=@my_data.csv;type=text/csv"
```

Response: `{"is_valid": true, "file_id": "my_data.csv", "file_uid": "fil_..."}`

## Multipart Upload Flow (> 255 MB)

### Step 1: Initiate Upload

```bash
curl -s -X POST "$BASE_URL/files/uploads/initiate" \
  -H "Authorization: Bearer $ATAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "filename": "large_dataset.csv",
    "file_type": "text/csv",
    "num_bytes": 8035497980
  }'
```

Response includes presigned S3 URLs for each part:

```json
{
  "upload_id": "upl_...",
  "file_uid": "fil_...",
  "strategy": "multipart",
  "num_parts": 20,
  "part_size": 419430400,
  "parts": [
    {"part_number": 1, "url": "https://s3...presigned...", "offset": 0, "length": 419430400},
    {"part_number": 2, "url": "https://s3...presigned...", "offset": 419430400, "length": 419430400}
  ],
  "expires_at": "2026-04-09T02:12:58Z"
}
```

### Step 2: Upload Each Part to S3

Read the file chunk at `offset` with `length` bytes and PUT directly to the presigned URL:

```python
import requests

with open("large_dataset.csv", "rb") as f:
    for part in parts:
        f.seek(part["offset"])
        data = f.read(part["length"])
        resp = requests.put(part["url"], data=data, headers={"Content-Length": str(part["length"])})
        etag = resp.headers["ETag"].strip('"')
        completed_parts.append({"part_number": part["part_number"], "part_token": etag})
```

### Step 3: Complete Upload

```bash
curl -s -X POST "$BASE_URL/files/uploads/{upload_id}/complete" \
  -H "Authorization: Bearer $ATAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "parts": [
      {"part_number": 1, "part_token": "etag-from-s3"},
      {"part_number": 2, "part_token": "etag-from-s3"}
    ]
  }'
```

Response:
```json
{
  "file_uid": "fil_...",
  "file_name": "large_dataset.csv",
  "file_status": "Registered",
  "num_bytes": 8035497980,
  "file_attributes": {"column_headers": [...], "num_columns": 28}
}
```

### Abort (if needed)

```bash
curl -s -X POST "$BASE_URL/files/uploads/{upload_id}/abort" \
  -H "Authorization: Bearer $ATAI_API_KEY"
```

## Supported File Types

| Extension | MIME Type |
|-----------|-----------|
| `.csv` | `text/csv` |
| `.json` | `text/plain` |
| `.jsonl` | `text/plain` |
| `.txt` | `text/plain` |
| `.png` | `image/png` |
| `.jpg`/`.jpeg` | `image/jpeg` |
| `.mp4` | `video/mp4` |

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v0.5/files` | POST | Simple upload (< 255 MB) |
| `/v0.5/files/uploads/initiate` | POST | Initiate multipart upload |
| `{presigned_url}` | PUT | Upload part to S3 |
| `/v0.5/files/uploads/{upload_id}/complete` | POST | Finalize upload |
| `/v0.5/files/uploads/{upload_id}/abort` | POST | Cancel upload |
| `/v0.5/files/info` | GET | Storage summary |
| `/v0.5/files/metadata` | GET | List all files |

## Best Practices

- **Presigned URLs expire** — complete the upload before `expires_at`
- **Collect ETags** from each PUT response — they're required for the complete step
- **Abort on failure** — if any part fails, abort to clean up server-side resources
- **The server chooses the strategy** — small files get `simple` (single part), large files get `multipart`

## Example Code

See [archetypeai-batch-examples-volve](https://github.com/archetypeai/archetypeai-batch-examples-volve) for full Python, shell, and curl implementations.
