# Newton API Reference

## Base URL

```
https://api.u1.archetypeai.app/v0.5
```

## Authentication

All endpoints require:
```
Authorization: Bearer ${ATAI_API_KEY}
```

## Endpoints

### File Management

#### Upload File
```
POST /files
Content-Type: multipart/form-data

Body: file (binary) — CSV or video file
Response: { "file_id": "string" }
```

#### Delete File
```
DELETE /files/delete/{file_id}
Response: 200 OK
```

### Lens Sessions

#### Create Session
```
POST /lens/sessions/create
Content-Type: application/json

Body: { "lens_id": "string" }
Response: { "session_id": "string" }
```

#### Process Events
```
POST /lens/sessions/events/process
Content-Type: application/json

Body: {
  "session_id": "string",
  "events": [
    {
      "kind": "session.modify" | "input_stream.set" | "output_stream.set",
      "payload": { ... }
    }
  ]
}
```

#### SSE Consumer
```
GET /lens/sessions/consumer/{session_id}
Accept: text/event-stream

Response: Server-Sent Events stream
```

#### Destroy Session
```
POST /lens/sessions/destroy
Content-Type: application/json

Body: { "session_id": "string" }
```

## Event Types

### session.modify

Configure a session with focus data and parameters.

**Machine State Lens:**
```json
{
  "kind": "session.modify",
  "payload": {
    "input_n_shot": [
      { "label": "class_a", "file_id": "uploaded_file_id" },
      { "label": "class_b", "file_id": "uploaded_file_id" }
    ],
    "csv_configs": {
      "window_size": 16,
      "step_size": 8
    }
  }
}
```

**Activity Monitor Lens:**
```json
{
  "kind": "session.modify",
  "payload": {
    "focus": "Domain context text describing what the visual shows",
    "instruction": "User's natural language question"
  }
}
```

### input_stream.set

Set the input data source for a session.

**CSV input (Machine State):**
```json
{
  "kind": "input_stream.set",
  "payload": {
    "streams": [{
      "id": "input-0",
      "source": { "type": "csv_file_reader", "file_id": "uploaded_file_id" }
    }]
  }
}
```

**Video input (Activity Monitor):**
```json
{
  "kind": "input_stream.set",
  "payload": {
    "streams": [{
      "id": "input-0",
      "source": { "type": "video_file_reader", "file_id": "uploaded_file_id" }
    }]
  }
}
```

### output_stream.set

Configure where results are delivered.

```json
{
  "kind": "output_stream.set",
  "payload": {
    "streams": [{
      "id": "output-0",
      "sink": { "type": "sse" }
    }]
  }
}
```

## SSE Event Format

### inference.result (Machine State)
```json
{
  "kind": "inference.result",
  "payload": {
    "result": ["label", { "class_a": 0.85, "class_b": 0.15 }]
  }
}
```

### inference.result (Activity Monitor)
```json
{
  "kind": "inference.result",
  "payload": {
    "result": "Natural language response text"
  }
}
```

### sse.stream.end
Indicates the stream has completed. Often arrives 60-80 seconds after last result — use early termination to avoid waiting.

## Multipart File Upload (> 255 MB)

#### Initiate Upload
```
POST /files/uploads/initiate
Content-Type: application/json

Body: { "filename": "string", "file_type": "string", "num_bytes": integer }
Response: { "upload_id": "string", "file_uid": "string", "strategy": "multipart", "parts": [...], "expires_at": "string" }
```

#### Upload Part to S3
```
PUT {presigned_url}
Content-Length: {part_length}

Body: raw bytes
Response headers: ETag (required for complete step)
```

#### Complete Upload
```
POST /files/uploads/{upload_id}/complete
Content-Type: application/json

Body: { "parts": [{ "part_number": integer, "part_token": "etag" }] }
Response: { "file_uid": "string", "file_status": "Registered", "num_bytes": integer }
```

#### Abort Upload
```
POST /files/uploads/{upload_id}/abort
```

## Batch Jobs

#### Create Batch Job
```
POST /batch/jobs
Content-Type: application/json

Body: {
  "name": "string",
  "pipeline_type": "batch",
  "pipeline_key": "machine-state-classification",
  "inputs": { "<port_name>": [{ "file_id": "string", "metadata": {} }] },
  "parameters": { "worker": { "parallelism": integer, "config": { ... } } }
}
Response: { "id": "job_...", "status": "PENDING", ... }
```

#### List Jobs
```
GET /batch/jobs?limit=10&offset=0
```

#### Get Job Status
```
GET /batch/jobs/{job_id}
Response: { "id": "string", "status": "PENDING|RUNNING|COMPLETED|FAILED|CANCELLED", ... }
```

#### Get Job Events
```
GET /batch/jobs/{job_id}/events
Response: { "events": [{ "level": "INFO|ERROR", "message": "string", "created_at": "string" }] }
```

#### Get Job Outputs
```
GET /batch/jobs/{job_id}/outputs?limit=50&offset=0

Response: {
  "total": integer,
  "outputs": [{
    "data": {
      "ref": "https://s3...presigned-url...",  (1-hour expiry, no auth needed)
      "filename": "pred_volve_inference_part_0.csv",
      "num_bytes": integer
    }
  }]
}
```

## Known Lens IDs

| Lens | ID | Input | Output |
|------|----|-------|--------|
| Machine State | `lns-1d519091822706e2-bc108andqxf8b4os` | CSV | Classification labels + scores |
| Activity Monitor | `lns-fd669361822b07e2-bc608aa3fdf8b4f9` | Video | Natural language text |

## Known Pipeline Keys

| Pipeline | Key | Type | Input Ports | Status |
|----------|-----|------|-------------|--------|
| Machine State Classification | `machine-state-classification` | batch | `worker.inference`, `worker.n_shots` | Active on prod (v1.1.0) |
| Activity Detection | `nano-inference-pipeline` | batch | `worker.data` | No active prod version |
