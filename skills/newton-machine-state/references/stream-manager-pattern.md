# NewtonStreamManager Pattern

The NewtonStreamManager is a server-side singleton that manages the lifecycle of a Machine State Lens session, buffers incoming sensor data, and runs periodic classification queries.

## Why a Singleton?

Newton sessions are stateful — they hold uploaded focus files and configuration. Creating a new session per query wastes time and resources. The singleton pattern ensures:

- Focus files are uploaded once and cached
- The session is reused across queries
- Buffer state persists between API calls
- Only one query runs at a time (prevents overlapping SSE streams)

## Implementation Notes

- The singleton lives on the server (Next.js API route or standalone Node.js process)
- It survives across HTTP requests but not across server restarts
- On error, it sets `sessionId = null` to force recreation on the next query
- Listeners receive results via a pub/sub pattern (used to push SSE to the frontend)

## Session Recovery

If a query fails (network error, session expired), the manager should:

1. Set `sessionId = null` and `sessionReady = false`
2. On the next query cycle, detect the missing session
3. Re-upload focus files and create a new session
4. Resume normal query loop

This self-healing behavior means the stream recovers automatically from transient failures.

## Scaling Considerations

- One singleton per sensor type (don't mix HRV and OBD2 in the same manager)
- For multi-user scenarios, key the singleton by user/device ID
- Consider Redis or a database for buffer persistence across server restarts
