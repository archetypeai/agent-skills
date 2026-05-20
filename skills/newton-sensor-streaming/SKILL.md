---
name: newton-sensor-streaming
description: >
  Build real-time sensor data ingestion pipelines for Newton. Use when
  the user wants to connect hardware sensors (BLE heart rate monitors,
  OBD2 adapters, serial devices, IoT sensors) to Newton, implement data
  buffering and batching, or build a streaming architecture that feeds
  sensor data into Newton lenses. Covers Web Bluetooth, Web Serial,
  OBD2 protocols, and the NewtonStreamManager singleton pattern.
  Do NOT use for classification logic (use newton-machine-state) or
  vision analysis (use newton-activity-monitor).
---

# Newton Sensor Streaming

Patterns for ingesting real-time sensor data from hardware devices and feeding it into Newton lenses for classification or analysis.

## When to Apply

- User wants to connect a BLE device (heart rate monitor, environmental sensor)
- User wants to read OBD2 vehicle diagnostics
- User is building a real-time sensor dashboard with Newton integration
- User asks about data buffering, batching, or streaming architecture
- User wants to implement the NewtonStreamManager pattern

## Supported Ingestion Methods

| Method | Use Case | Browser API |
|--------|----------|-------------|
| Web Bluetooth | BLE sensors (HR monitors, environmental) | `navigator.bluetooth` |
| Web Serial | Serial/USB devices | `navigator.serial` |
| OBD2 via BLE | Vehicle diagnostics (ELM327 adapters) | `navigator.bluetooth` |
| WebSocket | Network-connected sensors | `WebSocket` |
| HTTP polling | REST API sensors | `fetch` |

## Architecture Overview

```
Hardware Sensor
    ↓ (BLE / Serial / OBD2)
Browser (Web Bluetooth/Serial API)
    ↓ (POST /api/newton/stream/data)
Server-side NewtonStreamManager
    ↓ (every 15s, batch → CSV → upload)
Newton Machine State Lens
    ↓ (SSE inference results)
Frontend Dashboard (real-time updates)
```

## Web Bluetooth Pattern (BLE Sensors)

### Heart Rate Monitor Example

```typescript
async function connectHRMonitor() {
  const device = await navigator.bluetooth.requestDevice({
    filters: [{ services: ["heart_rate"] }],
  });

  const server = await device.gatt!.connect();
  const service = await server.getPrimaryService("heart_rate");
  const characteristic = await service.getCharacteristic("heart_rate_measurement");

  await characteristic.startNotifications();
  characteristic.addEventListener("characteristicvaluechanged", (event) => {
    const value = (event.target as BluetoothRemoteGATTCharacteristic).value!;
    const heartRate = parseHeartRate(value);

    // Send to server for Newton processing
    fetch("/api/newton/stream/data", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ heartRate, timestamp: Date.now() }),
    });
  });
}

function parseHeartRate(value: DataView): { hr: number; rrIntervals: number[] } {
  const flags = value.getUint8(0);
  const is16Bit = flags & 0x01;
  let offset = 1;

  const hr = is16Bit ? value.getUint16(offset, true) : value.getUint8(offset);
  offset += is16Bit ? 2 : 1;

  const rrIntervals: number[] = [];
  if (flags & 0x10) {
    while (offset < value.byteLength) {
      rrIntervals.push(value.getUint16(offset, true) / 1024 * 1000); // Convert to ms
      offset += 2;
    }
  }

  return { hr, rrIntervals };
}
```

## OBD2 via BLE Pattern (ELM327)

### Connecting to ELM327 Adapter

```typescript
const ELM327_SERVICE = "0000fff0-0000-1000-8000-00805f9b34fb";
const ELM327_CHARACTERISTIC = "0000fff1-0000-1000-8000-00805f9b34fb";

async function connectOBD2() {
  const device = await navigator.bluetooth.requestDevice({
    filters: [{ services: [ELM327_SERVICE] }],
  });

  const server = await device.gatt!.connect();
  const service = await server.getPrimaryService(ELM327_SERVICE);
  const characteristic = await service.getCharacteristic(ELM327_CHARACTERISTIC);

  // Initialize ELM327
  await sendCommand(characteristic, "ATZ");   // Reset
  await sendCommand(characteristic, "ATE0");  // Echo off
  await sendCommand(characteristic, "ATL0");  // Linefeeds off
  await sendCommand(characteristic, "ATSP0"); // Auto-detect protocol

  return characteristic;
}
```

### Common OBD2 PIDs

| PID | Name | Formula | Unit |
|-----|------|---------|------|
| `010C` | Engine RPM | `(A*256+B)/4` | rpm |
| `010D` | Vehicle Speed | `A` | km/h |
| `0105` | Coolant Temp | `A-40` | °C |
| `010F` | Intake Air Temp | `A-40` | °C |
| `0104` | Engine Load | `A*100/255` | % |
| `0111` | Throttle Position | `A*100/255` | % |
| `010B` | MAP | `A` | kPa |
| `012F` | Fuel Level | `A*100/255` | % |
| `0106` | Short-term Fuel Trim | `(A-128)*100/128` | % |
| `0107` | Long-term Fuel Trim | `(A-128)*100/128` | % |
| `0110` | MAF Rate | `(A*256+B)/100` | g/s |
| `0146` | Ambient Temp | `A-40` | °C |
| `015E` | Fuel Rate | `(A*256+B)*0.05` | L/h |

## NewtonStreamManager Singleton

The core pattern for buffering sensor data and periodically querying Newton:

```typescript
class NewtonStreamManager {
  private static instance: NewtonStreamManager;
  private buffer: SensorReading[] = [];
  private sessionId: string | null = null;
  private focusFileIds: string[] = [];
  private queryInterval: NodeJS.Timeout | null = null;
  private listeners: Set<(result: any) => void> = new Set();

  static readonly MAX_BUFFER_SIZE = 300;
  static readonly MIN_DATA_POINTS = 32;
  static readonly QUERY_INTERVAL_MS = 15000;
  static readonly WINDOW_SIZE = 16;
  static readonly STEP_SIZE = 8;

  static getInstance(): NewtonStreamManager {
    if (!this.instance) this.instance = new NewtonStreamManager();
    return this.instance;
  }

  addData(reading: SensorReading) {
    this.buffer.push(reading);
    if (this.buffer.length > NewtonStreamManager.MAX_BUFFER_SIZE) {
      this.buffer = this.buffer.slice(-NewtonStreamManager.MAX_BUFFER_SIZE);
    }
  }

  async start() {
    await this.ensureSession();
    this.queryInterval = setInterval(
      () => this.queryIfReady(),
      NewtonStreamManager.QUERY_INTERVAL_MS
    );
  }

  stop() {
    if (this.queryInterval) clearInterval(this.queryInterval);
    this.destroySession();
  }

  subscribe(listener: (result: any) => void) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private async queryIfReady() {
    if (this.buffer.length < NewtonStreamManager.MIN_DATA_POINTS) return;
    try {
      const csv = this.bufferToCsv();
      const results = await this.queryMachineState(csv);
      this.listeners.forEach((fn) => fn(results));
    } catch (error) {
      this.sessionId = null; // Force session recreation on next query
    }
  }

  private bufferToCsv(): string {
    // Convert buffer to CSV with appropriate columns
    const headers = Object.keys(this.buffer[0]).join(",");
    const rows = this.buffer.map((r) => Object.values(r).join(","));
    return [headers, ...rows].join("\n");
  }
}
```

## HRV Feature Computation

When streaming RR intervals for HRV analysis, compute these features over a sliding window:

```typescript
function computeHRVFeatures(rrIntervals: number[]) {
  const n = rrIntervals.length;
  const diffs = rrIntervals.slice(1).map((rr, i) => rr - rrIntervals[i]);

  const mean_rr = rrIntervals.reduce((a, b) => a + b, 0) / n;
  const mean_hr = 60000 / mean_rr;

  const sdnn = Math.sqrt(
    rrIntervals.reduce((sum, rr) => sum + (rr - mean_rr) ** 2, 0) / (n - 1)
  );

  const rmssd = Math.sqrt(
    diffs.reduce((sum, d) => sum + d ** 2, 0) / diffs.length
  );

  const pnn50 = (diffs.filter((d) => Math.abs(d) > 50).length / diffs.length) * 100;

  const sd1 = Math.sqrt(0.5 * (rmssd ** 2));

  return { rmssd, sdnn, mean_hr, pnn50, sd1 };
}
```

## Data Flow: Server API Routes

### POST /api/newton/stream/data

Receives sensor readings from the browser and buffers them:

```typescript
export async function POST(request: Request) {
  const data = await request.json();
  const manager = NewtonStreamManager.getInstance();
  manager.addData(data);
  return Response.json({ buffered: true });
}
```

### POST /api/newton/stream/start

Starts the periodic query loop:

```typescript
export async function POST() {
  const manager = NewtonStreamManager.getInstance();
  await manager.start();
  return Response.json({ streaming: true });
}
```

### POST /api/newton/stream/stop

Stops streaming and cleans up:

```typescript
export async function POST() {
  const manager = NewtonStreamManager.getInstance();
  manager.stop();
  return Response.json({ streaming: false });
}
```

## Best Practices

- **Buffer server-side, not client-side** — the browser sends individual readings; the server batches into CSV
- **15-second query interval** balances latency vs. API load
- **MIN_DATA_POINTS guard** prevents querying with insufficient data
- **MAX_BUFFER_SIZE cap** prevents unbounded memory growth
- **Session reuse** — create once, query many times; only recreate on error
- **Graceful degradation** — sensor streaming should work independently of Newton availability
- **Demo mode** — implement simulated data for testing without hardware

## Building a frontend on top of this

If you're wrapping a sensor stream in a React/Svelte/etc. UI — live telemetry dashboard, BLE/OBD2 viewer, real-time replay tool — **read [`DESIGN.md`](../../DESIGN.md) at the root of this repo before writing any CSS**. The Archetype design system (Tailwind v4 + `@archetypeai/ds-lib-tokens` + Geist sans/mono + OKLCH palette + dark-first) is the expected visual language for these demos. [`archetypeai-wifi-demo`](https://github.com/archetypeai/archetypeai-wifi-demo), [`archetypeai-swat-demo`](https://github.com/archetypeai/archetypeai-swat-demo), and [`archetypeai-obd2-demo`](https://github.com/archetypeai/archetypeai-obd2-demo) are the reference implementations — pattern-match them for layout (Menubar, Card, Badge with `good` / `warning` / `critical` variants, mono numbers + readouts, sharp 2px radii, gauges + sparkline strips for live values). Setting this up at the start is much cheaper than retrofitting later.
