# engine

The core API of drunkenBot IDE.

## Structured dataset preparation

Malformed or schema-invalid JSON/JSONL records are grouped into one bounded
diagnostic per source file. Each manifest entry uses `record_diagnostics` with
the complete `invalid_record_count`, at most 12 compressed `location_ranges`,
an `omitted_location_count`, bounded `reason_counts`, and a human-readable
`summary`. Valid JSONL rows remain independent documents.

`DatasetBuildResult` and `dataset_summary.json` expose `partial_file_count`,
`failed_file_count`, `invalid_record_count`, and `preparation_outcome`.
Each affected source emits one bounded `event_type="dataset_diagnostic"` event
with `level`, per-source `outcome`, `source_path`, and the compact `diagnostic`
object.
Preparation emits exactly one terminal progress event with
`event_type="completion"` and `outcome="completed"` or
`"completed_with_warnings"`; the event repeats the partial, failed, and invalid
counts for UI consumers. A file is partial only when it produced usable data
alongside invalid records. Files with no usable records are failed.

## Standalone local training worker

Local training can run outside the desktop process, with no Qt dependency. Create
a `TrainingJobSpec`, wrap it with
`engine.training_worker_protocol.create_worker_request()`, atomically persist it with
`write_worker_request()`, and launch it with
`launch_worker_process(request_path)`. The equivalent stable command is:

```text
python -m engine.training_worker --request <absolute-request.json>
```

`launch_worker_process()` redirects stdout/stderr to the model output directory
and starts a new process session/process group. The UI may release the returned
`Popen` object and close without stopping training. It must not supervise the
worker with `QThread`. A crash-recoverable `training_worker.lock` claim prevents
two workers from writing the same output directory; worker exit code `2` means a
launch was rejected because an active or already-finished run identity exists.
Claim creation, stale recovery, and release are serialized by an OS-backed
mutex file whose lock is released automatically if an acquiring process exits;
if that mutex cannot be acquired, launch fails closed rather than unlinking a
claim whose ownership transition cannot be proven.
Each run has its own manifest/control directory, so a prior stop sentinel cannot
affect a later run.

### Request protocol (version 1)

```json
{
  "schema": "drunkenbot.training-worker-request",
  "version": 1,
  "run_id": "run_<durable-id>",
  "job": { "job_id": "...", "dataset": {}, "model": {}, "training": {}, "runtime": {}, "artifacts": {} },
  "paths": {
    "manifest": "<output>/training_runs/<run-id>/manifest.json",
    "control": "<output>/training_runs/<run-id>/control.json",
    "telemetry_db": "<output>/training_telemetry.sqlite"
  },
  "heartbeat_interval_seconds": 2.0,
  "telemetry": { "batch_size": 25, "flush_interval_seconds": 1.0 },
  "notifier": { "config_path": "<path>/notifier_config.json" }
}
```

`notifier` may be `null`. It references the existing configuration file; notifier
credentials are never copied into the run manifest. Unknown schema versions fail
before training starts.

### Atomic run manifest (version 1)

The worker atomically replaces the manifest on every state transition and
heartbeat:

```json
{
  "schema": "drunkenbot.training-run-manifest",
  "version": 1,
  "run_id": "run_<durable-id>",
  "status": "starting|running|stopping|stopped|completed|failed",
  "pid": 1234,
  "process_identity": { "kind": "psutil-create-time", "value": "..." },
  "started_at": "...",
  "updated_at": "...",
  "heartbeat_at": "...",
  "finished_at": null,
  "request_path": "...",
  "output_paths": {
    "output_dir": "...",
    "telemetry_db": "...",
    "checkpoint": null,
    "summary": null
  },
  "config_fingerprint": "<sha256>",
  "exit_code": null,
  "failure": null,
  "stop_requested_at": null
}
```

On failure, `failure` contains `type`, `message`, and `details_path`; the details
file contains the traceback. `manifest_is_stale()` checks both heartbeat age and
the OS process creation identity. A reattaching UI must call
`manifest_process_is_current()` before any force-stop action, and must fail
closed when identity cannot be verified. Checking PID alone is unsafe because of
PID reuse.

### Cooperative stop

Call `write_stop_request(control_path, run_id)`. It atomically writes:

```json
{
  "schema": "drunkenbot.training-worker-control",
  "version": 1,
  "run_id": "run_<durable-id>",
  "action": "stop",
  "requested_at": "..."
}
```

The training `should_stop` callback validates this sentinel. A mismatched run ID,
unknown version, or unsupported action is rejected rather than affecting another
process.

## Training telemetry

`TrainingConfig` controls wall-clock sampling:

| Setting | Default | Work |
| --- | ---: | --- |
| `telemetry_interval_seconds` | 1 | Loss, throughput, ETA, VRAM, CPU, RAM, callback |
| `stability_metrics_interval_seconds` | 15 | Gradient scalar conversion, full weight norm, update ratio |
| `preview_interval_seconds` | 30 | Device-to-host token copy and preview decode |

Lifecycle, warning, validation, checkpoint, stop, failure, and completion events
remain immediate. Unsampled optimizer steps perform no parameter traversal,
preview copy/decode, system sampling, scalar loss conversion, or progress
callback. Throughput uses the exact batches and tokens completed during the
wall-clock sample window; `average_step_seconds` normalizes that window by the
number of completed optimizer steps.

The worker owns one `TelemetryWriter` connection in WAL mode, batches metric
samples and ordinary lifecycle records, and force-flushes validation, checkpoint,
warning, stop, failure, and completion events. Coalescible samples remain in
`live_metrics`; durable events remain in `telemetry_events`.

Incremental reader APIs are:

```python
metric_rows_after(db_path, run_id, last_row_id=0, limit=1000)
event_rows_after(db_path, run_id, last_row_id=0, limit=1000)
```

Pass the last returned `id` on the next poll. `rows_after` remains an alias for
`metric_rows_after`; `latest_run`, `rows_until`, and `insert_metric` remain
available for compatibility.

## External notifications

`NotificationManager` caches parsed configuration and checks file mtime at a
bounded interval; `reload()` forces an immediate refresh. One background
dispatcher owns network delivery. Progress is throttled and coalesced by stage,
the queue is bounded, and pending progress is evicted before back-pressuring
rather than dropping completion or failure notifications. The standalone worker
owns the manager so delivery continues after the UI exits.

## Repeatable telemetry benchmark

The CPU-only harness compares always-on simulated expensive sampling with cadence
selection without claiming a GPU threshold:

```text
python -m engine.benchmarks.telemetry_cadence --steps 100000
```
