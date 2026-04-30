# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Chost Hunter is an AIOps-based container resource optimization system. It detects "Ghost Containers" (idle/over-provisioned Docker containers) using an LSTM model trained on time-series resource usage, and dynamically adjusts `cpu_quota` / `mem_limit` via the Docker API.

The runtime splits into two layers that communicate only through Prometheus (no shared state, no in-process coupling):

1. **Monitoring layer** (`docker-compose.yml`, `prometheus/`, `grafana/`): cAdvisor scrapes container cgroup metrics → Prometheus stores them as time-series → Grafana visualizes. This layer runs as containers and is dataset-agnostic.
2. **AI agent layer** (`ai/`): A Python process that pulls metrics from Prometheus's HTTP API, runs LSTM inference, and issues `docker update` calls. It runs outside the compose stack.

Why this split matters: the agent never reads cgroup files directly. All resource observation goes through Prometheus, which means the agent works the same in dev (Docker Desktop) and prod (Linux), and historical data survives agent restarts.

## Common Commands

### Infrastructure (monitoring stack)
```bash
docker compose up -d           # Start cAdvisor (8080), Prometheus (9090), Grafana (3000, admin/admin)
docker compose down            # Stop the stack
docker compose logs -f cadvisor # Tail a specific service
```

On Docker Desktop (Windows/Mac), the **"Use containerd for pulling and storing images"** option must be **disabled** under Settings → General. Otherwise cAdvisor cannot enumerate individual containers and `name` labels disappear from metrics.

### AI agent (Python)
```bash
python -m venv .venv
.venv\Scripts\activate         # Windows
source .venv/bin/activate      # Linux/Mac
pip install -r requirements.txt

python -m ai.main              # Run the agent (uses absolute imports from repo root)
```

Run from the **repository root** — modules import as `from ai import config`, not relative.

### Testing the pipeline manually
```bash
# Spin up dummy containers to generate divergent load patterns
docker run -d --name test-busy busybox sh -c "while true; do :; done"
docker run -d --name test-idle busybox sh -c "sleep infinity"

# Verify metrics arrive in Prometheus (http://localhost:9090, Graph tab)
rate(container_cpu_usage_seconds_total{name=~"test.*"}[1m])

docker rm -f test-busy test-idle
```

## Architecture Notes for Code Changes

### `ai/config.py` is the single source of truth for hyperparameters
Anything tunable — window size, safety buffer, watchdog threshold, scrape URLs — lives here. Don't hardcode these values in modules; import from `ai.config`.

### Three safety layers prevent the AI from killing production
The agent must never reduce a container's resources past a point where it dies. Three independent mechanisms enforce this:

1. **Safety Buffer** (`SAFETY_BUFFER = 0.30`): Recommended limits are predicted-max × 1.3, never the raw prediction.
2. **Minimum Floor** (`MIN_CPU_QUOTA`, `MIN_MEMORY_BYTES`): Limits never drop below these values regardless of prediction.
3. **Watchdog** (`ai/agent/watchdog.py`): A separate thread polls Docker stats every 1s and rolls back to the previous limits if usage exceeds 90%. It bypasses Prometheus (which has 5s scrape latency) and reads from the Docker daemon directly for low-latency response.

When modifying `controller.update_limits()`, the **previous limits must be returned** so they can be registered with the watchdog before the new limits take effect. Order matters: register-then-update, never update-then-register.

### Dataset independence via `ai/data/loader.py`
The dataset (Bitbrain or otherwise) is intentionally not committed to. `loader.py` exposes `load_csv()`, `to_sliding_window()`, and `load_prometheus()` as the only entry points. Any new dataset is plugged in by implementing the CSV loader; the rest of the pipeline (`trainer.py`, `predictor.py`) consumes already-windowed NumPy arrays and is dataset-agnostic.

### LSTM model (PyTorch)
Defined as `LightweightLSTM(nn.Module)` in `ai/model/lstm.py`. Two-layer (64→32 units) by deliberate choice — inference happens in the control loop, so model overhead must stay below the time saved by optimization. Saved as a PyTorch `state_dict` to `models/pretrained.pt`, not a full model object, so the architecture lives in code and is versioned.

PyTorch was chosen over TensorFlow because the agent does **online fine-tuning** during operation (`trainer.finetune`) — PyTorch's dynamic graph makes the custom training loop easier to integrate with the watchdog and inference loop than Keras's `.fit()`.

### Branch workflow
- `main` — release-stable
- `develop` — integration branch; feature PRs target this
- `feat/*` — feature branches; `feat/infra-setup` and `feat/ai-setup` are the two seed branches

The infra and AI layers can be developed independently because they only share the Prometheus HTTP contract.

## Out-of-Scope (Do Not Generate)

- The `models/` directory holds binary weights — `.pt` and `.pkl` files are gitignored. Do not commit them.
- `datasets/` is gitignored for the same reason. Reference data sources by URL or loader code, not by checking in CSVs.
- The README's snapshot-based "zombie reclamation" workflow (CRIU, `docker commit`, .tar archives) is **out of scope** for the current direction. The project pivoted to live resource right-sizing via `docker update`. Treat README mentions of snapshot/freeze/quad-layer-backup as historical context only.
