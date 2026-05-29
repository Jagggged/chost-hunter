# Demo Workloads

This guide describes the Docker workloads used to demonstrate Chost Hunter in a
presentation environment. They generate visible CPU and memory changes without
depending on `stress-ng` or fixed 100% load.

## Containers

| Container | Policy | Purpose |
|---|---|---|
| `demo-idle` | `advisory` | Baseline idle container that mostly sleeps. |
| `demo-cpu-wave` | `advisory` | CPU duty-cycle workload that moves between roughly 10% and 50%. |
| `demo-memory-wave` | `advisory` | Stepwise memory allocation from 50 MB to 300 MB, then release. |
| `demo-memory-leak` | `auto` | Capped leak pattern that adds 50 MB every 30 seconds up to 700 MB and does not release it. |
| `demo-mixed-wave` | `auto` | Combined CPU and memory wave that Chost Hunter can manage automatically. |

All containers use the `python:3.11-slim` image and mount scripts from
`./workloads` into `/app`.

## Run

Start the normal stack with the demo overlay:

```bash
docker compose -f docker-compose.yml -f docker-compose.demo.yml up -d
```

Confirm the demo containers are running:

```bash
docker compose -f docker-compose.yml -f docker-compose.demo.yml ps
```

Tail workload logs:

```bash
docker logs -f demo-cpu-wave
docker logs -f demo-memory-wave
docker logs -f demo-memory-leak
docker logs -f demo-mixed-wave
```

Check only the memory leak workload:

```bash
docker stats demo-memory-leak
docker logs demo-memory-leak --tail=30
```

## Stop

Stop only the demo workloads:

```bash
docker compose -f docker-compose.yml -f docker-compose.demo.yml stop \
  demo-idle demo-cpu-wave demo-memory-wave demo-memory-leak demo-mixed-wave
```

Remove only the demo workload containers:

```bash
docker compose -f docker-compose.yml -f docker-compose.demo.yml rm -f \
  demo-idle demo-cpu-wave demo-memory-wave demo-memory-leak demo-mixed-wave
```

Stop the full stack:

```bash
docker compose -f docker-compose.yml -f docker-compose.demo.yml down
```

## Check With Docker Stats

Watch live usage:

```bash
docker stats demo-idle demo-cpu-wave demo-memory-wave demo-memory-leak demo-mixed-wave
```

Expected behavior:

| Container | Expected signal |
|---|---|
| `demo-idle` | CPU near zero, low stable memory. |
| `demo-cpu-wave` | CPU rises and falls repeatedly. |
| `demo-memory-wave` | Memory climbs through steps, then drops after release. |
| `demo-memory-leak` | Memory climbs in 50 MB steps and then stays high. |
| `demo-mixed-wave` | CPU and memory both change over time. |

`demo-memory-wave` is the normal pattern: memory grows during work and is then
released. `demo-memory-leak` is the abnormal pattern: memory is never released,
so the graph should move upward until it reaches the capped 700 MB plateau.

## Grafana

Open Grafana:

```text
http://localhost:3000
```

Default credentials:

```text
admin / admin
```

Check these graphs in the provisioned dashboard or with Prometheus-backed
panels:

| Graph | What to look for |
|---|---|
| Container CPU usage | `demo-cpu-wave` and `demo-mixed-wave` should show repeating waves. |
| Container memory usage | `demo-memory-wave` should step up and drop, while `demo-memory-leak` should step upward and hold. |
| Chost Hunter AI panel | Demo containers should appear with `advisory` or `auto` policy. |
| Recent actions | Recommendations should appear after enough Prometheus samples exist. |

For the leak demo, select `demo-memory-leak` in Container Overview. The
`Memory Usage per Container` panel is successful when it shows a staircase-like
upward trend that eventually flattens near the cap.

## Presentation Flow

1. Start the stack with the demo overlay.
2. Show `demo-idle` as the idle baseline container.
3. Show `demo-cpu-wave` as the CPU variation container.
4. Show `demo-memory-wave` as the normal memory allocation and release pattern.
5. Show `demo-memory-leak` as the abnormal memory leak pattern.
6. Show `demo-mixed-wave` as the combined CPU and memory workload.
7. Open Prometheus or Grafana and filter for `demo-*` containers.
8. Explain the labels:
   - `demo-idle`, `demo-cpu-wave`, and `demo-memory-wave` are advisory-only.
   - `demo-memory-leak` and `demo-mixed-wave` use `chost-hunter.policy=auto`.
9. Wait for Prometheus to collect enough samples for the AI Agent window.
10. Show Chost Hunter recommendations and action history in Grafana.
11. Explain that the workloads are intentionally shaped for presentation, not fixed max load.

## Success Criteria

The demo is working when:

| Check | Success condition |
|---|---|
| Containers | All five `demo-*` containers are running. |
| Logs | Each workload prints periodic state logs. |
| Docker stats | CPU and memory values visibly change over time. |
| Prometheus | cAdvisor metrics include `name="demo-..."` series. |
| Grafana | Demo containers appear in CPU and memory panels. |
| Chost Hunter | AI Agent lists the demo containers as managed targets. |

## Troubleshooting

### `python:3.11-slim` pull timeout

Retry the pull before starting the stack:

```bash
docker pull python:3.11-slim
docker compose -f docker-compose.yml -f docker-compose.demo.yml up -d
```

If the network is unreliable, pre-pull the image before the presentation.

### CPU is too low

Increase duty cycle or work duration in:

```text
workloads/cpu_wave.py
workloads/mixed_wave.py
```

Useful changes:

| Setting | Effect |
|---|---|
| Increase `MAX_DUTY_CYCLE` | Raises peak CPU usage. |
| Increase `MIN_DUTY_CYCLE` | Raises baseline CPU usage. |
| Increase `SLICE_SECONDS` | Makes each busy/sleep segment longer. |

Restart the affected containers after editing:

```bash
docker compose -f docker-compose.yml -f docker-compose.demo.yml restart \
  demo-cpu-wave demo-mixed-wave
```

### CPU is too high

Lower duty cycle or increase sleep time in:

```text
workloads/cpu_wave.py
workloads/mixed_wave.py
```

Useful changes:

| Setting | Effect |
|---|---|
| Decrease `MAX_DUTY_CYCLE` | Lowers peak CPU usage. |
| Decrease `MIN_DUTY_CYCLE` | Lowers baseline CPU usage. |
| Reduce `SLICE_SECONDS` | Makes the loop react faster to duty-cycle changes. |

Then restart the workload containers.

### Grafana does not show demo containers

Check that the demo containers are running:

```bash
docker ps --filter "name=demo-"
```

Check cAdvisor directly:

```bash
curl "http://localhost:8080/metrics" | grep 'name="demo-'
```

Check Prometheus targets:

```text
http://localhost:9090/targets
```

Run a Prometheus query:

```promql
rate(container_cpu_usage_seconds_total{name=~"demo-.*"}[1m])
```

If the query is empty, restart cAdvisor and Prometheus:

```bash
docker compose restart cadvisor prometheus
```

On Docker Desktop, make sure the containerd image store option is disabled if
cAdvisor does not expose container `name` labels.
