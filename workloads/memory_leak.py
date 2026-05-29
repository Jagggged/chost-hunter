"""Memory leak style demo workload.

This intentionally keeps allocated bytearrays alive so Grafana shows a steady
upward memory trend. The maximum is capped for macOS Docker Desktop demos.
"""

import time


STEP_MB = 50
MAX_MB = 700
ALLOC_INTERVAL_SECONDS = 30
HOLD_LOG_INTERVAL_SECONDS = 30
MB = 1024 * 1024


def allocate_mb(size_mb: int) -> list[bytearray]:
    """Allocate size_mb megabytes and touch each chunk so it is committed."""
    chunks = []
    for _ in range(size_mb):
        chunk = bytearray(MB)
        chunk[0] = 3
        chunk[-1] = 3
        chunks.append(chunk)
    return chunks


def main() -> None:
    """Grow memory usage to MAX_MB and then hold it forever."""
    allocations = []
    allocated_mb = 0
    print("[demo-memory-leak] starting capped memory leak workload", flush=True)

    while allocated_mb < MAX_MB:
        next_step_mb = min(STEP_MB, MAX_MB - allocated_mb)
        allocations.extend(allocate_mb(next_step_mb))
        allocated_mb += next_step_mb
        print(f"[demo-memory-leak] allocated={allocated_mb}MB", flush=True)
        time.sleep(ALLOC_INTERVAL_SECONDS)

    print("[demo-memory-leak] reached max allocation; holding", flush=True)
    while True:
        # Keep the list referenced so memory is not released.
        print(f"[demo-memory-leak] holding allocated={allocated_mb}MB", flush=True)
        time.sleep(HOLD_LOG_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
