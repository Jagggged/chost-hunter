"""Memory wave demo workload.

The workload allocates memory in visible steps, holds it briefly, then releases
it and forces garbage collection so cAdvisor and Prometheus show a sawtooth.
"""

import gc
import time


MEMORY_STEPS_MB = [50, 100, 200, 300]
HOLD_SECONDS = 12
RELEASE_SECONDS = 10
MB = 1024 * 1024


def allocate_mb(size_mb: int) -> list[bytearray]:
    """Allocate memory in 1 MB chunks and touch each page."""
    chunks = []
    for _ in range(size_mb):
        chunk = bytearray(MB)
        chunk[0] = 1
        chunk[-1] = 1
        chunks.append(chunk)
    return chunks


def main() -> None:
    """Repeat a 50 MB -> 300 MB allocation and release pattern."""
    print("[demo-memory-wave] starting memory wave workload", flush=True)

    while True:
        allocation = []
        for size_mb in MEMORY_STEPS_MB:
            allocation = allocate_mb(size_mb)
            print(
                f"[demo-memory-wave] allocated={size_mb}MB; holding",
                flush=True,
            )
            time.sleep(HOLD_SECONDS)

        allocation.clear()
        gc.collect()
        print("[demo-memory-wave] released allocation; cooling down", flush=True)
        time.sleep(RELEASE_SECONDS)


if __name__ == "__main__":
    main()
