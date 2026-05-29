"""Mixed CPU and memory demo workload.

This workload combines duty-cycle CPU work with stepwise memory allocation so it
looks closer to a small service with changing traffic and working-set size.
"""

import gc
import math
import time


CYCLE_SECONDS = 80
SLICE_SECONDS = 1.0
MIN_DUTY_CYCLE = 0.08
MAX_DUTY_CYCLE = 0.45
MEMORY_STEPS_MB = [64, 128, 192, 256]
STEP_SECONDS = 15
RELEASE_SECONDS = 8
MB = 1024 * 1024


def duty_cycle(elapsed: float) -> float:
    """Return a smooth CPU duty cycle for the mixed workload."""
    wave = (math.sin((elapsed / CYCLE_SECONDS) * math.tau) + 1.0) / 2.0
    return MIN_DUTY_CYCLE + wave * (MAX_DUTY_CYCLE - MIN_DUTY_CYCLE)


def burn_cpu(duration: float) -> int:
    """Run a hot arithmetic loop for duration seconds."""
    deadline = time.perf_counter() + duration
    value = 1
    loops = 0
    while time.perf_counter() < deadline:
        value = ((value << 5) - value + loops) & 0xFFFFFFFF
        loops += 1
    return value


def allocate_mb(size_mb: int) -> list[bytearray]:
    """Allocate and touch size_mb megabytes in 1 MB chunks."""
    chunks = []
    for _ in range(size_mb):
        chunk = bytearray(MB)
        chunk[0] = 7
        chunk[-1] = 7
        chunks.append(chunk)
    return chunks


def run_step(size_mb: int, started_at: float) -> None:
    """Hold a memory step while generating changing CPU load."""
    allocation = allocate_mb(size_mb)
    step_deadline = time.perf_counter() + STEP_SECONDS
    print(f"[demo-mixed-wave] allocated={size_mb}MB; running CPU wave", flush=True)

    while time.perf_counter() < step_deadline:
        elapsed = time.perf_counter() - started_at
        duty = duty_cycle(elapsed)
        busy_seconds = SLICE_SECONDS * duty
        sleep_seconds = max(SLICE_SECONDS - busy_seconds, 0.0)
        marker = burn_cpu(busy_seconds)
        print(
            "[demo-mixed-wave] "
            f"mem={size_mb}MB duty={duty:.2f} "
            f"busy={busy_seconds:.2f}s sleep={sleep_seconds:.2f}s "
            f"marker={marker}",
            flush=True,
        )
        time.sleep(sleep_seconds)

    allocation.clear()


def main() -> None:
    """Repeat combined CPU and memory waves indefinitely."""
    started_at = time.perf_counter()
    print("[demo-mixed-wave] starting mixed wave workload", flush=True)

    while True:
        for size_mb in MEMORY_STEPS_MB:
            run_step(size_mb, started_at)

        gc.collect()
        print("[demo-mixed-wave] released allocation; cooling down", flush=True)
        time.sleep(RELEASE_SECONDS)


if __name__ == "__main__":
    main()
