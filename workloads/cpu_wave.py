"""CPU wave demo workload.

The workload alternates short busy-loop sections with sleeps. This keeps CPU
usage visibly moving in docker stats and Grafana without pinning a core at 100%.
"""

import math
import time


CYCLE_SECONDS = 60
SLICE_SECONDS = 1.0
MIN_DUTY_CYCLE = 0.10
MAX_DUTY_CYCLE = 0.50


def duty_cycle(elapsed: float) -> float:
    """Return a smooth duty cycle between MIN_DUTY_CYCLE and MAX_DUTY_CYCLE."""
    wave = (math.sin((elapsed / CYCLE_SECONDS) * math.tau) + 1.0) / 2.0
    return MIN_DUTY_CYCLE + wave * (MAX_DUTY_CYCLE - MIN_DUTY_CYCLE)


def burn_cpu(duration: float) -> int:
    """Spend CPU time doing deterministic arithmetic for the requested duration."""
    deadline = time.perf_counter() + duration
    value = 0
    loops = 0
    while time.perf_counter() < deadline:
        # Integer arithmetic avoids external dependencies and keeps the loop hot.
        value = (value * 1664525 + 1013904223 + loops) & 0xFFFFFFFF
        loops += 1
    return value


def main() -> None:
    """Generate a repeating 10% to 50% CPU usage wave."""
    start = time.perf_counter()
    print("[demo-cpu-wave] starting CPU wave workload", flush=True)

    while True:
        elapsed = time.perf_counter() - start
        duty = duty_cycle(elapsed)
        busy_seconds = SLICE_SECONDS * duty
        sleep_seconds = max(SLICE_SECONDS - busy_seconds, 0.0)

        marker = burn_cpu(busy_seconds)
        print(
            "[demo-cpu-wave] "
            f"duty={duty:.2f} busy={busy_seconds:.2f}s "
            f"sleep={sleep_seconds:.2f}s marker={marker}",
            flush=True,
        )
        time.sleep(sleep_seconds)


if __name__ == "__main__":
    main()
