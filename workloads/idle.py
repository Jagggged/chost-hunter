"""Nearly idle demo workload for Chost Hunter presentations."""

import time


def main() -> None:
    """Keep the container alive while using almost no resources."""
    iteration = 0
    print("[demo-idle] starting nearly idle workload", flush=True)

    while True:
        iteration += 1
        print(f"[demo-idle] heartbeat={iteration}; sleeping", flush=True)
        time.sleep(30)


if __name__ == "__main__":
    main()
