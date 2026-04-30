"""
Watchdog: 비상 롤백 메커니즘

AI가 자원을 줄였는데 갑자기 부하가 튀면(Spike) 컨테이너가 죽을 수 있다.
별도 스레드로 1초마다 사용률을 감시하다가 임계치(90%) 초과 시
AI 판단을 무시하고 즉시 이전 limit으로 복구한다.

Prometheus 경유는 5초 지연이 있어 부적합. Docker stats로 직접 읽는다.
"""

import threading
import time

from ai import config
from ai.agent.controller import get_current_usage, rollback_limits


class Watchdog:
    """비상 롤백 감시 스레드"""

    def __init__(self):
        # 컨테이너별 직전 limit 보관 (롤백 시 사용)
        self._previous_limits: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def register(self, container_name: str, prev_limits: dict) -> None:
        """AI가 limit을 변경하기 직전에 호출하여 롤백 대상으로 등록한다."""
        with self._lock:
            self._previous_limits[container_name] = prev_limits

    def unregister(self, container_name: str) -> None:
        """롤백 후 또는 컨테이너 제거 시 감시 대상에서 빼낸다."""
        with self._lock:
            self._previous_limits.pop(container_name, None)

    def start(self) -> None:
        """감시 스레드 시작 (이미 돌고 있으면 무시)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """감시 스레드 종료를 요청한다 (다음 sleep 후 종료)."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _watch_loop(self) -> None:
        """주기적으로 등록된 컨테이너의 사용률을 체크하고 임계치 초과 시 롤백."""
        while not self._stop_event.is_set():
            # 반복 중 register/unregister와 충돌하지 않도록 스냅샷
            with self._lock:
                snapshot = dict(self._previous_limits)

            for name, prev in snapshot.items():
                try:
                    usage = get_current_usage(name)
                except Exception as e:
                    # 컨테이너가 사라졌거나 일시적 실패 - 다음 사이클에 다시 시도
                    print(f"[watchdog] usage error for {name}: {e}")
                    continue

                if (usage["cpu_pct"] > config.WATCHDOG_THRESHOLD
                        or usage["mem_pct"] > config.WATCHDOG_THRESHOLD):
                    print(f"[watchdog] ROLLBACK {name}: cpu={usage['cpu_pct']:.2f} "
                          f"mem={usage['mem_pct']:.2f}")
                    try:
                        rollback_limits(name, prev)
                    except Exception as e:
                        print(f"[watchdog] rollback error for {name}: {e}")
                    finally:
                        # 한 번 롤백한 컨테이너는 다시 AI가 limit을 정해야 하므로 등록 해제
                        self.unregister(name)

            time.sleep(config.WATCHDOG_INTERVAL_SEC)
