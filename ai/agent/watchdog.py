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
from ai.agent.controller import get_current_usage, update_limits


class Watchdog:
    """비상 롤백 감시 스레드"""

    def __init__(self):
        # 컨테이너별 직전 limit 보관 (롤백 시 사용)
        self._previous_limits: dict[str, dict] = {}
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def register(self, container_name: str, prev_limits: dict):
        """AI가 limit을 변경하기 직전에 호출하여 롤백 대상으로 등록"""
        # TODO: 구현
        raise NotImplementedError

    def start(self):
        """감시 스레드 시작"""
        # TODO: 구현
        # self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        # self._thread.start()
        raise NotImplementedError

    def stop(self):
        """감시 스레드 종료"""
        # TODO: 구현
        raise NotImplementedError

    def _watch_loop(self):
        """1초마다 모든 등록 컨테이너의 사용률 체크"""
        # TODO: 구현
        # while not self._stop_event.is_set():
        #     for name, prev in self._previous_limits.items():
        #         usage = get_current_usage(name)
        #         if usage["cpu_pct"] > config.WATCHDOG_THRESHOLD or \
        #            usage["mem_pct"] > config.WATCHDOG_THRESHOLD:
        #             update_limits(name, prev["cpu_quota"], prev["memory_bytes"])
        #             self._previous_limits.pop(name)
        #     time.sleep(config.WATCHDOG_INTERVAL_SEC)
        raise NotImplementedError
