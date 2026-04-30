"""
AI 에이전트 진입점
사전학습 모델을 로드하고 추론 루프 + Watchdog을 실행한다.

실행 흐름:
1. Pretrained 모델 로드 + Watchdog 스레드 시작
2. 추론 루프 (INFERENCE_INTERVAL_SEC마다):
   - 대상 컨테이너 목록 조회
   - 각 컨테이너에 대해:
     a. Prometheus에서 최근 윈도우 조회 (정규화)
     b. LSTM 추론 -> 정규화 해제 -> 권고 limit 산출
     c. 이전 limit을 Watchdog에 등록 (롤백 대비)
     d. docker update로 limit 적용
3. (선택) FINETUNE_INTERVAL_SEC마다 Fine-tuning

운영 데이터로 윈도우를 가져오는 부분(fetch_recent_window)은
loader.load_prometheus 구현 이후 채워 넣는다.
"""

import time
import traceback

from ai import config
from ai.agent.controller import list_target_containers, update_limits
from ai.agent.predictor import (
    inverse_scale,
    load_pretrained,
    predict,
    recommend_limits,
)
from ai.agent.watchdog import Watchdog
from ai.data.loader import fetch_container_window


def run_inference_cycle(model, watchdog: Watchdog) -> None:
    """추론 1 사이클: 대상 컨테이너 전체 처리."""
    for name in list_target_containers():
        try:
            window = fetch_container_window(name)
            pred_scaled = predict(model, window)
            pred = inverse_scale(pred_scaled)
            limits = recommend_limits(pred)
            prev = update_limits(name, **limits)
            watchdog.register(name, prev)
            print(f"[infer] {name}: cpu={limits['cpu_quota']:.2f} "
                  f"mem={limits['memory_bytes']} (prev={prev})")
        except ValueError as e:
            # 데이터 부족 등 예상된 실패 - 다음 사이클에 다시 시도
            print(f"[infer] skipped {name}: {e}")
        except Exception:
            print(f"[infer] error for {name}:")
            traceback.print_exc()


def main():
    print("[boot] loading pretrained model...")
    model = load_pretrained()

    print("[boot] starting watchdog...")
    watchdog = Watchdog()
    watchdog.start()

    print(f"[boot] inference loop starts (every {config.INFERENCE_INTERVAL_SEC}s)")
    try:
        while True:
            run_inference_cycle(model, watchdog)
            time.sleep(config.INFERENCE_INTERVAL_SEC)
    except KeyboardInterrupt:
        print("[shutdown] interrupted")
    finally:
        watchdog.stop()


if __name__ == "__main__":
    main()
