"""
AI 에이전트 진입점
사전학습 모델을 로드하고 추론 루프 + Watchdog을 실행한다.

실행 흐름:
1. Pretrained 모델 로드
2. Watchdog 스레드 시작
3. 추론 루프:
   - 5분마다 Prometheus에서 최근 데이터 조회
   - 모델 추론 -> 권고 limit 산출
   - Watchdog에 이전 limit 등록 -> 컨테이너 limit 갱신
4. 1시간마다 Fine-tuning
"""

import time

from ai import config


def main():
    # TODO: 구현
    # 1. model = load_model(config.PRETRAINED_MODEL_PATH)
    # 2. watchdog = Watchdog(); watchdog.start()
    # 3. while True:
    #        for container in target_containers:
    #            window = fetch_recent_window(container)
    #            pred = predict(model, window)
    #            limits = recommend_limits(pred)
    #            prev = update_limits(container, **limits)
    #            watchdog.register(container, prev)
    #        time.sleep(config.INFERENCE_INTERVAL_SEC)
    raise NotImplementedError


if __name__ == "__main__":
    main()
