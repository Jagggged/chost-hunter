"""
Lightweight LSTM 모델 정의
컨테이너 리소스 사용량의 시계열을 입력받아 미래 사용량을 예측한다.

Lightweight 설계 이유:
- 추론 자체가 시스템 부하가 되면 안 됨 (오버헤드 최소화)
- 2층 LSTM (64 -> 32 unit) + Dense
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from ai import config


def build_model(
    window_size: int = config.WINDOW_SIZE,
    n_features: int = config.N_FEATURES,
    horizon: int = config.PREDICT_HORIZON,
    units: list[int] = config.LSTM_UNITS,
    dropout: float = config.DROPOUT_RATE,
) -> Sequential:
    """
    LSTM 모델을 생성한다.

    Args:
        window_size: 입력 시계열 길이
        n_features: 입력 피처 수 (CPU, Memory 등)
        horizon: 출력 시계열 길이 (미래 예측 step 수)
        units: LSTM 레이어별 unit 수
        dropout: Dropout 비율

    Returns:
        컴파일되지 않은 Keras Sequential 모델
    """
    # TODO: 구현
    # model = Sequential([
    #     LSTM(units[0], return_sequences=True, input_shape=(window_size, n_features)),
    #     Dropout(dropout),
    #     LSTM(units[1]),
    #     Dropout(dropout),
    #     Dense(horizon * n_features),
    # ])
    raise NotImplementedError


def compile_model(model: Sequential, learning_rate: float = config.LEARNING_RATE):
    """
    모델 컴파일 (loss, optimizer 설정).
    """
    # TODO: 구현
    raise NotImplementedError
