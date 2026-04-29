"""
LSTM 학습 / Fine-tuning 로직
- 사전 학습(Pre-training): 공개 데이터셋(Bitbrain 등)으로 처음부터 학습
- Fine-tuning(Online Learning): 운영 중 Prometheus 데이터로 모델 업데이트
"""

import numpy as np
from tensorflow.keras.models import Sequential, load_model

from ai import config
from ai.model.lstm import build_model, compile_model


def pretrain(X: np.ndarray, y: np.ndarray) -> Sequential:
    """
    공개 데이터셋으로 LSTM을 처음부터 학습한다.
    학습 결과는 models/pretrained.h5에 저장된다.

    Args:
        X: 입력 시계열 (samples, window_size, n_features)
        y: 정답 시계열 (samples, horizon, n_features)

    Returns:
        학습된 Keras 모델
    """
    # TODO: 구현
    # model = build_model()
    # compile_model(model)
    # model.fit(X, y, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE,
    #           validation_split=config.VALIDATION_SPLIT)
    # model.save(config.PRETRAINED_MODEL_PATH)
    raise NotImplementedError


def finetune(model_path: str, X: np.ndarray, y: np.ndarray) -> Sequential:
    """
    저장된 모델을 운영 데이터로 Fine-tuning한다.
    적은 epoch만 돌려 컨테이너 개별 특성에 맞게 미세 조정한다.

    Args:
        model_path: 기존 모델 경로
        X, y: 새로운 운영 데이터

    Returns:
        Fine-tuning된 모델
    """
    # TODO: 구현
    # model = load_model(model_path)
    # model.fit(X, y, epochs=config.FINETUNE_EPOCHS, batch_size=config.BATCH_SIZE)
    # model.save(model_path)
    raise NotImplementedError
