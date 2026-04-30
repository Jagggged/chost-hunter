"""
추론 루프 (PyTorch)
주기적으로 Prometheus에서 최근 데이터를 가져와 모델에 추론을 돌리고
권고값(예상 사용량 + 안전 버퍼)을 산출한다.
"""

import numpy as np
import torch

from ai import config
from ai.model.lstm import LightweightLSTM, build_model, get_device


def load_pretrained(model_path: str = config.PRETRAINED_MODEL_PATH) -> LightweightLSTM:
    """
    저장된 state_dict를 로드해 추론용 모델을 만든다.
    eval 모드로 전환하여 Dropout 등을 비활성화한다.
    """
    # TODO: 구현
    # device = get_device()
    # model = build_model().to(device)
    # model.load_state_dict(torch.load(model_path, map_location=device))
    # model.eval()
    # return model
    raise NotImplementedError


def predict(model: LightweightLSTM, recent_window: np.ndarray) -> np.ndarray:
    """
    최근 시계열을 받아 미래 사용량을 예측한다.

    Args:
        model: 로드된 LSTM 모델 (eval 모드)
        recent_window: shape (1, window_size, n_features)

    Returns:
        예측값 (horizon, n_features)
    """
    # TODO: 구현
    # device = get_device()
    # x = torch.from_numpy(recent_window).float().to(device)
    # with torch.no_grad():
    #     pred = model(x)
    # return pred.cpu().numpy()[0]   # batch 차원 제거
    raise NotImplementedError


def recommend_limits(prediction: np.ndarray) -> dict:
    """
    예측값을 바탕으로 컨테이너 리소스 권고값을 계산한다.
    안전 버퍼와 Minimum Floor를 적용한다.

    Returns:
        {"cpu_quota": float, "memory_bytes": int}
    """
    # TODO: 구현
    # max_pred = prediction.max(axis=0)
    # cpu = max(max_pred[0] * (1 + config.SAFETY_BUFFER), config.MIN_CPU_QUOTA)
    # mem = max(max_pred[1] * (1 + config.SAFETY_BUFFER), config.MIN_MEMORY_BYTES)
    raise NotImplementedError
