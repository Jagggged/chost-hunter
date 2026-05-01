"""
LSTM 학습 / Fine-tuning 로직 (PyTorch)
- 사전 학습(Pre-training): 공개 데이터셋으로 처음부터 학습
- Fine-tuning(Online Learning): 운영 중 Prometheus 데이터로 모델 업데이트
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ai import config
from ai.model.lstm import LightweightLSTM, build_model, get_device


def _make_dataloader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool):
    """NumPy 배열을 PyTorch DataLoader로 감싼다"""
    # TODO: 구현
    # X_t = torch.from_numpy(X).float()
    # y_t = torch.from_numpy(y).float()
    # dataset = TensorDataset(X_t, y_t)
    # return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    raise NotImplementedError


def pretrain(X: np.ndarray, y: np.ndarray) -> LightweightLSTM:
    """
    공개 데이터셋으로 LSTM을 처음부터 학습한다.
    학습 결과는 models/pretrained.pt에 state_dict로 저장된다.

    Args:
        X: 입력 시계열 (samples, window_size, n_features)
        y: 정답 시계열 (samples, horizon, n_features)

    Returns:
        학습된 PyTorch 모델
    """
    # TODO: 구현
    # device = get_device()
    # model = build_model().to(device)
    # criterion = nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    # loader = _make_dataloader(X, y, config.BATCH_SIZE, shuffle=True)
    #
    # for epoch in range(config.EPOCHS):
    #     model.train()
    #     for xb, yb in loader:
    #         xb, yb = xb.to(device), yb.to(device)
    #         optimizer.zero_grad()
    #         pred = model(xb)
    #         loss = criterion(pred, yb)
    #         loss.backward()
    #         optimizer.step()
    #
    # torch.save(model.state_dict(), config.PRETRAINED_MODEL_PATH)
    # return model
    raise NotImplementedError


def finetune(model_path: str, X: np.ndarray, y: np.ndarray) -> LightweightLSTM:
    """
    저장된 모델을 운영 데이터로 Fine-tuning한다.
    적은 epoch만 돌려 컨테이너 개별 특성에 맞게 미세 조정한다.

    Args:
        model_path: 기존 state_dict 경로
        X, y: 새로운 운영 데이터

    Returns:
        Fine-tuning된 모델
    """
    # TODO: 구현
    # device = get_device()
    # model = build_model().to(device)
    # model.load_state_dict(torch.load(model_path, map_location=device))
    #
    # criterion = nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    # loader = _make_dataloader(X, y, config.BATCH_SIZE, shuffle=True)
    #
    # for epoch in range(config.FINETUNE_EPOCHS):
    #     model.train()
    #     for xb, yb in loader:
    #         xb, yb = xb.to(device), yb.to(device)
    #         optimizer.zero_grad()
    #         loss = criterion(model(xb), yb)
    #         loss.backward()
    #         optimizer.step()
    #
    # torch.save(model.state_dict(), model_path)
    # return model
    raise NotImplementedError
