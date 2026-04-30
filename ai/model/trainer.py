"""
LSTM 학습 / Fine-tuning 로직 (PyTorch)
- 사전 학습(Pre-training): 공개 데이터셋으로 처음부터 학습
- Fine-tuning(Online Learning): 운영 중 Prometheus 데이터로 모델 업데이트
"""

import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ai import config
from ai.model.lstm import LightweightLSTM, build_model, get_device


def _make_dataloader(
    X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool
) -> DataLoader:
    """NumPy 배열을 PyTorch DataLoader로 감싼다."""
    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).float()
    dataset = TensorDataset(X_t, y_t)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _train_one_epoch(
    model: LightweightLSTM,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """1 epoch 학습 후 평균 train loss 반환"""
    model.train()
    total_loss, n_batches = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def _evaluate(
    model: LightweightLSTM,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """검증셋 loss 계산"""
    model.eval()
    total_loss, n_batches = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb), yb)
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


def pretrain(
    X: np.ndarray,
    y: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
) -> LightweightLSTM:
    """
    공개 데이터셋으로 LSTM을 처음부터 학습한다.
    학습 결과(state_dict)는 models/pretrained.pt에 저장된다.

    Args:
        X, y: 학습 데이터 (samples, window/horizon, n_features)
        X_val, y_val: 검증 데이터 (선택)

    Returns:
        학습된 PyTorch 모델
    """
    device = get_device()
    model = build_model().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    train_loader = _make_dataloader(X, y, config.BATCH_SIZE, shuffle=True)
    val_loader = (
        _make_dataloader(X_val, y_val, config.BATCH_SIZE, shuffle=False)
        if X_val is not None and y_val is not None
        else None
    )

    print(f"[pretrain] device={device}, train_samples={len(X)}, "
          f"val_samples={len(X_val) if X_val is not None else 0}")

    for epoch in range(1, config.EPOCHS + 1):
        train_loss = _train_one_epoch(model, train_loader, criterion, optimizer, device)
        msg = f"[Epoch {epoch:02d}/{config.EPOCHS}] train_loss={train_loss:.6f}"
        if val_loader is not None:
            val_loss = _evaluate(model, val_loader, criterion, device)
            msg += f"  val_loss={val_loss:.6f}"
        print(msg)

    os.makedirs(os.path.dirname(config.PRETRAINED_MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), config.PRETRAINED_MODEL_PATH)
    print(f"[pretrain] saved to {config.PRETRAINED_MODEL_PATH}")
    return model


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
    device = get_device()
    model = build_model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    loader = _make_dataloader(X, y, config.BATCH_SIZE, shuffle=True)

    for epoch in range(1, config.FINETUNE_EPOCHS + 1):
        loss = _train_one_epoch(model, loader, criterion, optimizer, device)
        print(f"[finetune {epoch}/{config.FINETUNE_EPOCHS}] loss={loss:.6f}")

    torch.save(model.state_dict(), model_path)
    return model
