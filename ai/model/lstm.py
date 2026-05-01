"""
Lightweight LSTM 모델 정의 (PyTorch)
컨테이너 리소스 사용량의 시계열을 입력받아 미래 사용량을 예측한다.

Lightweight 설계 이유:
- 추론 자체가 시스템 부하가 되면 안 됨 (오버헤드 최소화)
- 2층 LSTM (64 -> 32 unit) + Linear

PyTorch를 선택한 이유:
- 동적 그래프로 Online Learning 시 학습 루프 제어가 유연함
- Watchdog 등 커스텀 로직과 통합이 용이함
"""

import torch
import torch.nn as nn

from ai import config


class LightweightLSTM(nn.Module):
    """
    2층 LSTM + Linear 출력
    Input:  (batch, window_size, n_features)
    Output: (batch, horizon, n_features)
    """

    def __init__(
        self,
        n_features: int = config.N_FEATURES,
        horizon: int = config.PREDICT_HORIZON,
        units: list[int] = config.LSTM_UNITS,
        dropout: float = config.DROPOUT_RATE,
    ):
        super().__init__()
        self.n_features = n_features
        self.horizon = horizon

        # TODO: 구현
        # self.lstm1 = nn.LSTM(input_size=n_features, hidden_size=units[0],
        #                      batch_first=True)
        # self.dropout1 = nn.Dropout(dropout)
        # self.lstm2 = nn.LSTM(input_size=units[0], hidden_size=units[1],
        #                      batch_first=True)
        # self.dropout2 = nn.Dropout(dropout)
        # self.fc = nn.Linear(units[1], horizon * n_features)
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, window_size, n_features)
        Returns:
            (batch, horizon, n_features)
        """
        # TODO: 구현
        # out, _ = self.lstm1(x)
        # out = self.dropout1(out)
        # _, (h, _) = self.lstm2(out)          # 마지막 hidden state
        # out = self.dropout2(h[-1])            # (batch, units[1])
        # out = self.fc(out)                    # (batch, horizon * n_features)
        # return out.view(-1, self.horizon, self.n_features)
        raise NotImplementedError


def build_model() -> LightweightLSTM:
    """모델 인스턴스 생성"""
    # TODO: 구현
    # return LightweightLSTM()
    raise NotImplementedError


def get_device() -> torch.device:
    """GPU 사용 가능하면 cuda, 아니면 cpu"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
