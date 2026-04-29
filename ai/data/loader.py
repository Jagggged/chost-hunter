"""
데이터 로더 인터페이스
데이터셋(Bitbrain 등)이나 Prometheus에서 시계열을 가져와
LSTM 입력 형태(슬라이딩 윈도우)로 변환한다.

데이터 소스에 무관하게 동일한 출력 포맷을 보장하기 위해
이 모듈에서 슬라이딩 윈도우 변환을 통일한다.
"""

import numpy as np
import pandas as pd


def to_sliding_window(
    df: pd.DataFrame,
    window_size: int,
    horizon: int,
    feature_cols: list[str],
):
    """
    DataFrame을 LSTM 입력용 슬라이딩 윈도우로 변환한다.

    Args:
        df: 시계열 DataFrame (행: 시간, 열: 피처)
        window_size: 입력 윈도우 길이 (과거 N step)
        horizon: 예측 길이 (미래 M step)
        feature_cols: 사용할 피처 컬럼명 리스트

    Returns:
        X: shape (samples, window_size, n_features)
        y: shape (samples, horizon, n_features)
    """
    # TODO: 구현
    raise NotImplementedError


def load_csv(path: str) -> pd.DataFrame:
    """
    CSV 파일을 시계열 DataFrame으로 로드한다.
    데이터셋 확정 후 컬럼 매핑/전처리를 추가한다.
    """
    # TODO: 데이터셋 확정 후 구현
    raise NotImplementedError


def load_prometheus(query: str, start: str, end: str, step: str) -> pd.DataFrame:
    """
    Prometheus API에서 PromQL 쿼리 결과를 DataFrame으로 로드한다.
    Online Learning 시 운영 데이터를 가져올 때 사용한다.
    """
    # TODO: 구현 (requests로 /api/v1/query_range 호출)
    raise NotImplementedError
