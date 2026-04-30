"""
AI 에이전트 설정값
하이퍼파라미터, 임계값, 경로 등을 한곳에서 관리한다.
"""

# ── LSTM 모델 하이퍼파라미터 ──────────────────────────────
WINDOW_SIZE = 60          # 입력 시계열 길이 (과거 60 step)
PREDICT_HORIZON = 10      # 예측 길이 (미래 10 step)
N_FEATURES = 2            # 입력 피처 수 (CPU, Memory)
LSTM_UNITS = [64, 32]     # 2층 LSTM unit 수 (Lightweight)
DROPOUT_RATE = 0.2

# ── 학습 설정 ─────────────────────────────────────────────
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

# ── Online Learning (Fine-tuning) ────────────────────────
FINETUNE_INTERVAL_SEC = 3600   # 1시간마다 fine-tune
FINETUNE_EPOCHS = 5             # fine-tune 시 적은 epoch

# ── 추론 설정 ─────────────────────────────────────────────
INFERENCE_INTERVAL_SEC = 300    # 5분마다 추론

# ── 안전 장치 ─────────────────────────────────────────────
SAFETY_BUFFER = 0.30            # 예측값 위에 30% 버퍼
MIN_CPU_QUOTA = 0.1             # 최소 CPU (코어 단위)
MIN_MEMORY_BYTES = 64 * 1024 * 1024  # 최소 메모리 64MB

# ── Watchdog (비상 롤백) ──────────────────────────────────
WATCHDOG_INTERVAL_SEC = 1       # 1초마다 사용률 체크
WATCHDOG_THRESHOLD = 0.90       # 사용률 90% 초과 시 롤백

# ── 외부 연동 ─────────────────────────────────────────────
PROMETHEUS_URL = "http://localhost:9090"
DOCKER_SOCKET = "unix://var/run/docker.sock"

# ── 경로 ──────────────────────────────────────────────────
MODEL_DIR = "models"
PRETRAINED_MODEL_PATH = f"{MODEL_DIR}/pretrained.pt"
SCALER_PATH = f"{MODEL_DIR}/scaler.pkl"
