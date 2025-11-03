# Python 3.9 slim 이미지 (용량↓)
FROM python:3.9-slim

# 런타임 기본 설정
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=5001 \
    WORKERS=2 \
    MODEL_DIR=/app/trained_model \
    GUNICORN_WORKER_CLASS=sync
    # FastAPI라면 빌드/런타임에 아래처럼 바꾸세요:
    # GUNICORN_WORKER_CLASS=uvicorn.workers.UvicornWorker

WORKDIR /app

# 시스템 의존성 (numpy/pandas 등 빌드 필요 시 주석 해제)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential curl \
#  && rm -rf /var/lib/apt/lists/*

# 필수 파일만 먼저 복사해 의존성 캐시 최적화
COPY requirements.txt ./

# 패키지 설치 (gunicorn이 requirements에 없으면 자동 설치)
RUN pip install --upgrade pip \
 && pip install -r requirements.txt || true \
 && pip install gunicorn

# 앱 및 모델/계약 파일 복사
COPY app.py ./app.py
COPY predict_contract.py ./predict_contract.py
COPY trained_model ./trained_model
# 선택: 예측 검증 샘플/피드백 데이터가 필요하면 포함
# COPY feedback_data.json ./feedback_data.json

# 비루트로 실행 (보안)
RUN useradd -m appuser
USER appuser

# 네트워크/헬스체크
EXPOSE ${PORT}

# 앱에 /health 라우트가 있다고 가정
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"5001\")}/health').read()" || exit 1

# Gunicorn 실행
# Flask: app:app  (현재와 동일)
# FastAPI면 env GUNICORN_WORKER_CLASS=uvicorn.workers.UvicornWorker 로 변경
CMD ["bash", "-lc", "exec gunicorn -w ${WORKERS} -k ${GUNICORN_WORKER_CLASS} -b 0.0.0.0:${PORT} --log-level info --access-logfile - --error-logfile - app:app"]
