# CPU 전용 이미지 (기본)
# GPU 사용 시: Dockerfile.gpu 참고
FROM python:3.11-slim

WORKDIR /app

# 시스템 패키지 (curl: 헬스체크용)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# 의존성 먼저 설치 (레이어 캐시 활용)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 복사
COPY . .

# HuggingFace 모델 캐시 위치 (볼륨 마운트 권장)
ENV HF_HOME=/app/models

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

ENTRYPOINT ["python", "server.py"]
CMD []
