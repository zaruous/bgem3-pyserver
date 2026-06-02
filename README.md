# BGE-M3 Embedding Server

[BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) 모델 기반의 임베딩 및 재순위화(reranking) API 서버입니다.  
FastAPI로 구현되었으며 Swagger UI와 OpenAI 호환 엔드포인트를 제공합니다.

---

## 원본 저작권 고지

이 프로젝트는 아래 원본 저장소를 기반으로 수정·개선되었습니다.

- **원본 저장소**: https://github.com/harshsavasil/bge-m3-server
- **원본 작성자**: harshsavasil

```
MIT License
Copyright (c) 2024 Tosone
```

원본 소스는 MIT 라이선스 하에 배포되었으며, 본 프로젝트도 동일한 라이선스를 따릅니다.

---

## 주요 기능

| 기능 | 설명 |
|---|---|
| **Dense 임베딩** | 문장당 1024차원 정규화 벡터 |
| **Sparse 임베딩** | 토큰 ID 기반 어휘 가중치 (BM25 스타일) |
| **ColBERT 임베딩** | 토큰 수만큼의 다중 벡터 |
| **Hybrid 임베딩** | Dense + Sparse 동시 반환 |
| **재순위화** | ColBERT + Sparse + Dense 융합 스코어링 |
| **OpenAI 호환** | `/v1/embeddings` 표준 포맷 지원 |
| **CLI 옵션** | 포트·모델·디바이스 등 실행 시 설정 |

---

## 프로젝트 구조

```
bge-m3-server/
├── server.py                   # FastAPI 앱 및 엔드포인트
├── enums/
│   └── embedding_type.py       # dense / sparse / colbert / hybrid
├── schemas/
│   ├── embedding_request.py    # POST /embedding 요청 스키마
│   ├── embedding_response.py   # POST /embedding 응답 스키마
│   ├── openai_embedding.py     # POST /v1/embeddings (OpenAI 호환)
│   ├── reranker_request.py     # POST /reranker 요청 스키마
│   └── reranker_response.py    # POST /reranker 응답 스키마
├── Dockerfile
├── pyproject.toml
└── README.md
```

---

## 요구 사항

- Python 3.11 이상
- [Poetry](https://python-poetry.org/) 또는 pip
- (선택) CUDA 지원 GPU

---

## 설치

### Poetry 사용

```sh
git clone https://github.com/harshsavasil/bge-m3-server
cd bge-m3-server
poetry install
```

### pip 사용

```sh
pip install fastapi "uvicorn[standard]" FlagEmbedding torch pydantic numpy
```

---

## 실행

### 기본 실행 (포트 3000, 디바이스 자동감지)

```sh
poetry run python server.py
```

### CLI 옵션 지정

```sh
# GPU + FP16, 포트 변경
python server.py --port 8080 --device cuda --fp16

# 로컬 모델 경로 지정
python server.py --model /models/bge-m3 --batch-size 32

# CPU 전용, 디버그 로그
python server.py --device cpu --log-level debug
```

### 전체 CLI 옵션

| 옵션 | 기본값 | 환경변수 | 설명 |
|---|---|---|---|
| `--host` | `0.0.0.0` | `HOST` | 바인딩 호스트 |
| `--port` | `3000` | `PORT` | 바인딩 포트 |
| `--model` | `BAAI/bge-m3` | `BGE_M3_MODEL_NAME` | 모델명 또는 로컬 경로 |
| `--device` | `auto` | `DEVICE` | `auto` / `cpu` / `cuda` / `mps` |
| `--batch-size` | `12` | `BGE_M3_BATCH_SIZE` | 인코딩 배치 크기 |
| `--max-length` | `8192` | `BGE_M3_MAX_LENGTH` | 임베딩/패시지 최대 토큰 수 |
| `--max-query-length` | `512` | `BGE_M3_MAX_QUERY_LENGTH` | 재순위화 쿼리 최대 토큰 수 |
| `--rerank-weights` | `0.4,0.2,0.4` | `BGE_M3_RERANKER_WEIGHTS` | 재순위화 가중치 (colbert,sparse,dense) |
| `--fp16` | off | `USE_FP16=true` | FP16 반정밀도 (CUDA 전용) |
| `--log-level` | `info` | `LOG_LEVEL` | `debug` / `info` / `warning` / `error` |

> `auto` 디바이스는 CUDA → MPS → CPU 순서로 자동 감지합니다.

---

## API 엔드포인트

서버 실행 후 Swagger UI: **http://localhost:3000/docs**  
ReDoc: **http://localhost:3000/redoc**

### GET /health

```json
{ "status": "ok" }
```

### GET /info

```json
{
  "model": "BAAI/bge-m3",
  "device": "cuda",
  "batch_size": 12,
  "max_length": 8192,
  "max_query_length": 512,
  "rerank_weights": [0.4, 0.2, 0.4]
}
```

### POST /v1/embeddings — OpenAI 호환

**요청**
```json
{
  "input": ["안녕하세요", "Hello world"],
  "model": "BAAI/bge-m3",
  "encoding_format": "float"
}
```

**응답**
```json
{
  "object": "list",
  "data": [
    { "object": "embedding", "index": 0, "embedding": [0.023, -0.041, ...] },
    { "object": "embedding", "index": 1, "embedding": [0.011,  0.093, ...] }
  ],
  "model": "BAAI/bge-m3",
  "usage": { "prompt_tokens": 6, "total_tokens": 6 }
}
```

OpenAI Python SDK와 직접 연동 가능합니다:

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:3000/v1",
    api_key="dummy",
)
resp = client.embeddings.create(model="BAAI/bge-m3", input=["안녕하세요"])
print(resp.data[0].embedding[:5])
```

### POST /embedding — 멀티모드 임베딩

**요청**
```json
{
  "sentences": ["문장 1", "문장 2"],
  "type": "hybrid"
}
```

`type` 값: `dense` (기본) / `sparse` / `colbert` / `hybrid`

**응답 (hybrid)**
```json
{
  "dense_embedding": [[...], [...]],
  "sparse_embedding": [{ "1024": 0.32, "3421": 0.18 }, ...],
  "colbert_embedding": null
}
```

| type | dense_embedding | sparse_embedding | colbert_embedding |
|---|---|---|---|
| `dense` | ✅ 1024차원 벡터 | — | — |
| `sparse` | — | ✅ {token_id: weight} | — |
| `colbert` | — | — | ✅ (토큰 수 × 1024) |
| `hybrid` | ✅ | ✅ | — |

### POST /reranker — 재순위화

**요청**
```json
{
  "target": "파이썬 비동기 프로그래밍",
  "sentences": ["asyncio 사용법", "자바스크립트 콜백", "Python async/await 예제"]
}
```

**응답**
```json
{ "scores": [0.821, 0.134, 0.876] }
```

점수는 입력 순서와 동일하며, ColBERT + Sparse + Dense 가중 합산 방식으로 계산됩니다.

---

## Docker

```sh
# 이미지 빌드
docker build -t bge-m3-server .

# CPU 실행
docker run -p 3000:3000 bge-m3-server

# GPU 실행
docker run --gpus all -p 3000:3000 bge-m3-server --device cuda --fp16

# 환경변수로 설정
docker run -p 8080:8080 -e PORT=8080 -e DEVICE=cpu bge-m3-server
```

---

## 모델 정보

| 항목 | 값 |
|---|---|
| 모델 | [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) |
| Dense 차원 | 1024 |
| 최대 토큰 수 | 8192 |
| 지원 언어 | 100개 이상 다국어 |
| 라이선스 | MIT |

모델은 첫 실행 시 HuggingFace Hub에서 자동 다운로드됩니다 (~2.3GB).  
이후 실행은 `~/.cache/huggingface/hub/` 캐시를 재사용합니다.

---

## 라이선스

MIT License — 자세한 내용은 [LICENSE](./LICENSE) 파일을 참고하세요.
