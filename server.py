import argparse
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import List

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from FlagEmbedding import BGEM3FlagModel

from schemas.openai_embedding import (
    OpenAIEmbeddingObject,
    OpenAIEmbeddingRequest,
    OpenAIEmbeddingResponse,
    OpenAIUsage,
    encode_base64,
)

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BGE-M3 Embedding Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"), help="Bind host")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "3000")), help="Bind port")
    parser.add_argument(
        "--model",
        default=os.getenv("BGE_M3_MODEL_NAME", "BAAI/bge-m3"),
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--device",
        default=os.getenv("DEVICE", "auto"),
        choices=["auto", "cpu", "cuda", "mps"],
        help="Compute device (auto detects CUDA > MPS > CPU)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("BGE_M3_BATCH_SIZE", "12")),
        help="Encoding batch size",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=int(os.getenv("BGE_M3_MAX_LENGTH", "8192")),
        help="Maximum token length per input",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=os.getenv("USE_FP16", "").lower() in ("1", "true", "yes"),
        help="Use FP16 half-precision (CUDA only)",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "info"),
        choices=["critical", "error", "warning", "info", "debug"],
        help="Log verbosity",
    )
    return parser.parse_args()


class EmbeddingRunner:
    def __init__(self, model_name: str, device: str, batch_size: int, max_length: int, use_fp16: bool):
        self.model_name = model_name
        self.device = _resolve_device(device)
        self.batch_size = batch_size
        self.max_length = max_length

        effective_fp16 = use_fp16 and self.device == "cuda"

        logger.info("Loading model=%s device=%s fp16=%s", model_name, self.device, effective_fp16)

        self.model = BGEM3FlagModel(
            model_name,
            use_fp16=effective_fp16,
            devices=self.device,
        )
        logger.info("Model loaded successfully")

    def encode(self, texts: List[str]) -> List[List[float]]:
        output = self.model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        raw = output.get("dense_vecs")
        try:
            return raw.tolist()
        except AttributeError:
            return [list(v) for v in raw]

    def count_tokens(self, texts: List[str]) -> int:
        try:
            encoded = self.model.tokenizer(texts, truncation=False, padding=False)
            return sum(len(ids) for ids in encoded["input_ids"])
        except Exception:
            return sum(len(s.split()) for s in texts)


# ---------- app factory ----------

_runner: EmbeddingRunner | None = None


def create_app(args: argparse.Namespace) -> FastAPI:

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _runner
        _runner = EmbeddingRunner(
            model_name=args.model,
            device=args.device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            use_fp16=args.fp16,
        )
        yield
        _runner = None

    app = FastAPI(
        title="BGE-M3 Embedding Server",
        description=(
            "OpenAI-compatible text embedding API powered by "
            "[BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3).\n\n"
            "| Endpoint | Description |\n"
            "|---|---|\n"
            "| `POST /v1/embeddings` | Generate dense embeddings |\n"
            "| `GET /health` | Liveness probe |\n"
            "| `GET /info` | Model info |"
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def get_runner() -> EmbeddingRunner:
        if _runner is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Model is not ready")
        return _runner

    @app.get("/health", tags=["System"], summary="Liveness probe")
    async def health():
        """Returns `{"status": "ok"}` when the model is loaded and ready."""
        get_runner()
        return {"status": "ok"}

    @app.get("/info", tags=["System"], summary="Model info")
    async def info():
        """Returns the active model name, device, and runtime settings."""
        r = get_runner()
        return {
            "model": r.model_name,
            "device": r.device,
            "batch_size": r.batch_size,
            "max_length": r.max_length,
            "embedding_dim": 1024,
        }

    @app.post(
        "/v1/embeddings",
        response_model=OpenAIEmbeddingResponse,
        tags=["Embedding"],
        summary="Create embeddings (OpenAI-compatible)",
    )
    async def create_embeddings(request: OpenAIEmbeddingRequest):
        """
        Drop-in replacement for OpenAI `/v1/embeddings`.

        - **input**: single string or list of strings (max 2048)
        - **encoding_format**: `float` (default) or `base64`

        Returns 1024-dim dense vectors.
        """
        if request.encoding_format not in ("float", "base64"):
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "encoding_format must be 'float' or 'base64'")

        r = get_runner()
        texts = [request.input] if isinstance(request.input, str) else list(request.input)

        if len(texts) > 2048:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Maximum 2048 inputs per request")

        dense_list = r.encode(texts)
        token_count = r.count_tokens(texts)

        data = [
            OpenAIEmbeddingObject(
                index=i,
                embedding=encode_base64(vec) if request.encoding_format == "base64" else vec,
            )
            for i, vec in enumerate(dense_list)
        ]

        return OpenAIEmbeddingResponse(
            data=data,
            model=r.model_name,
            usage=OpenAIUsage(prompt_tokens=token_count, total_tokens=token_count),
        )

    return app


def main():
    args = _parse_args()
    logging.getLogger().setLevel(args.log_level.upper())
    app = create_app(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
