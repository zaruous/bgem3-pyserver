import base64
import struct
from typing import List, Union

from pydantic import BaseModel, Field


class OpenAIEmbeddingRequest(BaseModel):
    input: Union[str, List[str]] = Field(..., description="Text or list of texts to embed (max 2048)")
    model: str = Field(default="BAAI/bge-m3", description="Model identifier (informational only)")
    encoding_format: str = Field(default="float", description="'float' or 'base64'")


class OpenAIEmbeddingObject(BaseModel):
    object: str = "embedding"
    index: int
    embedding: Union[List[float], str]  # str when encoding_format=base64


class OpenAIUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class OpenAIEmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[OpenAIEmbeddingObject]
    model: str
    usage: OpenAIUsage


def encode_base64(vec: List[float]) -> str:
    return base64.b64encode(struct.pack(f"<{len(vec)}f", *vec)).decode()
