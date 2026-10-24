"""Small, testable wrapper around the OpenAI embeddings API."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)


class EmbeddingError(RuntimeError):
    """Raised when embeddings cannot be generated or validated."""


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    """Embedding vectors and usage returned for a collection of texts."""

    vectors: tuple[tuple[float, ...], ...]
    prompt_tokens: int

    @property
    def dimension(self) -> int:
        return len(self.vectors[0]) if self.vectors else 0


class EmbeddingProvider(Protocol):
    """Interface used by the ingestion pipeline and fake test providers."""

    def embed_texts(self, texts: Sequence[str]) -> EmbeddingResult: ...


class OpenAIEmbeddingClient:
    """Generate embeddings in validated batches with bounded retries."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        batch_size: int,
        max_retries: int = 3,
        initial_retry_delay: float = 0.5,
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key.strip():
            raise ValueError("api_key must not be empty")
        if not model.strip():
            raise ValueError("model must not be empty")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")
        if initial_retry_delay < 0:
            raise ValueError("initial_retry_delay must not be negative")

        self._model = model
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._initial_retry_delay = initial_retry_delay
        self._sleep = sleep
        self._client = client or OpenAI(
            api_key=api_key,
            max_retries=0,
            timeout=30.0,
        )

    def embed_texts(self, texts: Sequence[str]) -> EmbeddingResult:
        """Embed texts in order and return total prompt-token usage."""

        inputs = list(texts)
        if any(not text.strip() for text in inputs):
            raise EmbeddingError("Embedding input must not be empty")
        if not inputs:
            return EmbeddingResult(vectors=(), prompt_tokens=0)

        vectors: list[tuple[float, ...]] = []
        prompt_tokens = 0
        expected_dimension: int | None = None

        for start in range(0, len(inputs), self._batch_size):
            batch = inputs[start : start + self._batch_size]
            response = self._request_batch(batch)
            batch_vectors, batch_tokens = _read_response(response, len(batch))
            batch_dimension = len(batch_vectors[0])
            if expected_dimension is None:
                expected_dimension = batch_dimension
            elif batch_dimension != expected_dimension:
                raise EmbeddingError("Embedding dimensions changed between batches")
            vectors.extend(batch_vectors)
            prompt_tokens += batch_tokens

        return EmbeddingResult(vectors=tuple(vectors), prompt_tokens=prompt_tokens)

    def _request_batch(self, batch: list[str]) -> object:
        for attempt in range(self._max_retries + 1):
            try:
                return self._client.embeddings.create(
                    model=self._model,
                    input=batch,
                    encoding_format="float",
                )
            except (
                APIConnectionError,
                APITimeoutError,
                RateLimitError,
                InternalServerError,
            ) as exc:
                if attempt == self._max_retries:
                    raise EmbeddingError(
                        f"Embedding request failed after {attempt + 1} attempts"
                    ) from exc
                self._sleep(self._initial_retry_delay * (2**attempt))
            except APIError as exc:
                raise EmbeddingError("Embedding request was rejected") from exc
        raise AssertionError("Retry loop ended unexpectedly")


def _read_response(response: Any, expected_count: int) -> tuple[list[tuple[float, ...]], int]:
    try:
        data = list(response.data)
        prompt_tokens = int(response.usage.prompt_tokens)
    except (AttributeError, TypeError, ValueError) as exc:
        raise EmbeddingError("Embedding response is missing required fields") from exc

    indexes = [getattr(item, "index", None) for item in data]
    if any(type(index) is not int for index in indexes) or sorted(indexes) != list(
        range(expected_count)
    ):
        raise EmbeddingError("Embedding response indexes do not match the request")
    if prompt_tokens < 0:
        raise EmbeddingError("Embedding response contains invalid token usage")

    ordered_data = sorted(data, key=lambda item: item.index)
    vectors: list[tuple[float, ...]] = []
    dimension: int | None = None
    for item in ordered_data:
        try:
            vector = tuple(float(value) for value in item.embedding)
        except (AttributeError, TypeError, ValueError) as exc:
            raise EmbeddingError("Embedding response contains an invalid vector") from exc
        if not vector or not all(math.isfinite(value) for value in vector):
            raise EmbeddingError("Embedding response contains an invalid vector")
        if dimension is None:
            dimension = len(vector)
        elif len(vector) != dimension:
            raise EmbeddingError("Embedding response contains inconsistent dimensions")
        vectors.append(vector)
    return vectors, prompt_tokens
