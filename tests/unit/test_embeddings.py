from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from types import SimpleNamespace

import httpx2
import pytest
from openai import APIConnectionError

from litellm_docs_mcp.embeddings import EmbeddingError, OpenAIEmbeddingClient


@dataclass(frozen=True)
class _FakeEmbedding:
    index: int
    embedding: list[float]


class _FakeEmbeddingsEndpoint:
    def __init__(self, responses: Sequence[object]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def create(self, **arguments: object) -> object:
        self.calls.append(arguments)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class _FakeClient:
    def __init__(self, responses: Sequence[object]) -> None:
        self.embeddings = _FakeEmbeddingsEndpoint(responses)


def _response(vectors: list[list[float]], prompt_tokens: int = 10) -> object:
    return SimpleNamespace(
        data=[
            _FakeEmbedding(index=index, embedding=vector) for index, vector in enumerate(vectors)
        ],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens),
    )


def _client(
    fake_client: _FakeClient,
    *,
    batch_size: int = 2,
    max_retries: int = 3,
    sleep=lambda _delay: None,
) -> OpenAIEmbeddingClient:
    return OpenAIEmbeddingClient(
        api_key="test-key",
        model="text-embedding-3-small",
        batch_size=batch_size,
        max_retries=max_retries,
        initial_retry_delay=0.25,
        client=fake_client,
        sleep=sleep,
    )


def test_embeddings_are_batched_and_returned_in_input_order() -> None:
    fake_client = _FakeClient(
        [
            _response([[1.0, 0.0], [0.0, 1.0]], prompt_tokens=4),
            _response([[0.5, 0.5]], prompt_tokens=2),
        ]
    )

    result = _client(fake_client).embed_texts(["one", "two", "three"])

    assert result.vectors == ((1.0, 0.0), (0.0, 1.0), (0.5, 0.5))
    assert result.dimension == 2
    assert result.prompt_tokens == 6
    assert [call["input"] for call in fake_client.embeddings.calls] == [
        ["one", "two"],
        ["three"],
    ]
    assert all(call["model"] == "text-embedding-3-small" for call in fake_client.embeddings.calls)
    assert all(call["encoding_format"] == "float" for call in fake_client.embeddings.calls)


def test_transient_error_is_retried_with_bounded_backoff() -> None:
    connection_error = APIConnectionError(
        message="temporary failure",
        request=httpx2.Request("POST", "https://api.openai.com/v1/embeddings"),
    )
    fake_client = _FakeClient([connection_error, _response([[1.0, 2.0]])])
    delays: list[float] = []

    result = _client(fake_client, sleep=delays.append).embed_texts(["retry me"])

    assert result.vectors == ((1.0, 2.0),)
    assert delays == [0.25]
    assert len(fake_client.embeddings.calls) == 2


def test_transient_error_stops_after_retry_limit() -> None:
    errors = [
        APIConnectionError(
            message="temporary failure",
            request=httpx2.Request("POST", "https://api.openai.com/v1/embeddings"),
        )
        for _ in range(3)
    ]
    fake_client = _FakeClient(errors)
    delays: list[float] = []

    with pytest.raises(EmbeddingError, match="after 3 attempts"):
        _client(fake_client, max_retries=2, sleep=delays.append).embed_texts(["retry me"])

    assert delays == [0.25, 0.5]
    assert len(fake_client.embeddings.calls) == 3


def test_dimensions_must_match_between_batches() -> None:
    fake_client = _FakeClient(
        [
            _response([[1.0, 2.0], [3.0, 4.0]]),
            _response([[5.0, 6.0, 7.0]]),
        ]
    )

    with pytest.raises(EmbeddingError, match="changed between batches"):
        _client(fake_client).embed_texts(["one", "two", "three"])
