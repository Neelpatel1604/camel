# ========= Copyright 2023-2026 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2026 @ CAMEL-AI.org. All Rights Reserved. =========

import json
from unittest.mock import patch

import httpx
import pytest

from camel.toolkits.memanto_toolkit import MemantoToolkit


@pytest.fixture
def memanto_http():
    requests = []
    responses = []

    def handle(request):
        requests.append(request)
        if request.url.path.endswith("/activate"):
            return httpx.Response(200, json={"session_token": "test-token"})
        return responses.pop(0)

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        with patch("httpx.Client", return_value=client):
            toolkit = MemantoToolkit(
                agent_id="test-agent", base_url="https://memanto.example/"
            )
        try:
            yield toolkit, requests, responses
        finally:
            toolkit.close()


def test_memanto_remember(memanto_http):
    toolkit, requests, responses = memanto_http
    responses.append(httpx.Response(200, json={"memory_id": "mem-123"}))

    result = toolkit.memanto_remember(
        content="User prefers Python",
        memory_type="preference",
        tags="language, python",
        confidence=0.9,
    )

    assert "mem-123" in result
    assert requests[0].method == "POST"
    assert requests[0].url.path == "/api/v2/agents/test-agent/activate"
    request = requests[-1]
    assert request.method == "POST"
    assert str(request.url) == (
        "https://memanto.example/api/v2/agents/test-agent/remember"
    )
    assert request.headers["X-Session-Token"] == "test-token"
    payload = json.loads(request.content)
    assert payload["content"] == "User prefers Python"
    assert payload["type"] == "preference"
    assert payload["confidence"] == 0.9
    assert payload["tags"] == ["language", "python"]


def test_memanto_recall(memanto_http):
    toolkit, requests, responses = memanto_http
    memories = [{"content": "User prefers Python", "type": "preference"}]
    responses.append(httpx.Response(200, json={"memories": memories}))

    result = toolkit.memanto_recall(
        query="What language does the user prefer?",
        limit=3,
        memory_type="preference, fact, ",
    )

    assert json.loads(result) == memories
    assert requests[-1].url.path == "/api/v2/agents/test-agent/recall"
    assert json.loads(requests[-1].content) == {
        "query": "What language does the user prefer?",
        "limit": 3,
        "type": ["preference", "fact"],
    }


def test_memanto_answer(memanto_http):
    toolkit, requests, responses = memanto_http
    answer = "The user prefers Python."
    responses.append(httpx.Response(200, json={"answer": answer}))

    assert toolkit.memanto_answer("Preferred language?") == answer
    assert requests[-1].url.path == "/api/v2/agents/test-agent/answer"
    assert json.loads(requests[-1].content) == {
        "question": "Preferred language?"
    }


def test_get_tools(memanto_http):
    toolkit, _, _ = memanto_http
    schemas = [tool.get_openai_tool_schema() for tool in toolkit.get_tools()]
    assert {schema["function"]["name"] for schema in schemas} == {
        "memanto_remember",
        "memanto_recall",
        "memanto_answer",
    }
    remember = schemas[0]["function"]["parameters"]["properties"]
    assert "preference" in remember["memory_type"]["enum"]


def test_initialization_requires_agent_id(monkeypatch):
    monkeypatch.delenv("MEMANTO_AGENT_ID", raising=False)
    with patch("httpx.Client") as client:
        with pytest.raises(ValueError, match="agent_id must be provided"):
            MemantoToolkit()
        client.assert_not_called()


def test_expired_session_retries_with_new_token(memanto_http):
    toolkit, requests, responses = memanto_http
    toolkit._client._session_token = "expired-token"
    responses.extend(
        [
            httpx.Response(401),
            httpx.Response(200, json={"memory_id": "mem-123"}),
        ]
    )

    assert "mem-123" in toolkit.memanto_remember("A fact")
    assert [request.url.path.rsplit("/", 1)[-1] for request in requests] == [
        "activate",
        "remember",
        "activate",
        "remember",
    ]
    assert requests[1].headers["X-Session-Token"] == "expired-token"
    assert requests[3].headers["X-Session-Token"] == "test-token"
    assert requests[1].content == requests[3].content


@pytest.mark.parametrize("status", [401, 500])
def test_failed_request_returns_error_without_unbounded_retry(
    memanto_http, status
):
    toolkit, requests, responses = memanto_http
    responses.extend([httpx.Response(status), httpx.Response(status)])

    assert toolkit.memanto_answer("Question?").startswith("[ERROR]")
    assert len(requests) == (4 if status == 401 else 2)


def test_failed_activation_closes_client():
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(404))
    ) as client:
        with patch("httpx.Client", return_value=client):
            with pytest.raises(httpx.HTTPStatusError):
                MemantoToolkit(agent_id="missing-agent")
        assert client.is_closed


def test_close(memanto_http):
    toolkit, _, _ = memanto_http
    toolkit.close()
    assert toolkit._client._client.is_closed
