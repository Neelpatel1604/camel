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
import logging
import os
from typing import Any, Dict, List, Literal, Optional

import httpx

from camel.toolkits.base import BaseToolkit
from camel.toolkits.function_tool import FunctionTool

logger = logging.getLogger(__name__)


MemantoMemoryType = Literal[
    "fact",
    "preference",
    "goal",
    "decision",
    "artifact",
    "learning",
    "event",
    "instruction",
    "relationship",
    "context",
    "observation",
    "commitment",
    "error",
]


class _MemantoRESTClient:
    r"""REST client for the Memanto memory API.

    Args:
        agent_id (str): Memanto agent identifier.
        base_url (str, optional): Memanto server URL. Defaults to
            :obj:`MEMANTO_BASE_URL` or ``http://localhost:8000``.
        timeout (float, optional): HTTP request timeout in seconds.
            (default: :obj:`30.0`)
    """

    def __init__(
        self,
        agent_id: str,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        self.agent_id = agent_id
        self.base_url = (
            base_url
            or os.getenv("MEMANTO_BASE_URL")
            or "http://localhost:8000"
        ).rstrip("/")
        self._client = httpx.Client(timeout=timeout)
        self._session_token: Optional[str] = None
        try:
            self.activate_session()
        except Exception:
            self._client.close()
            raise

    def activate_session(self) -> str:
        r"""Activate an agent session and store the session token.

        Returns:
            str: The session token for subsequent memory operations.
        """
        response = self._client.post(
            f"{self.base_url}/api/v2/agents/{self.agent_id}/activate"
        )
        response.raise_for_status()
        data = response.json()
        self._session_token = data["session_token"]
        return self._session_token

    def _headers(self) -> Dict[str, str]:
        if not self._session_token:
            self.activate_session()
        return {
            "X-Session-Token": self._session_token or "",
            "Content-Type": "application/json",
        }

    def _session_request(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> httpx.Response:
        r"""Send a request and re-activate the session once on 401."""
        response = self._client.request(
            method,
            url,
            headers=self._headers(),
            **kwargs,
        )
        if response.status_code == 401:
            self.activate_session()
            response = self._client.request(
                method,
                url,
                headers=self._headers(),
                **kwargs,
            )
        response.raise_for_status()
        return response

    def remember(
        self,
        content: str,
        memory_type: MemantoMemoryType = "context",
        title: Optional[str] = None,
        confidence: float = 1.0,
        tags: Optional[List[str]] = None,
    ) -> str:
        r"""Store a memory for the active agent.

        Args:
            content (str): Memory content.
            memory_type (MemantoMemoryType, optional): Semantic memory type.
                (default: :obj:`"context"`)
            title (Optional[str], optional): Short title for the memory.
            confidence (float, optional): Confidence score.
                (default: :obj:`1.0`)
            tags (Optional[List[str]], optional): Optional tags.

        Returns:
            str: The stored memory ID.
        """
        payload: Dict[str, Any] = {
            "content": content,
            "type": memory_type,
            "title": title or f"{memory_type.title()}: {content[:50]}",
            "confidence": confidence,
        }
        if tags:
            payload["tags"] = tags

        response = self._session_request(
            "POST",
            f"{self.base_url}/api/v2/agents/{self.agent_id}/remember",
            json=payload,
        )
        return response.json()["memory_id"]

    def recall(
        self,
        query: str,
        limit: int = 10,
        memory_type: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        r"""Retrieve memories by semantic similarity.

        Args:
            query (str): Natural-language search query.
            limit (int, optional): Maximum memories to return.
                (default: :obj:`10`)
            memory_type (Optional[List[str]], optional): Optional
                type filter.

        Returns:
            List[Dict[str, Any]]: Matching memory records.
        """
        payload: Dict[str, Any] = {"query": query, "limit": limit}
        if memory_type:
            payload["type"] = memory_type

        response = self._session_request(
            "POST",
            f"{self.base_url}/api/v2/agents/{self.agent_id}/recall",
            json=payload,
        )
        return response.json().get("memories", [])

    def answer(self, question: str) -> str:
        r"""Generate an answer grounded in stored memories.

        Args:
            question (str): Question to answer.

        Returns:
            str: Generated answer text.
        """
        response = self._session_request(
            "POST",
            f"{self.base_url}/api/v2/agents/{self.agent_id}/answer",
            json={"question": question},
        )
        return response.json().get("answer", "")

    def close(self) -> None:
        r"""Close the underlying HTTP client."""
        self._client.close()


class MemantoToolkit(BaseToolkit):
    r"""A toolkit for explicit Memanto memory operations via function calling.

    Args:
        agent_id (str, optional): Memanto agent identifier. Falls back to
            :obj:`MEMANTO_AGENT_ID` if not provided.
        base_url (str, optional): Memanto server URL. Defaults to
            :obj:`MEMANTO_BASE_URL` or ``http://localhost:8000``.
        timeout (Optional[float], optional): Maximum execution time allowed for
            toolkit operations in seconds. If None, no timeout is applied.
            (default: :obj:`None`)
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        super().__init__(timeout=timeout)
        resolved_agent_id = agent_id or os.getenv("MEMANTO_AGENT_ID")
        if not resolved_agent_id:
            raise ValueError(
                "agent_id must be provided or set via MEMANTO_AGENT_ID."
            )

        self.agent_id = resolved_agent_id
        self._client = _MemantoRESTClient(
            agent_id=self.agent_id,
            base_url=base_url,
        )

    def memanto_remember(
        self,
        content: str,
        memory_type: MemantoMemoryType = "fact",
        tags: Optional[str] = None,
        confidence: float = 0.8,
    ) -> str:
        r"""Store a typed memory in Memanto.

        Args:
            content (str): Atomic memory statement to store.
            memory_type (MemantoMemoryType, optional): Semantic memory type.
                (default: :obj:`"fact"`)
            tags (Optional[str], optional): Comma-separated tags.
            confidence (float, optional): Confidence score between 0 and 1.
                (default: :obj:`0.8`)

        Returns:
            str: Confirmation message with the stored memory ID.
        """
        try:
            tag_list = (
                [tag.strip() for tag in tags.split(",") if tag.strip()]
                if tags
                else None
            )
            memory_id = self._client.remember(
                content=content,
                memory_type=memory_type,
                confidence=confidence,
                tags=tag_list,
            )
            return f"Memory stored with ID: {memory_id}"
        except Exception as error:
            logger.error(f"Memanto remember failed: {error}")
            return f"[ERROR] Failed to store memory: {error}"

    def memanto_recall(
        self,
        query: str,
        limit: int = 5,
        memory_type: Optional[str] = None,
    ) -> str:
        r"""Search Memanto memories by semantic similarity.

        Args:
            query (str): Natural-language search query.
            limit (int, optional): Maximum memories to return.
                (default: :obj:`5`)
            memory_type (Optional[str], optional): Comma-separated memory types
                to filter by.

        Returns:
            str: JSON string of matching memories.
        """
        try:
            type_filter = None
            if memory_type:
                type_filter = [
                    item.strip()
                    for item in memory_type.split(",")
                    if item.strip()
                ]
            memories = self._client.recall(
                query=query,
                limit=limit,
                memory_type=type_filter,
            )
            return json.dumps(memories, indent=2)
        except Exception as error:
            logger.error(f"Memanto recall failed: {error}")
            return f"[ERROR] Failed to recall memories: {error}"

    def memanto_answer(self, question: str) -> str:
        r"""Generate an answer grounded in Memanto memories.

        Args:
            question (str): Question to answer from stored memories.

        Returns:
            str: Generated answer text.
        """
        try:
            return self._client.answer(question=question)
        except Exception as error:
            logger.error(f"Memanto answer failed: {error}")
            return f"[ERROR] Failed to generate answer: {error}"

    def get_tools(self) -> List[FunctionTool]:
        r"""Expose Memanto memory tools for function calling.

        Returns:
            List[FunctionTool]: Toolkit function tools.
        """
        return [
            FunctionTool(self.memanto_remember),
            FunctionTool(self.memanto_recall),
            FunctionTool(self.memanto_answer),
        ]

    def close(self) -> None:
        r"""Close the HTTP client when the toolkit is no longer needed."""
        self._client.close()
