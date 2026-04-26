"""SaaS web platform adapter.

This adapter is intended for product backends that sit between a browser UI
and a per-workspace Hermes profile.  The browser should authenticate to the
SaaS backend; the backend authenticates to this adapter over a private network
or loopback interface.  Inbound ``user_id`` values identify the human actor
inside the workspace, not the Hermes profile itself.
"""

import asyncio
import hmac
import json
import logging
import os
import socket as _socket
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, Optional

try:
    import aiohttp
    from aiohttp import web

    AIOHTTP_AVAILABLE = True
except ImportError:
    aiohttp = None  # type: ignore[assignment]
    web = None  # type: ignore[assignment]
    AIOHTTP_AVAILABLE = False

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    is_network_accessible,
)

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8652
DEFAULT_MAX_BODY_BYTES = 1_048_576
DEFAULT_EVENT_BUFFER_SIZE = 200
DEFAULT_IDEMPOTENCY_TTL_SECONDS = 3600


def check_saas_web_requirements() -> bool:
    """Check if SaaS web adapter dependencies are available."""
    return AIOHTTP_AVAILABLE


class SaasWebAdapter(BasePlatformAdapter):
    """Internal HTTP adapter for a SaaS backend/frontend shell."""

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.SAAS_WEB)
        extra = config.extra or {}

        self._host: str = extra.get("host", os.getenv("SAAS_WEB_HOST", DEFAULT_HOST))
        self._port: int = int(extra.get("port", os.getenv("SAAS_WEB_PORT", str(DEFAULT_PORT))))
        self._key: str = extra.get("key", os.getenv("SAAS_WEB_KEY", ""))
        self._callback_url: str = extra.get(
            "callback_url",
            os.getenv("SAAS_WEB_CALLBACK_URL", ""),
        )
        self._callback_key: str = extra.get(
            "callback_key",
            os.getenv("SAAS_WEB_CALLBACK_KEY", self._key),
        )
        self._default_user_id: str = extra.get(
            "user_id",
            os.getenv("SAAS_WEB_USER_ID", ""),
        )
        self._default_user_name: str = extra.get(
            "user_name",
            os.getenv("SAAS_WEB_USER_NAME", ""),
        )
        self._workspace_id: str = extra.get(
            "workspace_id",
            os.getenv("SAAS_WEB_WORKSPACE_ID", ""),
        )
        self._workspace_name: str = extra.get(
            "workspace_name",
            os.getenv("SAAS_WEB_WORKSPACE_NAME", ""),
        )
        self._max_body_bytes: int = int(extra.get("max_body_bytes", DEFAULT_MAX_BODY_BYTES))
        self._event_buffer_size: int = int(
            extra.get("event_buffer_size", DEFAULT_EVENT_BUFFER_SIZE)
        )
        self._idempotency_ttl: int = int(
            extra.get("idempotency_ttl", DEFAULT_IDEMPOTENCY_TTL_SECONDS)
        )

        self._runner = None
        self._site = None
        self._http_session: Optional["aiohttp.ClientSession"] = None
        self._seen_messages: Dict[str, float] = {}
        self._events: Dict[str, Deque[dict]] = defaultdict(
            lambda: deque(maxlen=self._event_buffer_size)
        )
        self._conversation_workspaces: Dict[str, tuple[str, str]] = {}
        self._event_seq = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        if not AIOHTTP_AVAILABLE:
            logger.error("[saas_web] aiohttp is not installed")
            return False

        if is_network_accessible(self._host) and not self._key:
            logger.error(
                "[saas_web] Refusing to start on %s without SAAS_WEB_KEY",
                self._host,
            )
            return False

        if not self._key:
            logger.warning(
                "[saas_web] No SAAS_WEB_KEY configured. Accepting unauthenticated "
                "loopback requests only; set a key before production use."
            )

        try:
            with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                sock.connect(("127.0.0.1", self._port))
            logger.error(
                "[saas_web] Port %d already in use. Set platforms.saas_web.extra.port "
                "or SAAS_WEB_PORT.",
                self._port,
            )
            return False
        except (ConnectionRefusedError, OSError):
            pass

        app = self._create_app()
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self._host, self._port)
        await self._site.start()
        self._mark_connected()

        logger.info("[saas_web] Listening on http://%s:%d", self._host, self._port)
        return True

    async def disconnect(self) -> None:
        if self._http_session is not None:
            await self._http_session.close()
            self._http_session = None
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
            self._site = None
        self._mark_disconnected()
        logger.info("[saas_web] Disconnected")

    def _create_app(self) -> "web.Application":
        app = web.Application()
        app.router.add_get("/health", self._handle_health)
        app.router.add_post("/messages", self._handle_message_post)
        app.router.add_get("/events/{conversation_id}", self._handle_events)
        return app

    # ------------------------------------------------------------------
    # Platform operations
    # ------------------------------------------------------------------

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        event = self._record_event(
            conversation_id=chat_id,
            event_type="message",
            payload={
                "conversation_id": chat_id,
                "message_id": f"msg_{uuid.uuid4().hex}",
                "reply_to": reply_to,
                "content": content,
                "metadata": metadata or {},
            },
        )

        if not self._callback_url:
            logger.info("[saas_web] Queued response for %s: %s", chat_id, content[:200])
            return SendResult(success=True, message_id=event["message_id"])

        try:
            await self._post_callback(event)
            return SendResult(success=True, message_id=event["message_id"])
        except Exception as exc:
            logger.warning("[saas_web] Callback delivery failed: %s", exc)
            return SendResult(success=False, error=str(exc), retryable=True)

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        event = self._record_event(
            conversation_id=chat_id,
            event_type="typing",
            payload={
                "conversation_id": chat_id,
                "metadata": metadata or {},
            },
        )
        if self._callback_url:
            try:
                await self._post_callback(event)
            except Exception as exc:
                logger.debug("[saas_web] Typing callback failed: %s", exc)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": chat_id, "type": "dm", "chat_id": chat_id}

    # ------------------------------------------------------------------
    # HTTP handlers
    # ------------------------------------------------------------------

    async def _handle_health(self, request: "web.Request") -> "web.Response":
        return web.json_response(
            {
                "status": "ok",
                "platform": "saas_web",
                "callback_configured": bool(self._callback_url),
                "workspace_id": self._workspace_id or None,
                "workspace_name": self._workspace_name or None,
            }
        )

    async def _handle_message_post(self, request: "web.Request") -> "web.Response":
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        content_length = request.content_length or 0
        if content_length > self._max_body_bytes:
            return web.json_response({"error": "Payload too large"}, status=413)

        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        text = str(body.get("text") or body.get("message") or "").strip()
        if not text:
            return web.json_response({"error": "Missing message text"}, status=400)

        conversation_id = str(
            body.get("conversation_id") or body.get("chat_id") or "default"
        )
        user_id = str(body.get("user_id") or self._default_user_id or "saas_user")
        user_name = body.get("user_name") or self._default_user_name or None
        workspace_id = str(body.get("workspace_id") or self._workspace_id or "")
        workspace_name = str(body.get("workspace_name") or self._workspace_name or "")
        message_id = str(body.get("message_id") or f"in_{uuid.uuid4().hex}")
        thread_id = body.get("thread_id")

        now = time.time()
        self._prune_seen_messages(now)
        if message_id in self._seen_messages:
            return web.json_response(
                {
                    "status": "duplicate",
                    "message_id": message_id,
                    "conversation_id": conversation_id,
                },
                status=200,
            )
        self._seen_messages[message_id] = now
        if workspace_id or workspace_name:
            self._conversation_workspaces[conversation_id] = (workspace_id, workspace_name)

        source = self.build_source(
            chat_id=conversation_id,
            chat_name=body.get("conversation_name") or conversation_id,
            chat_type=str(body.get("chat_type") or "dm"),
            user_id=user_id,
            user_name=user_name,
            thread_id=str(thread_id) if thread_id else None,
            chat_topic=body.get("conversation_topic") or None,
        )
        event = MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=body,
            message_id=message_id,
            auto_skill=body.get("auto_skill"),
            channel_prompt=body.get("channel_prompt"),
        )

        task = asyncio.create_task(self.handle_message(event))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

        return web.json_response(
            {
                "status": "accepted",
                "message_id": message_id,
                "conversation_id": conversation_id,
                "user_id": user_id,
                "workspace_id": workspace_id or None,
            },
            status=202,
        )

    async def _handle_events(self, request: "web.Request") -> "web.Response":
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        conversation_id = request.match_info.get("conversation_id", "")
        try:
            after = int(request.query.get("after", "0"))
        except ValueError:
            return web.json_response({"error": "Invalid after cursor"}, status=400)

        events = [
            event
            for event in list(self._events.get(conversation_id, ()))
            if int(event.get("seq", 0)) > after
        ]
        return web.json_response({"events": events})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_auth(self, request: "web.Request") -> Optional["web.Response"]:
        if not self._key:
            return None

        token = ""
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
        else:
            token = request.headers.get("X-Saas-Web-Key", "").strip()

        if token and hmac.compare_digest(token, self._key):
            return None

        return web.json_response({"error": "Unauthorized"}, status=401)

    def _record_event(self, *, conversation_id: str, event_type: str, payload: dict) -> dict:
        self._event_seq += 1
        event = {
            "seq": self._event_seq,
            "event": event_type,
            "platform": "saas_web",
            "conversation_id": conversation_id,
            "message_id": payload.get("message_id") or f"evt_{uuid.uuid4().hex}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        workspace_id, workspace_name = self._conversation_workspaces.get(
            conversation_id,
            (self._workspace_id, self._workspace_name),
        )
        if workspace_id and "workspace_id" not in event:
            event["workspace_id"] = workspace_id
        if workspace_name and "workspace_name" not in event:
            event["workspace_name"] = workspace_name
        self._events[conversation_id].append(event)
        return event

    async def _post_callback(self, event: dict) -> None:
        if self._http_session is None or self._http_session.closed:
            self._http_session = aiohttp.ClientSession()

        body = json.dumps(event, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._callback_key:
            headers["Authorization"] = f"Bearer {self._callback_key}"

        async with self._http_session.post(
            self._callback_url,
            data=body,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as response:
            if response.status >= 400:
                text = await response.text()
                raise RuntimeError(f"callback HTTP {response.status}: {text[:200]}")

    def _prune_seen_messages(self, now: float) -> None:
        cutoff = now - self._idempotency_ttl
        stale = [key for key, ts in self._seen_messages.items() if ts < cutoff]
        for key in stale:
            self._seen_messages.pop(key, None)
