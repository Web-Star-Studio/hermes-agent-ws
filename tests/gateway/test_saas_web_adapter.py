"""Tests for the SaaS web platform adapter."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import GatewayConfig, Platform, PlatformConfig, _apply_env_overrides
from gateway.platforms.base import MessageEvent
from gateway.platforms.saas_web import (
    SaasWebAdapter,
    check_saas_web_requirements,
)


def _make_adapter(**extra):
    config = PlatformConfig(
        enabled=True,
        extra={
            "host": "127.0.0.1",
            "port": 0,
            "key": "backend-secret",
            **extra,
        },
    )
    return SaasWebAdapter(config)


def _create_app(adapter: SaasWebAdapter) -> web.Application:
    return adapter._create_app()


class TestCheckRequirements:
    def test_returns_true_when_aiohttp_available(self):
        assert check_saas_web_requirements() is True

    @patch("gateway.platforms.saas_web.AIOHTTP_AVAILABLE", False)
    def test_returns_false_without_aiohttp(self):
        assert check_saas_web_requirements() is False


class TestAdapterInit:
    def test_defaults(self):
        adapter = SaasWebAdapter(PlatformConfig(enabled=True))
        assert adapter.platform == Platform.SAAS_WEB
        assert adapter._host == "127.0.0.1"
        assert adapter._port == 8652

    def test_custom_extra(self):
        adapter = _make_adapter(
            host="0.0.0.0",
            port=9999,
            callback_url="http://backend/events",
            callback_key="callback-secret",
            user_id="user_123",
            user_name="Alice",
            workspace_id="ws_123",
            workspace_name="Personal",
        )
        assert adapter._host == "0.0.0.0"
        assert adapter._port == 9999
        assert adapter._callback_url == "http://backend/events"
        assert adapter._callback_key == "callback-secret"
        assert adapter._default_user_id == "user_123"
        assert adapter._default_user_name == "Alice"
        assert adapter._workspace_id == "ws_123"
        assert adapter._workspace_name == "Personal"


class TestAuth:
    @pytest.mark.asyncio
    async def test_rejects_missing_auth(self):
        adapter = _make_adapter()
        client = TestClient(TestServer(_create_app(adapter)))
        await client.start_server()
        try:
            resp = await client.post("/messages", json={"text": "hello"})
            assert resp.status == 401
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_accepts_bearer_auth(self):
        adapter = _make_adapter()
        adapter.handle_message = AsyncMock()
        client = TestClient(TestServer(_create_app(adapter)))
        await client.start_server()
        try:
            resp = await client.post(
                "/messages",
                json={"text": "hello", "conversation_id": "conv_1", "user_id": "user_1"},
                headers={"Authorization": "Bearer backend-secret"},
            )
            assert resp.status == 202
        finally:
            await client.close()


class TestMessages:
    @pytest.mark.asyncio
    async def test_message_post_builds_message_event(self):
        adapter = _make_adapter(user_id="default_user", user_name="Default User")
        adapter.handle_message = AsyncMock()
        client = TestClient(TestServer(_create_app(adapter)))
        await client.start_server()
        try:
            resp = await client.post(
                "/messages",
                json={
                    "text": "plan my day",
                    "conversation_id": "conv_day",
                    "conversation_name": "Daily dashboard",
                    "message_id": "msg_1",
                    "thread_id": "thread_a",
                    "workspace_id": "ws_123",
                    "workspace_name": "Personal",
                },
                headers={"Authorization": "Bearer backend-secret"},
            )
            assert resp.status == 202
            payload = await resp.json()
            assert payload["conversation_id"] == "conv_day"
            assert payload["user_id"] == "default_user"
            assert payload["workspace_id"] == "ws_123"

            await asyncio.sleep(0)
            adapter.handle_message.assert_awaited_once()
            event = adapter.handle_message.await_args.args[0]
            assert isinstance(event, MessageEvent)
            assert event.text == "plan my day"
            assert event.message_id == "msg_1"
            assert event.source.platform == Platform.SAAS_WEB
            assert event.source.chat_id == "conv_day"
            assert event.source.chat_type == "dm"
            assert event.source.user_id == "default_user"
            assert event.source.user_name == "Default User"
            assert event.source.thread_id == "thread_a"
            assert event.raw_message["workspace_id"] == "ws_123"
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_duplicate_message_id_is_idempotent(self):
        adapter = _make_adapter()
        adapter.handle_message = AsyncMock()
        client = TestClient(TestServer(_create_app(adapter)))
        await client.start_server()
        try:
            body = {
                "text": "hello",
                "conversation_id": "conv_1",
                "user_id": "user_1",
                "message_id": "msg_repeat",
            }
            headers = {"Authorization": "Bearer backend-secret"}
            first = await client.post("/messages", json=body, headers=headers)
            second = await client.post("/messages", json=body, headers=headers)

            assert first.status == 202
            assert second.status == 200
            assert (await second.json())["status"] == "duplicate"
            await asyncio.sleep(0)
            adapter.handle_message.assert_awaited_once()
        finally:
            await client.close()


class TestDelivery:
    @pytest.mark.asyncio
    async def test_send_records_pollable_event_when_callback_missing(self):
        adapter = _make_adapter(callback_url="")
        result = await adapter.send("conv_1", "hello back", reply_to="msg_1")
        assert result.success is True

        client = TestClient(TestServer(_create_app(adapter)))
        await client.start_server()
        try:
            resp = await client.get(
                "/events/conv_1",
                headers={"Authorization": "Bearer backend-secret"},
            )
            assert resp.status == 200
            events = (await resp.json())["events"]
            assert len(events) == 1
            assert events[0]["event"] == "message"
            assert events[0]["content"] == "hello back"
            assert events[0]["reply_to"] == "msg_1"
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_send_posts_callback(self):
        received = []

        async def callback(request):
            received.append(
                {
                    "auth": request.headers.get("Authorization"),
                    "body": await request.json(),
                }
            )
            return web.json_response({"ok": True})

        callback_app = web.Application()
        callback_app.router.add_post("/callback", callback)
        callback_client = TestClient(TestServer(callback_app))
        await callback_client.start_server()

        adapter = _make_adapter(
            callback_url=str(callback_client.make_url("/callback")),
            callback_key="callback-secret",
            workspace_id="ws_123",
            workspace_name="Personal",
        )
        try:
            result = await adapter.send("conv_1", "hello callback")
            assert result.success is True
            assert len(received) == 1
            assert received[0]["auth"] == "Bearer callback-secret"
            assert received[0]["body"]["event"] == "message"
            assert received[0]["body"]["conversation_id"] == "conv_1"
            assert received[0]["body"]["workspace_id"] == "ws_123"
            assert received[0]["body"]["workspace_name"] == "Personal"
            assert received[0]["body"]["content"] == "hello callback"
        finally:
            await adapter.disconnect()
            await callback_client.close()


class TestConfig:
    def test_env_overrides_enable_saas_web(self, monkeypatch):
        monkeypatch.setenv("SAAS_WEB_ENABLED", "true")
        monkeypatch.setenv("SAAS_WEB_KEY", "secret")
        monkeypatch.setenv("SAAS_WEB_PORT", "9001")
        monkeypatch.setenv("SAAS_WEB_CALLBACK_URL", "http://backend/events")
        monkeypatch.setenv("SAAS_WEB_USER_ID", "user_123")
        monkeypatch.setenv("SAAS_WEB_WORKSPACE_ID", "ws_123")

        config = GatewayConfig()
        _apply_env_overrides(config)

        platform_config = config.platforms[Platform.SAAS_WEB]
        assert platform_config.enabled is True
        assert platform_config.extra["key"] == "secret"
        assert platform_config.extra["port"] == 9001
        assert platform_config.extra["callback_url"] == "http://backend/events"
        assert platform_config.extra["user_id"] == "user_123"
        assert platform_config.extra["workspace_id"] == "ws_123"
