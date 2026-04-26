"""Tests for the SaaS web platform adapter."""

import asyncio
import importlib
import sys
import time
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import GatewayConfig, Platform, PlatformConfig, _apply_env_overrides
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource
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


class TestRichEvents:
    @pytest.mark.asyncio
    async def test_rich_event_records_and_callbacks_when_enabled(self):
        received = []

        async def callback(request):
            received.append(await request.json())
            return web.json_response({"ok": True})

        callback_app = web.Application()
        callback_app.router.add_post("/callback", callback)
        callback_client = TestClient(TestServer(callback_app))
        await callback_client.start_server()

        adapter = _make_adapter(
            callback_url=str(callback_client.make_url("/callback")),
            rich_events=True,
            workspace_id="ws_123",
            workspace_name="Personal",
        )
        try:
            event = await adapter.emit_runtime_event(
                "conv_1",
                "tool.started",
                {
                    "tool_call_id": "call_1",
                    "tool_name": "terminal",
                    "preview": "pwd",
                    "args_keys": ["command"],
                },
                run_id="msg_user_1",
                reply_to="msg_user_1",
                parent_message_id="msg_user_1",
            )

            assert event["event"] == "tool.started"
            assert event["run_id"] == "msg_user_1"
            assert event["workspace_id"] == "ws_123"
            assert event["metadata"] == {}
            assert received[0]["message_id"].startswith("evt_")
            assert received[0]["seq"] == event["seq"]

            client = TestClient(TestServer(_create_app(adapter)))
            await client.start_server()
            try:
                resp = await client.get(
                    "/events/conv_1",
                    headers={"Authorization": "Bearer backend-secret"},
                )
                events = (await resp.json())["events"]
                assert [item["seq"] for item in events] == sorted(item["seq"] for item in events)
                assert events[0]["event"] == "tool.started"
            finally:
                await client.close()
        finally:
            await adapter.disconnect()
            await callback_client.close()

    @pytest.mark.asyncio
    async def test_rich_event_ignored_when_disabled(self):
        adapter = _make_adapter(rich_events=False)
        event = await adapter.emit_runtime_event(
            "conv_1",
            "message.delta",
            {"delta": "hello", "index": 1},
            run_id="msg_1",
        )
        assert event is None
        assert list(adapter._events.get("conv_1", ())) == []

    def test_env_overrides_rich_events(self, monkeypatch):
        monkeypatch.setenv("SAAS_WEB_ENABLED", "true")
        monkeypatch.setenv("SAAS_WEB_KEY", "secret")
        monkeypatch.setenv("SAAS_WEB_RICH_EVENTS", "true")

        config = GatewayConfig()
        _apply_env_overrides(config)

        assert config.platforms[Platform.SAAS_WEB].extra["rich_events"] == "true"


class RichFakeAgent:
    def __init__(self, **kwargs):
        self.tools = []
        self.tool_progress_callback = None
        self.tool_start_callback = None
        self.tool_complete_callback = None
        self.step_callback = None
        self.stream_delta_callback = None
        self.reasoning_callback = None
        self.status_callback = None
        self.background_review_callback = None
        self.context_compressor = SimpleNamespace(last_prompt_tokens=0)
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.model = kwargs.get("model", "fake-model")
        self.session_id = kwargs.get("session_id")

    def run_conversation(self, message, conversation_history=None, task_id=None):
        if self.status_callback:
            self.status_callback("context_pressure", "Context pressure is high")
        if self.reasoning_callback:
            self.reasoning_callback("private reasoning " * 200)
        if self.tool_progress_callback:
            self.tool_progress_callback(
                "tool.started",
                "terminal",
                "pwd",
                {"command": "pwd", "secret": "do-not-emit-full-result"},
            )
        if self.tool_start_callback:
            self.tool_start_callback("call_1", "terminal", {"command": "pwd"})
        time.sleep(0.01)
        if self.tool_progress_callback:
            self.tool_progress_callback(
                "tool.completed",
                "terminal",
                None,
                None,
                duration=1.234,
                is_error=False,
            )
        if self.tool_complete_callback:
            self.tool_complete_callback("call_1", "terminal", {"command": "pwd"}, "x" * 2000)
        if self.step_callback:
            self.step_callback(2, [{"name": "terminal", "result": "raw result", "arguments": "{}"}])
        if self.stream_delta_callback:
            self.stream_delta_callback("Hello")
            self.stream_delta_callback(" world")
        return {
            "final_response": "Hello world",
            "last_reasoning": "summary " * 400,
            "messages": [
                {"role": "user", "content": message},
                {"role": "assistant", "content": "Hello world"},
            ],
            "api_calls": 1,
        }


def _make_rich_runner(adapter):
    gateway_run = importlib.import_module("gateway.run")
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {Platform.SAAS_WEB: adapter}
    runner._voice_mode = {}
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = ""
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._session_db = None
    runner._running_agents = {}
    runner._session_run_generation = {}
    runner._agent_cache = None
    runner._agent_cache_lock = None
    runner._draining = False
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(
        thread_sessions_per_user=False,
        group_sessions_per_user=False,
        stt_enabled=False,
    )
    return runner


async def _run_rich_agent(monkeypatch, tmp_path, *, rich_events=True):
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = RichFakeAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    adapter = _make_adapter(rich_events=rich_events, workspace_id="ws_123")
    runner = _make_rich_runner(adapter)
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})

    source = SessionSource(
        platform=Platform.SAAS_WEB,
        chat_id="conv_1",
        chat_type="dm",
        user_id="user_1",
    )
    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess_1",
        session_key="agent:main:saas_web:dm:conv_1",
        event_message_id="msg_user_1",
    )
    await asyncio.sleep(0.05)
    return adapter, result


class TestGatewayRichEvents:
    @pytest.mark.asyncio
    async def test_run_agent_emits_structured_runtime_events(self, monkeypatch, tmp_path):
        adapter, result = await _run_rich_agent(monkeypatch, tmp_path, rich_events=True)

        assert result["final_response"] == "Hello world"
        events = list(adapter._events["conv_1"])
        event_names = [event["event"] for event in events]
        assert "status" in event_names
        assert "reasoning.started" in event_names
        assert "reasoning.summary" in event_names
        assert "tool.started" in event_names
        assert "tool.completed" in event_names
        assert "step.completed" in event_names
        assert event_names.count("message.delta") == 2

        tool_started = next(event for event in events if event["event"] == "tool.started")
        assert tool_started["tool_call_id"] == "call_1"
        assert tool_started["run_id"] == "msg_user_1"
        assert tool_started["reply_to"] == "msg_user_1"
        assert tool_started["workspace_id"] == "ws_123"

        tool_done = next(event for event in events if event["event"] == "tool.completed")
        assert tool_done["duration_ms"] == 1234
        assert len(tool_done["result_preview"]) <= 500
        assert "x" * 600 not in tool_done["result_preview"]

        reasoning = next(event for event in events if event["event"] == "reasoning.summary")
        assert len(reasoning["summary"]) <= 1200
        assert reasoning["truncated"] is True

    @pytest.mark.asyncio
    async def test_run_agent_omits_rich_events_when_disabled(self, monkeypatch, tmp_path):
        adapter, result = await _run_rich_agent(monkeypatch, tmp_path, rich_events=False)

        assert result["final_response"] == "Hello world"
        assert not any("." in event["event"] for event in adapter._events.get("conv_1", ()))


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
