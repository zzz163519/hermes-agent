"""Plugin observers can distinguish cache-parity forks from live requests (#142321)."""

import pytest

from agent.turn_api_request import _fire_pre_api_request_hook
from hermes_cli import plugins
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from run_agent import AIAgent


def _agent(origin=None):
    agent = object.__new__(AIAgent)
    agent.session_id = "shared-session"
    agent.platform = "cli"
    agent.model = "test-model"
    agent.provider = "custom"
    agent.base_url = "https://provider.example/v1"
    agent.api_mode = "chat_completions"
    agent.max_tokens = 100
    agent.tools = []
    if origin is not None:
        agent._turn_origin = origin
    return agent


def _emit(agent, text):
    messages = [{"role": "user", "content": text}]
    request = {"model": agent.model, "messages": messages}
    _fire_pre_api_request_hook(
        agent, request, messages, [], messages=messages, original_user_message=text,
        approx_tokens=10, total_chars=len(text), retry_count=0, api_call_count=1,
        api_request_id="request-1", api_start_time=0, effective_task_id="task-1",
        turn_id="turn-1",
    )
    assert request == {"model": agent.model, "messages": [{"role": "user", "content": text}]}


@pytest.fixture
def hook_context(monkeypatch):
    manager = PluginManager()
    manager.discover_and_load()
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    return PluginContext(PluginManifest(name="capture-test"), manager)


@pytest.mark.parametrize("origin", [None, "background_review", "side_question"])
def test_pre_api_request_delivers_origin_to_registered_plugin(hook_context, origin):
    events = []
    hook_context.register_hook("pre_api_request", lambda **payload: events.append(payload))

    _emit(_agent(origin), "Hello")

    assert len(events) == 1
    assert events[0]["turn_origin"] == origin
    assert events[0]["session_id"] == "shared-session"
    assert events[0]["request"]["body"]["messages"] == [{"role": "user", "content": "Hello"}]


@pytest.mark.parametrize("origin", ["background_review", "side_question"])
def test_fork_request_does_not_replace_live_session_capture(hook_context, origin):
    captures = {}

    def capture(*, session_id, request, turn_origin=None, **kwargs):
        if turn_origin not in {"background_review", "side_question"}:
            captures[session_id] = request["body"]

    hook_context.register_hook("pre_api_request", capture)

    _emit(_agent(), "Live conversation")
    _emit(_agent(origin), "Detached fork")

    assert captures == {
        "shared-session": {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Live conversation"}],
        }
    }
