"""Explicit response disposition survives live, queued and crash-left delivery."""

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent.response_skip import receipt
from tests.gateway.test_gateway_silence_tokens import _event, _runner, _source


def _result():
    return {"final_response": "", "completed": True, "failed": False, "interrupted": False,
            "turn_exit_reason": "skip_response", "api_calls": 1, "tools": [],
            "history_offset": 0, "last_prompt_tokens": 0,
            "messages": [{"role": "user", "content": "Stop replying."}, receipt("User requested silence.")]}


@pytest.mark.asyncio
async def test_explicit_skip_sends_nothing_in_live_and_queued_delivery(monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(return_value=_result())
    assert await runner._handle_message_with_agent(
        _event(reply_expected=True), _source(), "agent:main:telegram:group:-1001:12345", 1,
    ) == ""

    runner._deliver_queued_first_response = AsyncMock()
    ctx = SimpleNamespace(session_key="chat", stream_consumer_holder=[None], mute_notification_reply=False,
                          persist_user_display_kind=None, reply_expected=True, source=_source(),
                          _status_thread_metadata=None, event_message_id=None, inbound_message_id="42",
                          run_generation=1)
    result = _result()
    await runner._run_agent_deliver_first_response(ctx, None, result, result, None)
    runner._deliver_queued_first_response.assert_not_awaited()

    # A later human answer must not inherit the preceding turn's disposition.
    # The adapter normally releases its delivery lease after the handler returns.
    assert runner._release_turn_lease("agent:main:telegram:group:-1001:12345", 1)
    runner._run_agent.return_value = {**_result(), "turn_exit_reason": "text_response(finish_reason=stop)",
                                     "final_response": "Answering your new request."}
    assert await runner._handle_message_with_agent(
        _event(reply_expected=True), _source(), "agent:main:telegram:group:-1001:12345", 1,
    ) == "Answering your new request."


def test_crash_left_skip_receipt_is_settled_without_a_recovery_message(monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path)
    started = datetime.now(UTC).timestamp()
    history = _result()["messages"]
    history[-1]["timestamp"] = started + 1
    assert runner._crash_left_reply(history, started, _source()) == ""
    # An unrelated hidden/empty assistant row is not proof of an intentional skip.
    history[-1] = {"role": "assistant", "content": "", "display_kind": "hidden", "timestamp": started + 1}
    assert runner._crash_left_reply(history, started, _source()) is None


@pytest.mark.parametrize("override", [
    {"failed": True}, {"interrupted": True}, {"completed": False}, {"partial": True},
    {"error": "storage failed"},
    {"turn_exit_reason": "text_response(finish_reason=stop)"}, {"final_response": "An ordinary answer."},
])
def test_skip_outcome_never_hides_failures_or_ordinary_answers(override):
    from gateway.response_filters import is_intentional_silence_agent_result

    result = {**_result(), **override}
    assert not is_intentional_silence_agent_result(result, result["final_response"])


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [True, False])
async def test_streaming_skip_does_not_finalize_with_a_placeholder_or_buffered_text(native):
    from gateway.run_turn_runner import TurnRunner
    from gateway.stream_consumer import GatewayStreamConsumer
    from tests.gateway.test_stream_consumer_wecom_native import _make_native_streaming_adapter

    adapter = _make_native_streaming_adapter(supports_native=native)
    consumer = GatewayStreamConsumer(adapter, "chat")
    consumer.on_delta("Buffered text must not become a final reply.")
    turn = object.__new__(TurnRunner)
    turn._ctx = SimpleNamespace(result_holder=[None])
    turn._finish_stream_consumer(_result(), [], consumer)
    await consumer.run()
    adapter.send.assert_not_awaited()
    adapter.edit_message.assert_not_awaited()
    assert all(frame["text"] == "" for frame in adapter.frames)
    assert not consumer.final_response_sent


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata", [{}, {"message_id": "skip-turn"}])
async def test_skip_closes_a_running_relay_draft_without_intercepting_the_next_send(monkeypatch, metadata):
    from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
    from tests.gateway.relay.test_relay_live_cards import _connected_adapter

    adapter, connector = _connected_adapter(supported_ops=("send", "edit", "typing", "draft"))
    consumer = GatewayStreamConsumer(
        adapter, "C1", StreamConsumerConfig(
            transport="auto", chat_type="dm", edit_interval=0.01, buffer_threshold=1, cursor=""),
        metadata=metadata,
    )
    draft_sent = asyncio.Event()
    send_draft = adapter.send_draft

    async def observe_draft(*args, **kwargs):
        result = await send_draft(*args, **kwargs)
        draft_sent.set()
        return result

    monkeypatch.setattr(adapter, "send_draft", observe_draft)
    task = asyncio.create_task(consumer.run())
    try:
        consumer.on_delta("Current unfinalized preview")
        await asyncio.wait_for(draft_sent.wait(), 2)
        consumer.finish_silently()
        await asyncio.wait_for(task, 2)

        assert not adapter._open_draft_by_chat
        frames = [frame for frame in connector.sent if frame["op"] == "draft"]
        assert frames[-1]["final"] is True
        assert frames[-1]["draft_id"] == frames[0]["draft_id"]
        assert frames[-1]["content"] == frames[0]["content"]
        assert not consumer.final_response_sent
        await adapter.send("C1", "Unrelated next-turn notification")
        assert connector.sent[-1]["op"] == "send"
        assert connector.sent[-1]["content"] == "Unrelated next-turn notification"
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_skip_without_a_preview_does_not_close_another_turns_draft():
    from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
    from tests.gateway.relay.test_relay_live_cards import _connected_adapter

    adapter, connector = _connected_adapter(supported_ops=("send", "edit", "typing", "draft"))
    other_metadata = {"message_id": "other-turn"}
    await adapter.send_draft("C1", 123, "Another turn's preview", metadata=other_metadata)
    consumer = GatewayStreamConsumer(adapter, "C1", StreamConsumerConfig(transport="auto", chat_type="dm"))
    consumer.finish_silently()
    await asyncio.wait_for(consumer.run(), 2)

    assert adapter._open_draft_by_chat == {"C1:turn:other-turn": 123}
    assert all(frame["op"] == "draft" and frame["final"] is False for frame in connector.sent)
    # Close the fixture-owned draft and prove teardown left no interception state.
    await adapter.abandon_open_draft("C1", "Another turn's preview", metadata=other_metadata)
    assert not adapter._open_draft_by_chat
