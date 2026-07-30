from __future__ import annotations
import asyncio

import json

import pytest

from litecoder.tasks.message_bus import MessageBus, TeamMessage


@pytest.fixture
def mailbox_root(tmp_path):
    return tmp_path / "mailboxes"


@pytest.fixture
def bus(mailbox_root):
    return MessageBus(mailbox_root)


@pytest.mark.asyncio
async def test_receive_reads_all_and_removes_mailbox(bus, mailbox_root) -> None:
    await bus.send("agent-1", TeamMessage("lead", "agent-1", "one"))
    await bus.send("agent-1", TeamMessage("lead", "agent-1", "two"))

    received = await bus.receive("agent-1")

    assert [message.body for message in received] == ["one", "two"]
    assert not (mailbox_root / "agent-1.jsonl").exists()
    assert await bus.receive("agent-1") == []


@pytest.mark.asyncio
async def test_send_appends_one_json_object_per_line(bus, mailbox_root) -> None:
    first = TeamMessage("lead", "agent-1", "héllo")
    second = TeamMessage("lead", "agent-1", "world")
    await bus.send("agent-1", first)
    await bus.send("agent-1", second)

    lines = (mailbox_root / "agent-1.jsonl").read_text(encoding="utf-8").splitlines()

    assert len(lines) == 2
    assert [json.loads(line)["body"] for line in lines] == ["héllo", "world"]
    assert json.loads(lines[0])["id"] != json.loads(lines[1])["id"]


@pytest.mark.asyncio
async def test_agent_id_cannot_escape_mailbox_root(bus) -> None:
    with pytest.raises(ValueError):
        await bus.send("../outside", TeamMessage("lead", "../outside", "nope"))


@pytest.mark.asyncio
async def test_receive_is_at_most_once_under_concurrency(bus, mailbox_root) -> None:
    await bus.send("agent-1", TeamMessage("lead", "agent-1", "one"))

    first, second = await pytest.importorskip("asyncio").gather(
        bus.receive("agent-1"), bus.receive("agent-1")
    )

    assert sorted(message.body for message in first + second) == ["one"]
    assert not (mailbox_root / "agent-1.jsonl").exists()

@pytest.mark.asyncio
async def test_wait_observes_mailbox_written_by_another_bus_instance(
    mailbox_root,
) -> None:
    waiting_bus = MessageBus(mailbox_root)
    sending_bus = MessageBus(mailbox_root)
    waiter = asyncio.create_task(waiting_bus.wait_for_messages("agent-1"))

    await asyncio.sleep(0.05)
    await sending_bus.send(
        "agent-1", TeamMessage("lead", "agent-1", "cross-instance")
    )
    await asyncio.wait_for(waiter, timeout=1.0)

    received = await waiting_bus.receive("agent-1")
    assert [message.body for message in received] == ["cross-instance"]


@pytest.mark.asyncio
async def test_receive_quarantines_malformed_lines_and_delivers_valid_messages(
    bus, mailbox_root
) -> None:
    path = mailbox_root / "agent-1.jsonl"
    valid = TeamMessage("lead", "agent-1", "deliver")
    path.parent.mkdir()
    path.write_text("not-json\n" + json.dumps(valid.to_dict()) + "\n", encoding="utf-8")

    received = await bus.receive("agent-1")

    assert [message.body for message in received] == ["deliver"]
    corrupt = tuple(mailbox_root.glob("agent-1.corrupt-*.jsonl"))
    assert len(corrupt) == 1 and corrupt[0].read_text(encoding="utf-8") == "not-json\n"
