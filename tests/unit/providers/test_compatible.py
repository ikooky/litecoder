from __future__ import annotations

import pytest

from litecoder.providers.compatible import CompatibleProvider
from litecoder.providers.models import ModelRequest, StopReason
from tests.unit.providers.conftest import StubCall


def request() -> ModelRequest:
    return ModelRequest(
        model="request-model",
        system=[{"type": "text", "text": "system"}],
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": "hello"}],
            }
        ],
        tools=[
            {
                "name": "read_file",
                "description": "Read a file",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            }
        ],
        max_tokens=512,
    )


def provider(
    completion_call: StubCall,
    responses_call: StubCall,
    *,
    api_style: str = "openai-chat-completions",
    base_url: str | None = None,
) -> CompatibleProvider:
    return CompatibleProvider(
        completion=completion_call,
        responses=responses_call,
        model="configured-model",
        api_style=api_style,  # type: ignore[arg-type]
        api_key="secret",
        base_url=base_url,
    )


@pytest.mark.parametrize(
    ("api_style", "custom_provider", "expected_api_base"),
    [
        ("anthropic-messages", "anthropic", "https://gateway.invalid"),
        (
            "openai-chat-completions",
            "openai",
            "https://gateway.invalid/v1",
        ),
    ],
)
async def test_completion_protocols_use_completion_call(
    completion_call: StubCall,
    responses_call: StubCall,
    api_style: str,
    custom_provider: str,
    expected_api_base: str,
) -> None:
    completion_call.events = [
        {
            "id": "req-1",
            "choices": [
                {"index": 0, "delta": {"content": "ok"}, "finish_reason": None}
            ],
        },
        {
            "id": "req-1",
            "choices": [
                {"index": 0, "delta": {}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
        },
    ]

    events = [
        event
        async for event in provider(
            completion_call,
            responses_call,
            api_style=api_style,
            base_url="https://gateway.invalid/v1",
        ).stream(request())
    ]

    assert responses_call.calls == []
    assert len(completion_call.calls) == 1
    call = completion_call.calls[0]
    assert call["model"] == "configured-model"
    assert call["custom_llm_provider"] == custom_provider
    assert call["api_base"] == expected_api_base
    assert call["max_retries"] == 0
    assert call["stream"] is True
    assert call["messages"][0] == {
        "role": "system",
        "content": [{"type": "text", "text": "system"}],
    }
    assert call["tools"][0]["function"]["name"] == "read_file"
    assert [event.type for event in events] == [
        "response.request_id",
        "content.started",
        "content.delta",
        "text.delta",
        "usage",
        "content.completed",
        "response.completed",
    ]
    assert events[-1].stop_reason is StopReason.END_TURN
    assert events[-1].usage is not None
    assert events[-1].usage.total_tokens == 4


@pytest.mark.parametrize(
    "base_url",
    [
        "https://gateway.invalid",
        "https://gateway.invalid/",
        "https://gateway.invalid/v1",
        "https://gateway.invalid/v1/",
    ],
)
async def test_anthropic_base_url_accepts_optional_v1_suffix(
    completion_call: StubCall,
    responses_call: StubCall,
    base_url: str,
) -> None:
    completion_call.events = [
        {
            "choices": [
                {"index": 0, "delta": {}, "finish_reason": "stop"}
            ]
        }
    ]

    _ = [
        event
        async for event in provider(
            completion_call,
            responses_call,
            api_style="anthropic-messages",
            base_url=base_url,
        ).stream(request())
    ]

    assert completion_call.calls[0]["api_base"] == "https://gateway.invalid"


async def test_completion_accepts_sdk_metadata_and_usage_after_finish(
    completion_call: StubCall, responses_call: StubCall
) -> None:
    completion_call.events = [
        {
            "id": "req-main",
            "choices": [
                {"index": 0, "delta": {"content": "ok"}, "finish_reason": None}
            ],
        },
        {
            "id": "req-main",
            "choices": [
                {"index": 0, "delta": {}, "finish_reason": "stop"}
            ],
        },
        {"id": "synthetic-id", "choices": []},
        {
            "id": "",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": None,
                        "role": None,
                        "tool_calls": None,
                    },
                    "finish_reason": None,
                }
            ],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2},
        },
    ]

    events = [
        event
        async for event in provider(completion_call, responses_call).stream(request())
    ]

    assert not any(event.type == "provider.error" for event in events)
    assert events[-1].type == "response.completed"
    assert events[-1].usage is not None
    assert events[-1].usage.total_tokens == 6


async def test_completion_stream_normalizes_tool_calls(
    completion_call: StubCall, responses_call: StubCall
) -> None:
    completion_call.events = [
        {
            "id": "req-tool",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-1",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path":',
                                },
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "req-tool",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "",
                                "function": {
                                    "name": "",
                                    "arguments": '"README.md"}',
                                },
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        },
    ]

    events = [
        event
        async for event in provider(completion_call, responses_call).stream(request())
    ]

    completed = next(event for event in events if event.type == "tool_call.completed")
    assert completed.tool_call is not None
    assert completed.tool_call.call_id == "call-1"
    assert completed.tool_call.input == {"path": "README.md"}
    assert events[-1].stop_reason is StopReason.TOOL_USE


async def test_completion_synthesizes_empty_provider_tool_call_id(
    completion_call: StubCall, responses_call: StubCall
) -> None:
    completion_call.events = [
        {
            "id": "req-tool",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path":',
                                },
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "req-tool",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": None,
                                "function": {"arguments": '"solution.py"}'},
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        },
    ]

    events = [
        event
        async for event in provider(completion_call, responses_call).stream(request())
    ]

    completed = next(event for event in events if event.type == "tool_call.completed")
    assert completed.tool_call is not None
    assert completed.tool_call.call_id.startswith("call_litecoder_")
    assert completed.tool_call.name == "read_file"
    assert completed.tool_call.input == {"path": "solution.py"}
    assert not any(event.type == "provider.error" for event in events)


async def test_completion_rejects_non_string_provider_tool_call_id(
    completion_call: StubCall, responses_call: StubCall
) -> None:
    completion_call.events = [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": 7,
                                "function": {
                                    "name": "read_file",
                                    "arguments": "{}",
                                },
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
    ]

    events = [
        event
        async for event in provider(completion_call, responses_call).stream(request())
    ]

    assert events[-1].type == "provider.error"
    assert events[-1].error is not None
    assert events[-1].error.details["provider_data_reason"] == (
        "provider tool call id is invalid"
    )


async def test_completion_ignores_empty_identity_in_argument_deltas(
    completion_call: StubCall, responses_call: StubCall
) -> None:
    completion_call.events = [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-read",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path":',
                                },
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "",
                                "function": {
                                    "name": "",
                                    "arguments": '"solution.py"}',
                                },
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
    ]

    events = [
        event
        async for event in provider(completion_call, responses_call).stream(request())
    ]

    completed = next(event for event in events if event.type == "tool_call.completed")
    assert completed.tool_call is not None
    assert completed.tool_call.call_id == "call-read"
    assert completed.tool_call.name == "read_file"
    assert completed.tool_call.input == {"path": "solution.py"}
    assert not any(event.type == "provider.error" for event in events)


async def test_completion_stream_rejects_conflicting_nonempty_tool_identity(
    completion_call: StubCall, responses_call: StubCall
) -> None:
    completion_call.events = [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-1",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path":',
                                },
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-2",
                                "function": {"arguments": '"README.md"}'},
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
    ]

    events = [
        event
        async for event in provider(completion_call, responses_call).stream(request())
    ]

    assert events[-1].type == "provider.error"
    assert events[-1].error is not None
    assert events[-1].error.details == {
        "provider_error_type": "invalid_provider_data",
        "provider_data_reason": "conflicting provider tool call id",
    }


async def test_anthropic_completion_preserves_thinking_signature(
    completion_call: StubCall, responses_call: StubCall
) -> None:
    completion_call.events = [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "reasoning_content": "considering",
                        "thinking_blocks": [
                            {"type": "thinking", "thinking": "considering"}
                        ],
                    },
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "reasoning_content": "",
                        "thinking_blocks": [
                            {
                                "type": "thinking",
                                "thinking": "considering",
                                "signature": "signed",
                            }
                        ],
                    },
                    "finish_reason": "stop",
                }
            ]
        },
    ]

    events = [
        event
        async for event in provider(
            completion_call,
            responses_call,
            api_style="anthropic-messages",
        ).stream(request())
    ]

    block = next(
        event.block
        for event in events
        if event.type == "content.completed" and event.block
    )
    assert block == {
        "type": "thinking",
        "thinking": "considering",
        "signature": "signed",
    }


async def test_completion_replays_standard_history(
    completion_call: StubCall, responses_call: StubCall
) -> None:
    completion_call.events = [
        {
            "choices": [
                {"index": 0, "delta": {}, "finish_reason": "stop"}
            ]
        }
    ]
    replay = ModelRequest(
        model="ignored",
        system=[],
        messages=[
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "work", "signature": "sig"},
                    {
                        "type": "tool_call",
                        "call_id": "call-1",
                        "name": "read_file",
                        "input": {"path": "a.py"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_call_id": "call-1",
                        "content": "contents",
                    }
                ],
            },
        ],
        tools=[],
        max_tokens=10,
    )

    _ = [
        event
        async for event in provider(completion_call, responses_call).stream(replay)
    ]

    messages = completion_call.calls[0]["messages"]
    assert messages[0]["content"] == []
    assert messages[0]["reasoning_content"] == "work"
    assert messages[0]["tool_calls"][0]["id"] == "call-1"
    assert messages[1] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": "contents",
    }


async def test_provider_reasoning_content_enables_proactive_replay(
    completion_call: StubCall, responses_call: StubCall
) -> None:
    completion_call.events = [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"reasoning_content": "work"},
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "done"},
                    "finish_reason": "stop",
                }
            ]
        },
    ]
    compatible = provider(completion_call, responses_call)
    _ = [event async for event in compatible.stream(request())]

    completion_call.events = [
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
    ]
    follow_up = ModelRequest(
        model="ignored",
        system=[],
        messages=[
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "previous answer"}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "continue"}],
            },
        ],
        tools=[],
        max_tokens=10,
    )
    _ = [event async for event in compatible.stream(follow_up)]

    assert len(completion_call.calls) == 2
    assert completion_call.calls[1]["messages"][0]["reasoning_content"] == " "


async def test_reasoning_history_proactively_fills_other_assistant_turns(
    completion_call: StubCall, responses_call: StubCall
) -> None:
    completion_call.events = [
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
    ]
    replay = ModelRequest(
        model="ignored",
        system=[],
        messages=[
            {
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": "work"}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "previous answer"}],
            },
        ],
        tools=[],
        max_tokens=10,
    )

    _ = [
        event
        async for event in provider(completion_call, responses_call).stream(replay)
    ]

    messages = completion_call.calls[0]["messages"]
    assert messages[0]["reasoning_content"] == "work"
    assert messages[1]["reasoning_content"] == " "


async def test_provider_specific_reasoning_content_enables_replay(
    completion_call: StubCall, responses_call: StubCall
) -> None:
    completion_call.events = [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "provider_specific_fields": {
                            "reasoning_content": "provider work"
                        }
                    },
                    "finish_reason": None,
                }
            ]
        },
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]
    compatible = provider(completion_call, responses_call)
    _ = [event async for event in compatible.stream(request())]

    completion_call.events = [
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
    ]
    follow_up = ModelRequest(
        model="ignored",
        system=[],
        messages=[
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "previous answer"}],
            }
        ],
        tools=[],
        max_tokens=10,
    )
    _ = [event async for event in compatible.stream(follow_up)]

    assert completion_call.calls[1]["messages"][0]["reasoning_content"] == " "


async def test_anthropic_replay_preserves_native_thinking_block(
    completion_call: StubCall, responses_call: StubCall
) -> None:
    completion_call.events = [
        {
            "choices": [
                {"index": 0, "delta": {}, "finish_reason": "stop"}
            ]
        }
    ]
    replay = ModelRequest(
        model="ignored",
        system=[],
        messages=[
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "work", "signature": "sig"}
                ],
            }
        ],
        tools=[],
        max_tokens=10,
    )

    _ = [
        event
        async for event in provider(
            completion_call,
            responses_call,
            api_style="anthropic-messages",
        ).stream(replay)
    ]

    assert completion_call.calls[0]["messages"][0]["content"] == [
        {"type": "thinking", "thinking": "work", "signature": "sig"}
    ]


async def test_responses_protocol_uses_responses_call(
    completion_call: StubCall, responses_call: StubCall
) -> None:
    responses_call.events = [
        {"type": "response.created", "response": {"id": "resp-1"}},
        {
            "type": "response.output_text.delta",
            "response_id": "resp-1",
            "output_index": 0,
            "content_index": 0,
            "delta": "hello",
        },
        {
            "type": "response.completed",
            "response": {
                "id": "resp-1",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "hello"}],
                    }
                ],
                "usage": {"input_tokens": 5, "output_tokens": 2},
            },
        },
    ]

    events = [
        event
        async for event in provider(
            completion_call,
            responses_call,
            api_style="openai-responses",
            base_url="https://responses.invalid/v1",
        ).stream(request())
    ]

    assert completion_call.calls == []
    assert len(responses_call.calls) == 1
    call = responses_call.calls[0]
    assert call["model"] == "configured-model"
    assert call["custom_llm_provider"] == "openai"
    assert call["api_base"] == "https://responses.invalid/v1"
    assert call["instructions"] == "system"
    assert call["input"] == [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "hello"}],
        }
    ]
    assert call["tools"][0]["name"] == "read_file"
    assert events[-1].stop_reason is StopReason.END_TURN
    assert events[-1].usage is not None
    assert events[-1].usage.total_tokens == 7


async def test_responses_stream_normalizes_function_calls(
    completion_call: StubCall, responses_call: StubCall
) -> None:
    responses_call.events = [
        {"type": "response.created", "response": {"id": "resp-tool"}},
        {
            "type": "response.output_item.added",
            "response_id": "resp-tool",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "id": "fc-1",
                "call_id": "call-1",
                "name": "read_file",
                "arguments": '{"path":"README.md"}',
            },
        },
        {
            "type": "response.function_call_arguments.delta",
            "response_id": "resp-tool",
            "output_index": 0,
            "delta": '{"path":"README.md"}',
        },
        {
            "type": "response.completed",
            "response": {
                "id": "resp-tool",
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "id": "fc-1",
                        "call_id": "call-1",
                        "name": "read_file",
                        "arguments": '{"path":"README.md"}',
                    }
                ],
                "usage": {"input_tokens": 2, "output_tokens": 3},
            },
        },
    ]

    events = [
        event
        async for event in provider(
            completion_call,
            responses_call,
            api_style="openai-responses",
        ).stream(request())
    ]

    completed = next(event for event in events if event.type == "tool_call.completed")
    assert completed.tool_call is not None
    assert completed.tool_call.call_id == "call-1"
    assert completed.tool_call.input == {"path": "README.md"}
    assert events[-1].stop_reason is StopReason.TOOL_USE


@pytest.mark.parametrize("fallback_event", ["done", "terminal"])
async def test_responses_function_call_uses_full_arguments_without_deltas(
    completion_call: StubCall,
    responses_call: StubCall,
    fallback_event: str,
) -> None:
    function_call = {
        "type": "function_call",
        "id": "fc-1",
        "call_id": "call-1",
        "name": "read_file",
        "arguments": '{"path":"README.md"}',
    }
    response_output = [function_call] if fallback_event == "terminal" else []
    responses_call.events = [
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": function_call,
        }
    ]
    if fallback_event == "done":
        responses_call.events.append(
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": function_call,
            }
        )
    responses_call.events.append(
        {
            "type": "response.completed",
            "response": {
                "id": "resp-tool",
                "status": "completed",
                "output": response_output,
                "usage": {"input_tokens": 2, "output_tokens": 3},
            },
        }
    )

    events = [
        event
        async for event in provider(
            completion_call,
            responses_call,
            api_style="openai-responses",
        ).stream(request())
    ]

    completed = next(event for event in events if event.type == "tool_call.completed")
    assert completed.tool_call is not None
    assert completed.tool_call.input == {"path": "README.md"}


async def test_responses_reasoning_item_is_persisted_and_replayed(
    completion_call: StubCall, responses_call: StubCall
) -> None:
    reasoning_item = {
        "type": "reasoning",
        "id": "rs-1",
        "summary": [{"type": "summary_text", "text": "plan"}],
        "encrypted_content": "encrypted",
        "status": "completed",
    }
    responses_call.events = [
        {
            "type": "response.reasoning_summary_text.delta",
            "output_index": 0,
            "summary_index": 0,
            "delta": "plan",
        },
        {
            "type": "response.completed",
            "response": {
                "id": "resp-reasoning",
                "status": "completed",
                "output": [reasoning_item],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        },
    ]

    events = [
        event
        async for event in provider(
            completion_call,
            responses_call,
            api_style="openai-responses",
        ).stream(request())
    ]
    reasoning = next(
        event.block
        for event in events
        if event.type == "content.completed"
        and event.block
        and event.block.get("type") == "reasoning"
    )
    assert reasoning["item"] == {
        "type": "reasoning",
        "id": "rs-1",
        "summary": [{"type": "summary_text", "text": "plan"}],
        "encrypted_content": "encrypted",
    }

    responses_call.events = [
        {
            "type": "response.completed",
            "response": {
                "id": "resp-next",
                "status": "completed",
                "output": [],
                "usage": {"input_tokens": 1, "output_tokens": 0},
            },
        }
    ]
    replay = ModelRequest(
        model="ignored",
        system=[],
        messages=[{"role": "assistant", "content": [reasoning]}],
        tools=[],
        max_tokens=10,
    )
    _ = [
        event
        async for event in provider(
            completion_call,
            responses_call,
            api_style="openai-responses",
        ).stream(replay)
    ]
    assert responses_call.calls[-1]["input"][0] == reasoning["item"]


async def test_responses_history_preserves_standard_item_order(
    completion_call: StubCall, responses_call: StubCall
) -> None:
    responses_call.events = [
        {
            "type": "response.completed",
            "response": {
                "id": "resp-next",
                "status": "completed",
                "output": [],
                "usage": {"input_tokens": 1, "output_tokens": 0},
            },
        }
    ]
    replay = ModelRequest(
        model="ignored",
        system=[],
        messages=[
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "before"},
                    {
                        "type": "reasoning",
                        "text": "plan",
                        "item": {
                            "type": "reasoning",
                            "id": "rs-1",
                            "summary": [],
                        },
                    },
                    {
                        "type": "tool_call",
                        "call_id": "call-1",
                        "name": "read_file",
                        "input": {"path": "a.py"},
                    },
                    {"type": "text", "text": "after"},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "prefix"},
                    {
                        "type": "tool_result",
                        "tool_call_id": "call-1",
                        "content": "contents",
                    },
                    {"type": "text", "text": "suffix"},
                ],
            },
        ],
        tools=[],
        max_tokens=10,
    )

    _ = [
        event
        async for event in provider(
            completion_call,
            responses_call,
            api_style="openai-responses",
        ).stream(replay)
    ]

    assert [item["type"] for item in responses_call.calls[0]["input"]] == [
        "message",
        "reasoning",
        "function_call",
        "message",
        "message",
        "function_call_output",
        "message",
    ]


async def test_provider_errors_are_normalized_without_exposing_message(
    completion_call: StubCall, responses_call: StubCall
) -> None:
    completion_call.setup_error = RuntimeError("secret upstream detail")

    events = [
        event
        async for event in provider(completion_call, responses_call).stream(request())
    ]

    assert len(events) == 1
    assert events[0].type == "provider.error"
    assert events[0].error is not None
    assert "secret upstream detail" not in str(events[0].error)


async def test_invalid_completion_stream_keeps_adapter_reason(
    completion_call: StubCall, responses_call: StubCall
) -> None:
    completion_call.events = [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": None,
                                "id": "call-1",
                                "function": {
                                    "name": "read_file",
                                    "arguments": "",
                                },
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        }
    ]

    events = [
        event
        async for event in provider(completion_call, responses_call).stream(request())
    ]

    error = events[-1].error
    assert events[-1].type == "provider.error"
    assert error is not None
    assert error.code.value == "provider_invalid_response"
    assert error.retryable is True
    assert error.details == {
        "provider_error_type": "invalid_provider_data",
        "provider_data_reason": "provider content index is invalid",
    }


async def test_missing_terminal_event_is_rejected(
    completion_call: StubCall, responses_call: StubCall
) -> None:
    responses_call.events = [
        {"type": "response.created", "response": {"id": "resp-1"}}
    ]

    events = [
        event
        async for event in provider(
            completion_call,
            responses_call,
            api_style="openai-responses",
        ).stream(request())
    ]

    assert events[-1].type == "provider.error"
    assert events[-1].error is not None
    assert events[-1].error.details["provider_error_type"] == "missing_response_completed"
