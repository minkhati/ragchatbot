"""
Tests for AIGenerator — verifies it correctly calls CourseSearchTool via tool_manager.
"""
import pytest
from unittest.mock import MagicMock, call

from ai_generator import AIGenerator


# ── helpers ───────────────────────────────────────────────────────────────────

def _tool_use_response(tool_name, tool_input, tool_id="tid_001"):
    block = MagicMock()
    block.type = "tool_use"
    block.id = tool_id
    block.name = tool_name
    block.input = tool_input

    resp = MagicMock()
    resp.stop_reason = "tool_use"
    resp.content = [block]
    return resp


def _text_response(text):
    block = MagicMock()
    block.type = "text"
    block.text = text

    resp = MagicMock()
    resp.stop_reason = "end_turn"
    resp.content = [block]
    return resp


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def generator():
    gen = AIGenerator(api_key="test-key", model="claude-sonnet-4-6")
    gen.client = MagicMock()
    return gen


@pytest.fixture
def tool_defs():
    return [{"name": "search_course_content", "description": "Search course content"}]


# ── 1. tool is called for content queries ─────────────────────────────────────

def test_tool_manager_execute_called_on_tool_use(generator, tool_defs):
    generator.client.messages.create.side_effect = [
        _tool_use_response("search_course_content", {"query": "what is MCP?"}),
        _text_response("MCP is a protocol."),
    ]
    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "MCP result text"

    generator.generate_response(
        query="what is MCP?", tools=tool_defs, tool_manager=tool_manager
    )

    tool_manager.execute_tool.assert_called_once_with(
        "search_course_content", query="what is MCP?"
    )


def test_tool_result_forwarded_in_second_api_call(generator, tool_defs):
    generator.client.messages.create.side_effect = [
        _tool_use_response("search_course_content", {"query": "MCP"}, tool_id="tid_abc"),
        _text_response("Final answer"),
    ]
    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "MCP content from search"

    generator.generate_response(
        query="explain MCP", tools=tool_defs, tool_manager=tool_manager
    )

    second_call = generator.client.messages.create.call_args_list[1]
    messages = second_call[1]["messages"]

    # The last message must be the tool result
    tool_msg = next(
        m for m in messages if m["role"] == "user" and isinstance(m["content"], list)
    )
    result_block = tool_msg["content"][0]
    assert result_block["type"] == "tool_result"
    assert result_block["tool_use_id"] == "tid_abc"
    assert result_block["content"] == "MCP content from search"


def test_two_api_calls_made_when_tool_used(generator, tool_defs):
    generator.client.messages.create.side_effect = [
        _tool_use_response("search_course_content", {"query": "deep learning"}),
        _text_response("Deep learning answer"),
    ]
    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "some content"

    generator.generate_response(
        query="explain deep learning", tools=tool_defs, tool_manager=tool_manager
    )

    assert generator.client.messages.create.call_count == 2


def test_final_text_response_returned(generator, tool_defs):
    generator.client.messages.create.side_effect = [
        _tool_use_response("search_course_content", {"query": "MCP"}),
        _text_response("MCP enables tool use in LLMs."),
    ]
    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "raw search result"

    result = generator.generate_response(
        query="what is MCP?", tools=tool_defs, tool_manager=tool_manager
    )

    assert result == "MCP enables tool use in LLMs."


# ── 2. no tool use for general knowledge ──────────────────────────────────────

def test_single_api_call_for_direct_answer(generator, tool_defs):
    generator.client.messages.create.return_value = _text_response("2 + 2 = 4")

    generator.generate_response(query="what is 2+2?", tools=tool_defs, tool_manager=MagicMock())

    assert generator.client.messages.create.call_count == 1


def test_tool_manager_not_called_on_direct_answer(generator, tool_defs):
    generator.client.messages.create.return_value = _text_response("4")
    tool_manager = MagicMock()

    generator.generate_response(query="2+2?", tools=tool_defs, tool_manager=tool_manager)

    tool_manager.execute_tool.assert_not_called()


def test_tool_manager_not_called_when_tools_none(generator):
    generator.client.messages.create.return_value = _text_response("Direct answer")
    tool_manager = MagicMock()

    generator.generate_response(query="anything", tools=None, tool_manager=tool_manager)

    tool_manager.execute_tool.assert_not_called()


# ── 3. forced final call must NOT include tools (avoid infinite tool loop) ─────

def test_second_call_has_no_tools_key(generator, tool_defs):
    """After both rounds are exhausted, the forced-final call must omit tools."""
    generator.client.messages.create.side_effect = [
        _tool_use_response("search_course_content", {"query": "MCP"}),
        _tool_use_response("search_course_content", {"query": "MCP details"}),
        _text_response("Answer"),
    ]
    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "result"

    generator.generate_response(query="MCP?", tools=tool_defs, tool_manager=tool_manager)

    third_kwargs = generator.client.messages.create.call_args_list[2][1]
    assert "tools" not in third_kwargs


def test_second_call_has_no_tool_choice_key(generator, tool_defs):
    """After both rounds are exhausted, the forced-final call must omit tool_choice."""
    generator.client.messages.create.side_effect = [
        _tool_use_response("search_course_content", {"query": "MCP"}),
        _tool_use_response("search_course_content", {"query": "MCP details"}),
        _text_response("Answer"),
    ]
    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "result"

    generator.generate_response(query="MCP?", tools=tool_defs, tool_manager=tool_manager)

    third_kwargs = generator.client.messages.create.call_args_list[2][1]
    assert "tool_choice" not in third_kwargs


# ── 4. conversation history is included in system prompt ─────────────────────

def test_conversation_history_injected_into_system(generator, tool_defs):
    generator.client.messages.create.return_value = _text_response("Answer")

    generator.generate_response(
        query="follow-up question",
        conversation_history="User: hi\nAssistant: hello",
        tools=tool_defs,
        tool_manager=MagicMock(),
    )

    first_call = generator.client.messages.create.call_args_list[0][1]
    assert "Previous conversation" in first_call["system"]
    assert "User: hi" in first_call["system"]


# ── 6. two-round sequential tool calling ─────────────────────────────────────

def test_two_round_flow_makes_three_api_calls(generator, tool_defs):
    generator.client.messages.create.side_effect = [
        _tool_use_response("get_course_outline", {"course_name": "MCP"}, tool_id="tid_r1"),
        _tool_use_response("search_course_content", {"query": "lesson 4 topic"}, tool_id="tid_r2"),
        _text_response("Final synthesized answer"),
    ]
    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "some result"

    result = generator.generate_response(
        query="find a course covering the same topic as lesson 4 of MCP",
        tools=tool_defs, tool_manager=tool_manager,
    )

    assert generator.client.messages.create.call_count == 3
    assert result == "Final synthesized answer"


def test_tool_manager_called_twice_in_two_round_flow(generator, tool_defs):
    generator.client.messages.create.side_effect = [
        _tool_use_response("get_course_outline", {"course_name": "MCP"}, tool_id="tid_r1"),
        _tool_use_response("search_course_content", {"query": "topic"}, tool_id="tid_r2"),
        _text_response("Final answer"),
    ]
    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "result"

    generator.generate_response(
        query="cross-course question", tools=tool_defs, tool_manager=tool_manager,
    )

    assert tool_manager.execute_tool.call_count == 2


def test_intermediate_call_includes_tools(generator, tool_defs):
    generator.client.messages.create.side_effect = [
        _tool_use_response("get_course_outline", {"course_name": "X"}, tool_id="tid_r1"),
        _tool_use_response("search_course_content", {"query": "topic"}, tool_id="tid_r2"),
        _text_response("Done"),
    ]
    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "result"

    generator.generate_response(query="q", tools=tool_defs, tool_manager=tool_manager)

    second_call_kwargs = generator.client.messages.create.call_args_list[1][1]
    assert "tools" in second_call_kwargs
    assert "tool_choice" in second_call_kwargs


def test_forced_final_call_has_no_tools(generator, tool_defs):
    generator.client.messages.create.side_effect = [
        _tool_use_response("get_course_outline", {"course_name": "X"}, tool_id="tid_r1"),
        _tool_use_response("search_course_content", {"query": "topic"}, tool_id="tid_r2"),
        _text_response("Done"),
    ]
    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "result"

    generator.generate_response(query="q", tools=tool_defs, tool_manager=tool_manager)

    third_call_kwargs = generator.client.messages.create.call_args_list[2][1]
    assert "tools" not in third_call_kwargs
    assert "tool_choice" not in third_call_kwargs


def test_early_termination_on_no_tool_use_in_round2(generator, tool_defs):
    generator.client.messages.create.side_effect = [
        _tool_use_response("search_course_content", {"query": "MCP"}, tool_id="tid_r1"),
        _text_response("Sufficient answer after one tool"),
    ]
    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "result"

    result = generator.generate_response(
        query="question", tools=tool_defs, tool_manager=tool_manager,
    )

    assert generator.client.messages.create.call_count == 2
    assert result == "Sufficient answer after one tool"


def test_tool_error_does_not_raise(generator, tool_defs):
    generator.client.messages.create.side_effect = [
        _tool_use_response("search_course_content", {"query": "MCP"}, tool_id="tid_r1"),
        _text_response("Fallback answer"),
    ]
    tool_manager = MagicMock()
    tool_manager.execute_tool.side_effect = RuntimeError("ChromaDB unavailable")

    result = generator.generate_response(
        query="question", tools=tool_defs, tool_manager=tool_manager,
    )

    assert isinstance(result, str)


def test_tool_error_forwarded_in_tool_result(generator, tool_defs):
    generator.client.messages.create.side_effect = [
        _tool_use_response("search_course_content", {"query": "MCP"}, tool_id="tid_r1"),
        _text_response("Fallback answer"),
    ]
    tool_manager = MagicMock()
    tool_manager.execute_tool.side_effect = RuntimeError("ChromaDB unavailable")

    generator.generate_response(query="question", tools=tool_defs, tool_manager=tool_manager)

    second_call = generator.client.messages.create.call_args_list[1][1]
    messages = second_call["messages"]
    tool_msg = next(m for m in messages if m["role"] == "user" and isinstance(m["content"], list))
    assert "ChromaDB unavailable" in tool_msg["content"][0]["content"]


def test_accumulated_messages_in_final_call(generator, tool_defs):
    generator.client.messages.create.side_effect = [
        _tool_use_response("get_course_outline", {"course_name": "X"}, tool_id="tid_r1"),
        _tool_use_response("search_course_content", {"query": "topic"}, tool_id="tid_r2"),
        _text_response("Done"),
    ]
    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "result"

    generator.generate_response(query="q", tools=tool_defs, tool_manager=tool_manager)

    third_call_messages = generator.client.messages.create.call_args_list[2][1]["messages"]
    roles = [m["role"] for m in third_call_messages]
    assert roles == ["user", "assistant", "user", "assistant", "user"]


# ── 5. edge case: empty content list causes AttributeError ────────────────────

def test_empty_content_list_in_final_response_raises(generator, tool_defs):
    """
    If the final API response has an empty content list, content[0].text raises
    IndexError — this surfaces as HTTP 500 / 'query failed' in the frontend.
    """
    tool_use_resp = _tool_use_response("search_course_content", {"query": "MCP"})
    bad_final = MagicMock()
    bad_final.stop_reason = "end_turn"
    bad_final.content = []  # ← empty

    generator.client.messages.create.side_effect = [tool_use_resp, bad_final]
    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "result"

    with pytest.raises((IndexError, AttributeError)):
        generator.generate_response(
            query="what is MCP?", tools=tool_defs, tool_manager=tool_manager
        )
