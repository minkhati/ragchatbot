"""
Tests for how the RAG system handles content-query questions end-to-end.
Exercises the full pipeline: RAGSystem.query → AIGenerator → ToolManager →
CourseSearchTool → VectorStore.
"""
import pytest
from unittest.mock import MagicMock, patch

from vector_store import SearchResults
from search_tools import CourseSearchTool, ToolManager
from ai_generator import AIGenerator
from rag_system import RAGSystem
from config import Config


# ── helpers ───────────────────────────────────────────────────────────────────

def _search_results(docs, metadata):
    return SearchResults(
        documents=docs,
        metadata=metadata,
        distances=[0.1] * len(docs),
    )


def _tool_block(tool_name, tool_input, tool_id="tid_001"):
    block = MagicMock()
    block.type = "tool_use"
    block.id = tool_id
    block.name = tool_name
    block.input = tool_input
    return block


def _text_block(text):
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _tool_use_response(tool_name, tool_input, tool_id="tid_001"):
    resp = MagicMock()
    resp.stop_reason = "tool_use"
    resp.content = [_tool_block(tool_name, tool_input, tool_id)]
    return resp


def _text_response(text):
    resp = MagicMock()
    resp.stop_reason = "end_turn"
    resp.content = [_text_block(text)]
    return resp


# ── fixture: RAGSystem with mocked infrastructure ─────────────────────────────

@pytest.fixture
def rag(tmp_path):
    """
    A RAGSystem where ChromaDB and the Anthropic client are replaced with mocks,
    but all routing logic (ToolManager, CourseSearchTool, AIGenerator) is real.
    """
    cfg = Config(ANTHROPIC_API_KEY="test-key", CHROMA_PATH=str(tmp_path / "chroma"))

    with patch("rag_system.VectorStore") as MockVS, \
         patch("rag_system.DocumentProcessor"), \
         patch("rag_system.SessionManager") as MockSM:

        mock_vs = MockVS.return_value
        mock_vs.search.return_value = SearchResults(documents=[], metadata=[], distances=[])
        mock_vs.get_lesson_link.return_value = None

        mock_sm = MockSM.return_value
        mock_sm.get_conversation_history.return_value = None

        system = RAGSystem(cfg)

    # Replace the Anthropic client inside the real AIGenerator
    mock_client = MagicMock()
    system.ai_generator.client = mock_client

    # Keep a reference for test assertions
    system._mock_vs = mock_vs
    system._mock_client = mock_client

    return system


# ── 1. search_course_content tool is registered ───────────────────────────────

def test_search_course_content_tool_registered(rag):
    names = [t["name"] for t in rag.tool_manager.get_tool_definitions()]
    assert "search_course_content" in names


def test_get_course_outline_tool_registered(rag):
    names = [t["name"] for t in rag.tool_manager.get_tool_definitions()]
    assert "get_course_outline" in names


# ── 2. content query triggers VectorStore via tool pipeline ───────────────────

def test_content_query_reaches_vector_store(rag):
    """Full pipeline: RAGSystem.query → AI tool use → CourseSearchTool → VectorStore."""
    rag._mock_client.messages.create.side_effect = [
        _tool_use_response("search_course_content", {"query": "what is MCP?"}),
        _text_response("MCP explanation"),
    ]

    response, _ = rag.query("what is MCP?")

    rag._mock_vs.search.assert_called_once()
    assert response == "MCP explanation"


def test_content_query_passes_correct_query_to_vector_store(rag):
    rag._mock_client.messages.create.side_effect = [
        _tool_use_response("search_course_content", {"query": "deep learning intro"}),
        _text_response("Answer"),
    ]

    rag.query("explain deep learning")

    call_kwargs = rag._mock_vs.search.call_args[1]
    assert call_kwargs["query"] == "deep learning intro"


def test_content_query_with_course_filter_reaches_vector_store(rag):
    rag._mock_client.messages.create.side_effect = [
        _tool_use_response(
            "search_course_content",
            {"query": "agents", "course_name": "MCP Course"},
        ),
        _text_response("Agents explanation"),
    ]

    rag.query("explain agents in MCP Course")

    call_kwargs = rag._mock_vs.search.call_args[1]
    assert call_kwargs["course_name"] == "MCP Course"


# ── 3. search results flow back to Claude ─────────────────────────────────────

def test_search_results_are_included_in_second_api_call(rag):
    rag._mock_vs.search.return_value = _search_results(
        docs=["MCP is a protocol for tools."],
        metadata=[{"course_title": "MCP Course", "lesson_number": 1}],
    )
    rag._mock_client.messages.create.side_effect = [
        _tool_use_response("search_course_content", {"query": "MCP"}),
        _text_response("Final answer"),
    ]

    rag.query("what is MCP?")

    second_call = rag._mock_client.messages.create.call_args_list[1]
    messages = second_call[1]["messages"]

    # Find the tool-result message
    tool_msg = next(
        m for m in messages
        if m["role"] == "user" and isinstance(m["content"], list)
    )
    result_content = tool_msg["content"][0]["content"]
    assert "MCP Course" in result_content
    assert "MCP is a protocol for tools." in result_content


# ── 4. RAG returns proper response (not 'query failed') ───────────────────────

def test_rag_does_not_return_query_failed_on_content_question(rag):
    rag._mock_client.messages.create.side_effect = [
        _tool_use_response("search_course_content", {"query": "neural networks"}),
        _text_response("Neural networks are composed of layers."),
    ]

    response, _ = rag.query("explain neural networks")

    assert "query failed" not in response.lower()


def test_rag_returns_response_when_search_empty(rag):
    """Claude should still answer even if search returns nothing."""
    rag._mock_vs.search.return_value = SearchResults(documents=[], metadata=[], distances=[])
    rag._mock_client.messages.create.side_effect = [
        _tool_use_response("search_course_content", {"query": "unknown topic"}),
        _text_response("I couldn't find relevant content."),
    ]

    response, _ = rag.query("tell me about unknown topic")

    assert isinstance(response, str)
    assert len(response) > 0


# ── 5. unhandled VectorStore exception propagates (reveals missing error guard) ──

def test_vector_store_exception_does_not_propagate(rag):
    """
    CourseSearchTool.execute() catches VectorStore exceptions and returns an error
    string so the pipeline completes and the frontend never sees 'query failed'.
    """
    rag._mock_vs.search.side_effect = Exception("ChromaDB internal error")
    rag._mock_client.messages.create.side_effect = [
        _tool_use_response("search_course_content", {"query": "MCP"}),
        _text_response("Handled gracefully"),
    ]

    response, _ = rag.query("what is MCP?")

    assert response == "Handled gracefully"


# ── 6. vector store n_results > collection size triggers the search-error path ──

def test_search_error_from_small_collection_is_handled(rag):
    """
    ChromaDB raises when n_results > number of indexed documents.
    VectorStore.search() catches this; CourseSearchTool must pass the error
    string back to Claude instead of letting it propagate.
    """
    rag._mock_vs.search.return_value = SearchResults.empty(
        "Search error: Number of requested results 5 is greater than number of elements in index 2"
    )
    rag._mock_client.messages.create.side_effect = [
        _tool_use_response("search_course_content", {"query": "MCP"}),
        _text_response("Based on available info..."),
    ]

    response, _ = rag.query("what is MCP?")

    # Should complete successfully, not raise
    assert isinstance(response, str)


# ── 7. config model name is a valid Claude 4 model ───────────────────────────

def test_config_model_name_matches_known_claude4_format():
    """
    The configured model must be a valid Claude 4 model ID.
    An invalid model causes every API call to fail → all queries return 'query failed'.
    """
    from config import config
    valid_models = {
        "claude-opus-4-7",
        "claude-sonnet-4-6",
        "claude-haiku-4-5-20251001",
        # legacy claude-3 family still valid
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
    }
    assert config.ANTHROPIC_MODEL in valid_models, (
        f"Model '{config.ANTHROPIC_MODEL}' is not a recognised Claude model ID. "
        "Update config.ANTHROPIC_MODEL to a valid model."
    )
