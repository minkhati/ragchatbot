"""
Tests for CourseSearchTool.execute() — verifies outputs and VectorStore integration.
"""
import pytest
from unittest.mock import MagicMock

from search_tools import CourseSearchTool
from vector_store import SearchResults


# ── helpers ──────────────────────────────────────────────────────────────────

def _results(docs, metadata):
    return SearchResults(
        documents=docs,
        metadata=metadata,
        distances=[0.1] * len(docs),
    )


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_store():
    store = MagicMock()
    store.get_lesson_link.return_value = "http://example.com/lesson"
    return store


@pytest.fixture
def tool(mock_store):
    return CourseSearchTool(mock_store)


# ── 1. happy-path output format ───────────────────────────────────────────────

def test_execute_returns_course_and_lesson_header(tool, mock_store):
    mock_store.search.return_value = _results(
        docs=["MCP enables tools to connect to LLMs."],
        metadata=[{"course_title": "MCP Course", "lesson_number": 1}],
    )
    result = tool.execute(query="What is MCP?")

    assert "MCP Course" in result
    assert "Lesson 1" in result
    assert "MCP enables tools to connect to LLMs." in result


def test_execute_multiple_results_are_separated(tool, mock_store):
    mock_store.search.return_value = _results(
        docs=["First chunk.", "Second chunk."],
        metadata=[
            {"course_title": "Course A", "lesson_number": 1},
            {"course_title": "Course B", "lesson_number": 2},
        ],
    )
    result = tool.execute(query="something")

    assert "Course A" in result
    assert "Course B" in result
    assert "First chunk." in result
    assert "Second chunk." in result


def test_execute_result_without_lesson_number(tool, mock_store):
    mock_store.search.return_value = _results(
        docs=["Intro text."],
        metadata=[{"course_title": "Intro Course"}],  # no lesson_number key
    )
    result = tool.execute(query="intro")

    assert "Intro Course" in result
    assert "Intro text." in result


# ── 2. empty / no-match responses ────────────────────────────────────────────

def test_execute_returns_no_content_found_when_empty(tool, mock_store):
    mock_store.search.return_value = SearchResults(documents=[], metadata=[], distances=[])

    result = tool.execute(query="unknown topic")

    assert "No relevant content found" in result


def test_execute_no_content_message_includes_course_filter(tool, mock_store):
    mock_store.search.return_value = SearchResults(documents=[], metadata=[], distances=[])

    result = tool.execute(query="something", course_name="MCP Course")

    assert "No relevant content found" in result
    assert "MCP Course" in result


def test_execute_no_content_message_includes_lesson_filter(tool, mock_store):
    mock_store.search.return_value = SearchResults(documents=[], metadata=[], distances=[])

    result = tool.execute(query="something", lesson_number=3)

    assert "No relevant content found" in result
    assert "lesson 3" in result.lower()


# ── 3. error propagation ──────────────────────────────────────────────────────

def test_execute_returns_error_string_on_search_error(tool, mock_store):
    mock_store.search.return_value = SearchResults.empty("Search error: index too small")

    result = tool.execute(query="any query")

    assert "Search error" in result


def test_execute_returns_no_course_found_on_unresolved_course(tool, mock_store):
    mock_store.search.return_value = SearchResults.empty(
        "No course found matching 'Nonexistent Course'"
    )

    result = tool.execute(query="something", course_name="Nonexistent Course")

    assert "No course found" in result


def test_execute_does_not_raise_when_store_raises(tool, mock_store):
    """VectorStore exceptions must be caught and returned as an error string."""
    mock_store.search.side_effect = Exception("ChromaDB connection failed")

    result = tool.execute(query="what is MCP?")

    assert "Search error" in result
    assert "ChromaDB connection failed" in result


# ── 4. filter forwarding ──────────────────────────────────────────────────────

def test_execute_forwards_query_only(tool, mock_store):
    mock_store.search.return_value = SearchResults(documents=[], metadata=[], distances=[])

    tool.execute(query="neural networks")

    mock_store.search.assert_called_once_with(
        query="neural networks", course_name=None, lesson_number=None
    )


def test_execute_forwards_course_name_filter(tool, mock_store):
    mock_store.search.return_value = SearchResults(documents=[], metadata=[], distances=[])

    tool.execute(query="activation functions", course_name="Deep Learning")

    mock_store.search.assert_called_once_with(
        query="activation functions", course_name="Deep Learning", lesson_number=None
    )


def test_execute_forwards_lesson_number_filter(tool, mock_store):
    mock_store.search.return_value = SearchResults(documents=[], metadata=[], distances=[])

    tool.execute(query="backpropagation", lesson_number=2)

    mock_store.search.assert_called_once_with(
        query="backpropagation", course_name=None, lesson_number=2
    )


# ── 5. source tracking ────────────────────────────────────────────────────────

def test_execute_populates_last_sources(tool, mock_store):
    mock_store.search.return_value = _results(
        docs=["content"],
        metadata=[{"course_title": "MCP Course", "lesson_number": 1}],
    )

    tool.execute(query="test")

    assert len(tool.last_sources) == 1
    assert tool.last_sources[0]["label"] == "MCP Course - Lesson 1"


def test_execute_last_sources_empty_on_no_results(tool, mock_store):
    mock_store.search.return_value = SearchResults(documents=[], metadata=[], distances=[])

    tool.execute(query="nothing")

    assert tool.last_sources == []
