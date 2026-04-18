import sys
import os

# Make backend modules importable from within the tests/ subdirectory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import MagicMock


@pytest.fixture
def mock_rag_system():
    """RAGSystem mock with sensible defaults for API-level tests."""
    rag = MagicMock()
    rag.session_manager.create_session.return_value = "test-session-id"
    rag.query.return_value = ("Test answer", [{"label": "Course A - Lesson 1", "url": "http://example.com"}])
    rag.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Course A", "Course B"],
    }
    return rag
