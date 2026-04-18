"""
Tests for FastAPI endpoints (/api/query, /api/courses).

A local test app is built from scratch to mirror app.py's routes without
mounting the ../frontend static directory, which does not exist in the test
environment.
"""
import pytest
from unittest.mock import MagicMock
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel
from typing import List, Optional


# ── test app factory ──────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    session_id: str

class CourseStats(BaseModel):
    total_courses: int
    course_titles: List[str]


def create_test_app(rag) -> FastAPI:
    """Build a FastAPI app with the same routes as app.py, no static mount."""
    test_app = FastAPI()

    @test_app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id or rag.session_manager.create_session()
            answer, sources = rag.query(request.query, session_id)
            return QueryResponse(answer=answer, sources=sources, session_id=session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @test_app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = rag.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return test_app


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def client(mock_rag_system):
    app = create_test_app(mock_rag_system)
    return TestClient(app)


# ── POST /api/query ───────────────────────────────────────────────────────────

def test_query_returns_200(client):
    resp = client.post("/api/query", json={"query": "what is MCP?"})
    assert resp.status_code == 200


def test_query_response_contains_answer(client):
    resp = client.post("/api/query", json={"query": "what is MCP?"})
    assert resp.json()["answer"] == "Test answer"


def test_query_response_contains_sources(client):
    resp = client.post("/api/query", json={"query": "what is MCP?"})
    sources = resp.json()["sources"]
    assert isinstance(sources, list)
    assert sources[0]["label"] == "Course A - Lesson 1"


def test_query_response_contains_session_id(client):
    resp = client.post("/api/query", json={"query": "what is MCP?"})
    assert "session_id" in resp.json()


def test_query_creates_new_session_when_not_provided(client, mock_rag_system):
    client.post("/api/query", json={"query": "hello"})
    mock_rag_system.session_manager.create_session.assert_called_once()


def test_query_reuses_session_when_provided(client, mock_rag_system):
    client.post("/api/query", json={"query": "hello", "session_id": "existing-session"})
    mock_rag_system.session_manager.create_session.assert_not_called()


def test_query_passes_session_id_to_rag(client, mock_rag_system):
    client.post("/api/query", json={"query": "hello", "session_id": "my-session"})
    mock_rag_system.query.assert_called_once_with("hello", "my-session")


def test_query_returns_new_session_id_in_body(client, mock_rag_system):
    mock_rag_system.session_manager.create_session.return_value = "generated-session"
    resp = client.post("/api/query", json={"query": "hello"})
    assert resp.json()["session_id"] == "generated-session"


def test_query_missing_query_field_returns_422(client):
    resp = client.post("/api/query", json={})
    assert resp.status_code == 422


def test_query_returns_500_when_rag_raises(client, mock_rag_system):
    mock_rag_system.query.side_effect = RuntimeError("ChromaDB unavailable")
    resp = client.post("/api/query", json={"query": "anything"})
    assert resp.status_code == 500


def test_query_500_detail_contains_error_message(client, mock_rag_system):
    mock_rag_system.query.side_effect = RuntimeError("ChromaDB unavailable")
    resp = client.post("/api/query", json={"query": "anything"})
    assert "ChromaDB unavailable" in resp.json()["detail"]


# ── GET /api/courses ──────────────────────────────────────────────────────────

def test_courses_returns_200(client):
    resp = client.get("/api/courses")
    assert resp.status_code == 200


def test_courses_returns_total_count(client):
    resp = client.get("/api/courses")
    assert resp.json()["total_courses"] == 2


def test_courses_returns_course_titles(client):
    resp = client.get("/api/courses")
    assert resp.json()["course_titles"] == ["Course A", "Course B"]


def test_courses_returns_500_when_rag_raises(client, mock_rag_system):
    mock_rag_system.get_course_analytics.side_effect = RuntimeError("DB error")
    resp = client.get("/api/courses")
    assert resp.status_code == 500


def test_courses_500_detail_contains_error_message(client, mock_rag_system):
    mock_rag_system.get_course_analytics.side_effect = RuntimeError("DB error")
    resp = client.get("/api/courses")
    assert "DB error" in resp.json()["detail"]


def test_courses_empty_catalog(client, mock_rag_system):
    mock_rag_system.get_course_analytics.return_value = {
        "total_courses": 0,
        "course_titles": [],
    }
    resp = client.get("/api/courses")
    assert resp.status_code == 200
    assert resp.json()["total_courses"] == 0
    assert resp.json()["course_titles"] == []
