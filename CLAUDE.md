# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Management

Always use `uv` for all dependency management and execution — never use `pip` or bare `python` directly.

```bash
uv sync              # install/update dependencies
uv add <package>     # add a new dependency
uv run <command>     # run any command in the project environment
```

## Running the Application

```bash
# Start the server (from project root)
./run.sh

# Or manually
cd backend && uv run uvicorn app:app --reload --port 8000
```

Requires a `.env` file in the project root:
```
ANTHROPIC_API_KEY=your-key-here
```

App runs at `http://localhost:8000`. API docs at `http://localhost:8000/docs`.

## Architecture

This is a RAG chatbot with a FastAPI backend and plain HTML/JS/CSS frontend. The frontend is served as static files by FastAPI itself — there is no separate frontend build step.`

### Request Flow

1. Frontend (`frontend/script.js`) POSTs `{ query, session_id }` to `/api/query`
2. `app.py` creates a session if needed and delegates to `RAGSystem.query()`
3. `RAGSystem` fetches conversation history from `SessionManager`, then calls `AIGenerator`
4. `AIGenerator` makes a first Claude API call with the `search_course_content` tool available
5. If Claude decides to search, `ToolManager` dispatches to `CourseSearchTool`, which queries ChromaDB
6. Search results are sent back to Claude in a second API call; the final answer is returned
7. Sources and answer are stored in session history, then returned to the frontend

### Key Design Decisions

- **Tool-based retrieval**: Claude decides when to search — general knowledge questions skip ChromaDB entirely. The system prompt caps Claude at one search per query.
- **Two ChromaDB collections**: `course_catalog` stores course-level metadata for semantic course name resolution; `course_content` stores the actual text chunks for retrieval.
- **Course name resolution**: When a `course_name` filter is passed to `VectorStore.search()`, it first does a semantic lookup in `course_catalog` to resolve fuzzy names to exact titles before filtering `course_content`.
- **Session history as plain text**: `SessionManager` stores history as formatted strings injected into the system prompt, not as structured message arrays.

### Configuration (`backend/config.py`)

All tunable parameters live in the `Config` dataclass:
- `ANTHROPIC_MODEL` — Claude model used for generation
- `EMBEDDING_MODEL` — SentenceTransformer model (`all-MiniLM-L6-v2`)
- `CHUNK_SIZE` / `CHUNK_OVERLAP` — text chunking parameters
- `MAX_RESULTS` — number of ChromaDB results returned per search
- `MAX_HISTORY` — number of conversation exchanges retained per session
- `CHROMA_PATH` — where ChromaDB persists data (`./chroma_db` relative to `backend/`)

### Course Document Format

Files in `docs/` must follow this structure for `DocumentProcessor` to parse them correctly:

```
Course Title: <title>
Course Link: <url>
Course Instructor: <name>

Lesson 0: <lesson title>
Lesson Link: <url>
<lesson content...>

Lesson 1: <lesson title>
...
```

Documents are loaded from `docs/` on startup. Already-indexed courses (matched by title) are skipped.
