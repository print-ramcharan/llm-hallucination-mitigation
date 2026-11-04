"""Unit and integration test suite for Module 7: Cross-Session External Memory & Client Sync."""

import shutil
import tempfile

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.generation.models import (
    ClaimStatus,
    ClaimVerification,
    GroundingReport,
)
from src.memory.models import (
    EpisodicMemoryItem,
    HierarchicalSummary,
    MemoryPermissions,
    Session,
    WebContextCaptureRequest,
)
from src.memory.session_registry import SessionRegistry
from src.memory.store import ExternalMemoryStore
from src.memory.summarizer import HierarchicalSummarizer
from src.memory.synchronizer import CrossSessionMemorySynchronizer

client = TestClient(app)


@pytest.fixture
def temp_memory_store():
    """Create an isolated ExternalMemoryStore backed by a temporary directory."""
    temp_dir = tempfile.mkdtemp()
    store = ExternalMemoryStore(storage_dir=temp_dir)
    yield store
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def memory_components(temp_memory_store):
    """Fixture providing isolated SessionRegistry, Summarizer, and Synchronizer."""
    registry = SessionRegistry(store=temp_memory_store)
    summarizer = HierarchicalSummarizer(turns_per_summary=3)
    synchronizer = CrossSessionMemorySynchronizer(
        store=temp_memory_store,
        registry=registry,
        summarizer=summarizer,
    )
    return {
        "store": temp_memory_store,
        "registry": registry,
        "summarizer": summarizer,
        "synchronizer": synchronizer,
    }


# =============================================================================
# 1. Data Model Tests
# =============================================================================


def test_session_model():
    """Validate Session model initialization, default values, and serialization."""
    session = Session(
        session_id="sess_test_123",
        user_id="user_alice",
        title="Project Research",
        active_document_ids=["doc_1", "doc_2"],
    )
    assert session.session_id == "sess_test_123"
    assert session.user_id == "user_alice"
    assert session.turn_count == 0
    assert session.is_active is True
    assert len(session.active_document_ids) == 2
    data = session.model_dump()
    assert data["title"] == "Project Research"


def test_memory_permissions_model():
    """Validate MemoryPermissions controls and default limits."""
    perms = MemoryPermissions()
    assert perms.read_enabled is True
    assert perms.write_enabled is True
    assert perms.auto_summarize is True
    assert perms.max_injected_memories == 3
    assert perms.max_injected_tokens == 250


def test_episodic_memory_item_model():
    """Validate EpisodicMemoryItem construction and verified facts tracking."""
    item = EpisodicMemoryItem(
        memory_id="mem_01",
        session_id="sess_01",
        query="What is the vacation policy?",
        answer="Employees receive 20 days paid leave [Doc 1, Chunk 0].",
        verified_facts=["Employees receive 20 days paid leave."],
        citations=["[Doc 1, Chunk 0]"],
    )
    assert item.memory_id == "mem_01"
    assert len(item.verified_facts) == 1
    assert item.importance_score == 1.0


# =============================================================================
# 2. Session Registry Tests
# =============================================================================


def test_session_lifecycle(memory_components):
    """Test full session lifecycle: creation, retrieval, listing, updating, deletion."""
    registry = memory_components["registry"]

    session = registry.create_session(
        user_id="user_bob",
        title="HR Policy Analysis",
        active_document_ids=["doc_hr_01"],
    )
    assert session.session_id.startswith("sess_")
    assert session.title == "HR Policy Analysis"

    # Get session
    retrieved = registry.get_session(session.session_id)
    assert retrieved is not None
    assert retrieved.title == "HR Policy Analysis"

    # List sessions
    sessions = registry.list_sessions(user_id="user_bob")
    assert len(sessions) >= 1
    assert sessions[0].session_id == session.session_id

    # Update session
    updated = registry.update_session(
        session_id=session.session_id,
        title="Updated HR Policy Analysis",
        active_document_ids=["doc_hr_01", "doc_hr_02"],
    )
    assert updated.title == "Updated HR Policy Analysis"
    assert len(updated.active_document_ids) == 2

    # Delete session
    deleted = registry.delete_session(session.session_id)
    assert deleted is True
    assert registry.get_session(session.session_id) is None


def test_session_permissions_governance(memory_components):
    """Test updating and querying read/write permissions per session."""
    registry = memory_components["registry"]
    session = registry.create_session(title="Governance Session")

    # Initial permissions are default
    perms = registry.get_permissions(session.session_id)
    assert perms.read_enabled is True
    assert perms.write_enabled is True

    # Disable write permission
    new_perms = MemoryPermissions(
        read_enabled=True, write_enabled=False, auto_summarize=False
    )
    registry.set_permissions(session.session_id, new_perms)

    saved_perms = registry.get_permissions(session.session_id)
    assert saved_perms.write_enabled is False
    assert saved_perms.read_enabled is True
    assert saved_perms.auto_summarize is False


# =============================================================================
# 3. Store & Semantic Search Tests
# =============================================================================


def test_store_memory_item_and_search(memory_components):
    """Test adding episodic memory turns and semantic vector search."""
    store = memory_components["store"]
    registry = memory_components["registry"]
    session = registry.create_session(title="Search Session")

    item1 = EpisodicMemoryItem(
        memory_id="mem_vacation",
        session_id=session.session_id,
        turn_index=0,
        query="How many annual vacation days are allowed?",
        answer="Full-time staff get 20 days vacation per calendar year.",
        verified_facts=["Full-time staff get 20 days vacation per calendar year."],
        referenced_doc_ids=["doc_vacation"],
    )
    store.save_memory_item(item1)

    item2 = EpisodicMemoryItem(
        memory_id="mem_health",
        session_id=session.session_id,
        turn_index=1,
        query="What health insurance plans exist?",
        answer="The company offers Comprehensive Health Plus with dental.",
        verified_facts=["Company offers Comprehensive Health Plus with dental."],
        referenced_doc_ids=["doc_health"],
    )
    store.save_memory_item(item2)

    # Search for vacation
    matches, scores = store.search_memories(
        query="vacation leave allowance",
        session_id=session.session_id,
        top_k=2,
    )
    assert len(matches) > 0
    assert matches[0].memory_id == "mem_vacation"

    # Search for health insurance
    matches_health, _ = store.search_memories(
        query="health dental insurance coverage",
        session_id=session.session_id,
        top_k=2,
    )
    assert len(matches_health) > 0
    assert matches_health[0].memory_id == "mem_health"


def test_clear_session_memory(memory_components):
    """Test clearing memories wipes turns but preserves session container."""
    store = memory_components["store"]
    registry = memory_components["registry"]
    session = registry.create_session(title="Clearable Session")

    item = EpisodicMemoryItem(
        memory_id="mem_temp",
        session_id=session.session_id,
        turn_index=0,
        query="Temporary query",
        answer="Temporary answer",
    )
    store.save_memory_item(item)
    assert len(store.list_session_memories(session.session_id)) == 1

    registry.clear_session_memory(session.session_id)
    assert len(store.list_session_memories(session.session_id)) == 0
    # Session itself still exists
    assert registry.get_session(session.session_id) is not None


# =============================================================================
# 4. Hierarchical Summarizer Tests
# =============================================================================


def test_hierarchical_summarizer(memory_components):
    """Test summarizer condensing multiple episodic turns into persistent summary."""
    summarizer = memory_components["summarizer"]
    session = Session(session_id="sess_sum", title="Cloud Migration")

    turns = [
        EpisodicMemoryItem(
            memory_id="m1",
            session_id=session.session_id,
            turn_index=0,
            query="Which cloud provider was selected?",
            answer="Google Cloud Platform was chosen for BigQuery integration.",
            verified_facts=["Google Cloud Platform was chosen."],
            referenced_doc_ids=["doc_cloud_eval"],
            tags=["cloud", "gcp"],
        ),
        EpisodicMemoryItem(
            memory_id="m2",
            session_id=session.session_id,
            turn_index=1,
            query="What is the migration timeline?",
            answer="Phase 1 will complete by Q4 2026.",
            verified_facts=["Phase 1 will complete by Q4 2026."],
            tags=["timeline"],
        ),
    ]

    summary: HierarchicalSummary = summarizer.summarize_session(session, turns)
    assert summary.session_id == session.session_id
    assert summary.turn_count == 2
    assert "Cloud Migration" in summary.title
    assert "Google Cloud Platform" in summary.summary_text
    assert len(summary.key_entities) > 0


# =============================================================================
# 5. Synchronizer (Pre & Post Sync) Tests
# =============================================================================


def test_pre_inference_sync_budgeting(memory_components):
    """Test pre-inference memory injection respects max token budget."""
    sync = memory_components["synchronizer"]
    registry = memory_components["registry"]
    store = memory_components["store"]

    session = registry.create_session(title="Budgeting Session")

    # Add past turn
    item = EpisodicMemoryItem(
        memory_id="mem_budget_test",
        session_id=session.session_id,
        turn_index=0,
        query="What is the remote work policy?",
        answer="Employees may work remotely up to 3 days per week with manager approval.",
        verified_facts=["Remote work allowed up to 3 days per week."],
    )
    store.save_memory_item(item)

    # Perform pre-inference sync
    context = sync.pre_inference_sync(
        query="remote work days from home",
        session_id=session.session_id,
        max_tokens=150,
    )
    assert context.read_enabled is True
    assert len(context.injected_memories) > 0
    assert "Remote work allowed" in context.formatted_memory_context
    assert context.estimated_tokens <= 150


def test_pre_inference_sync_disabled(memory_components):
    """Test pre-inference sync returns empty context when read permission is disabled."""
    sync = memory_components["synchronizer"]
    registry = memory_components["registry"]

    session = registry.create_session(title="Private Session")
    registry.set_permissions(
        session.session_id,
        MemoryPermissions(read_enabled=False),
    )

    context = sync.pre_inference_sync(
        query="secret query",
        session_id=session.session_id,
    )
    assert context.read_enabled is False
    assert context.formatted_memory_context == ""
    assert len(context.injected_memories) == 0


def test_post_inference_sync_verified_facts(memory_components):
    """Test post-inference memory sync only stores entailed facts from grounding report."""
    sync = memory_components["synchronizer"]
    registry = memory_components["registry"]
    store = memory_components["store"]

    session = registry.create_session(title="Verification Session")

    # Grounding report with entailed and neutral claims
    grounding_report = GroundingReport(
        faithfulness_score=0.5,
        hallucination_detected=False,
        total_claims=2,
        entailed_claims_count=1,
        neutral_claims_count=1,
        contradicted_claims_count=0,
        claims=[
            ClaimVerification(
                claim_text="Staff receive 20 days vacation.",
                status=ClaimStatus.ENTAILED,
                confidence=0.95,
                cited_sources=["[Doc 1, Chunk 0]"],
            ),
            ClaimVerification(
                claim_text="Staff get free lunch on Fridays.",
                status=ClaimStatus.NEUTRAL,
                confidence=0.40,
            ),
        ],
        verified_citations=["[Doc 1, Chunk 0]"],
        unverified_citations=[],
    )

    saved_item = sync.post_inference_sync(
        session_id=session.session_id,
        query="Vacation and perks",
        answer="Staff receive 20 days vacation [Doc 1, Chunk 0]. Staff get free lunch on Fridays.",
        grounding_report=grounding_report,
    )

    assert saved_item is not None
    assert len(saved_item.verified_facts) == 1
    assert "Staff receive 20 days vacation." in saved_item.verified_facts
    # Neutral claim should not be saved as a verified fact
    assert "Staff get free lunch on Fridays." not in saved_item.verified_facts

    # Verify session turns count updated
    mems = store.list_session_memories(session.session_id)
    assert len(mems) == 1


def test_web_context_capture(memory_components):
    """Test browser extension web context capture."""
    sync = memory_components["synchronizer"]
    registry = memory_components["registry"]
    session = registry.create_session(title="Web Session")

    req = WebContextCaptureRequest(
        url="https://docs.python.org/3/library/sqlite3.html",
        title="sqlite3 — DB-API 2.0 interface for SQLite databases",
        selected_text="SQLite is a C library that provides a lightweight disk-based database.",
        session_id=session.session_id,
        tags=["python", "sqlite"],
    )

    captured = sync.capture_web_context(req)
    assert captured.intent == "WEB_CONTEXT"
    assert "SQLite is a C library" in captured.answer
    assert "URL: https://docs.python.org/3/library/sqlite3.html" in captured.citations
    assert "python" in captured.tags


# =============================================================================
# 6. REST API Endpoint Tests
# =============================================================================


def test_api_session_crud():
    """Test /api/memory/sessions endpoints."""
    # Create session
    create_resp = client.post(
        "/api/memory/sessions",
        json={"title": "API Test Session", "user_id": "test_user"},
    )
    assert create_resp.status_code == 201
    created_data = create_resp.json()
    session_id = created_data["session_id"]
    assert created_data["title"] == "API Test Session"

    # List sessions
    list_resp = client.get("/api/memory/sessions?user_id=test_user")
    assert list_resp.status_code == 200
    assert any(s["session_id"] == session_id for s in list_resp.json())

    # Get details
    get_resp = client.get(f"/api/memory/sessions/{session_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["session"]["title"] == "API Test Session"

    # Update session
    patch_resp = client.patch(
        f"/api/memory/sessions/{session_id}",
        json={"title": "Updated API Test Session"},
    )
    assert patch_resp.status_code == 200
    assert patch_resp.json()["title"] == "Updated API Test Session"

    # Delete session
    del_resp = client.delete(f"/api/memory/sessions/{session_id}")
    assert del_resp.status_code == 200


def test_api_memory_permissions():
    """Test GET and PUT /api/memory/sessions/{session_id}/permissions."""
    create_resp = client.post("/api/memory/sessions", json={"title": "Perms Test"})
    session_id = create_resp.json()["session_id"]

    # Get permissions
    get_perms = client.get(f"/api/memory/sessions/{session_id}/permissions")
    assert get_perms.status_code == 200
    assert get_perms.json()["read_enabled"] is True

    # Update permissions
    put_perms = client.put(
        f"/api/memory/sessions/{session_id}/permissions",
        json={
            "read_enabled": False,
            "write_enabled": True,
            "auto_summarize": False,
            "max_injected_memories": 2,
            "max_injected_tokens": 200,
        },
    )
    assert put_perms.status_code == 200
    data = put_perms.json()
    assert data["read_enabled"] is False
    assert data["auto_summarize"] is False
    assert data["max_injected_memories"] == 2


def test_api_capture_web_and_search():
    """Test /api/memory/capture-web and /api/memory/search."""
    # Capture web text
    capture_resp = client.post(
        "/api/memory/capture-web",
        json={
            "url": "https://fastapi.tiangolo.com",
            "title": "FastAPI Web Framework",
            "selected_text": "FastAPI is a modern, fast web framework for building APIs with Python.",
            "tags": ["web", "fastapi"],
        },
    )
    assert capture_resp.status_code == 201
    captured = capture_resp.json()
    assert "FastAPI is a modern" in captured["answer"]

    # Semantic search
    search_resp = client.post(
        "/api/memory/search",
        json={"query": "fast web framework python"},
    )
    assert search_resp.status_code == 200
    search_data = search_resp.json()
    assert search_data["total_found"] > 0
    assert any("FastAPI" in m["answer"] for m in search_data["memories"])


def test_api_export_markdown_and_clear():
    """Test /api/memory/sessions/{session_id}/export and /clear."""
    create_resp = client.post("/api/memory/sessions", json={"title": "Export Session"})
    session_id = create_resp.json()["session_id"]

    # Add a post-inference interaction
    client.post(
        "/api/memory/sync/post",
        json={
            "session_id": session_id,
            "query": "What is Python?",
            "answer": "Python is a high-level general-purpose programming language.",
            "verified_facts": ["Python is a high-level programming language."],
        },
    )

    # Export markdown
    export_resp = client.get(
        f"/api/memory/sessions/{session_id}/export?format=markdown"
    )
    assert export_resp.status_code == 200
    assert "# Session Memory Export" in export_resp.text
    assert "What is Python?" in export_resp.text

    # Clear memory
    clear_resp = client.post(f"/api/memory/sessions/{session_id}/clear")
    assert clear_resp.status_code == 200

    # Verify turns are cleared
    get_resp = client.get(f"/api/memory/sessions/{session_id}")
    assert len(get_resp.json()["turns"]) == 0
