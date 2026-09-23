"""Tests for the FastAPI Ingestion Endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.ingestion.store import default_store


@pytest.fixture(autouse=True)
def clean_store():
    default_store.clear()
    yield
    default_store.clear()


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert ".pdf" in data["supported_extensions"]
    assert ".docx" in data["supported_extensions"]


def test_upload_txt_file():
    content = b"Line 1 text.\n\nLine 2 text."
    files = {"file": ("test_doc.txt", content, "text/plain")}
    response = client.post("/api/ingest/upload", files=files)

    assert response.status_code == 201
    res_data = response.json()
    assert res_data["success"] is True
    doc = res_data["document"]
    assert doc["metadata"]["source_name"] == "test_doc.txt"
    assert doc["metadata"]["file_type"] == "txt"
    assert len(doc["elements"]) == 2


def test_upload_unsupported_file_returns_415():
    files = {"file": ("binary_blob.bin", b"\x00\x01\x02\x03", "application/octet-stream")}
    response = client.post("/api/ingest/upload", files=files)
    assert response.status_code == 415


def test_ingest_raw_text():
    payload = {
        "title": "notes.md",
        "content": "# Research Notes\nExploring context dilution.",
        "format": "md",
    }
    response = client.post("/api/ingest/text", json=payload)
    assert response.status_code == 201
    res_data = response.json()
    assert res_data["document"]["metadata"]["file_type"] == "md"
    assert "Exploring context dilution" in res_data["document"]["content"]


def test_list_and_get_and_delete_documents():
    # Ingest two items
    client.post("/api/ingest/text", json={"title": "d1.txt", "content": "Document 1", "format": "txt"})
    client.post("/api/ingest/text", json={"title": "d2.txt", "content": "Document 2", "format": "txt"})

    # List
    list_res = client.get("/api/ingest/documents")
    assert list_res.status_code == 200
    docs = list_res.json()
    assert len(docs) == 2

    target_id = docs[0]["document_id"]

    # Get single
    get_res = client.get(f"/api/ingest/documents/{target_id}")
    assert get_res.status_code == 200
    assert get_res.json()["id"] == target_id

    # Delete
    del_res = client.delete(f"/api/ingest/documents/{target_id}")
    assert del_res.status_code == 200

    # Verify deleted
    get_res2 = client.get(f"/api/ingest/documents/{target_id}")
    assert get_res2.status_code == 404


def test_get_document_chunks_and_metadata():
    payload = {
        "title": "employee_handbook.txt",
        "content": "All employees receive 20 days off.\n\nSick leave is 10 days.",
        "format": "txt",
        "document_type": "HR_POLICY",
    }
    create_res = client.post("/api/ingest/text", json=payload)
    assert create_res.status_code == 201
    doc_id = create_res.json()["document"]["id"]

    # Retrieve chunks endpoint
    chunks_res = client.get(f"/api/ingest/documents/{doc_id}/chunks")
    assert chunks_res.status_code == 200
    chunks = chunks_res.json()
    assert len(chunks) >= 1
    c0 = chunks[0]
    assert c0["metadata"]["document_type"] == "HR_POLICY"
    assert c0["metadata"]["source"] == "employee_handbook.txt"
    assert "chunk_id" in c0["metadata"]
    assert "token_count" in c0["metadata"]
