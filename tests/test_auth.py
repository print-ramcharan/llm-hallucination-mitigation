"""Account token and user-ownership tests."""

from pathlib import Path

from src.auth import AuthStore
from src.ingestion.models import Document, DocumentChunk, DocumentMetadata, DocumentType
from src.ingestion.store import DocumentStore


def test_account_registration_login_and_signed_token(tmp_path: Path) -> None:
    store = AuthStore(tmp_path / "auth.sqlite")
    profile = store.register("alice", "safe-password-123")

    assert store.verify(profile["access_token"]) == profile["user_id"]
    assert store.login("ALICE", "safe-password-123")["user_id"] == profile["user_id"]
    assert store.login("alice", "wrong-password") is None
    assert store.verify(profile["access_token"] + "tampered") is None


def test_duplicate_accounts_and_weak_passwords_are_rejected(tmp_path: Path) -> None:
    store = AuthStore(tmp_path / "auth.sqlite")
    store.register("alice", "safe-password-123")

    try:
        store.register("ALICE", "another-safe-password")
        assert False, "duplicate account should be rejected"
    except ValueError as error:
        assert "already exists" in str(error)
    try:
        store.register("bob", "short")
        assert False, "weak password should be rejected"
    except ValueError as error:
        assert "8 characters" in str(error)


def test_document_storage_is_owner_scoped_and_ids_are_namespaced(
    tmp_path: Path,
) -> None:
    store = DocumentStore(tmp_path / "documents")
    meta = DocumentMetadata(
        document_id="source-id",
        source_name="report.txt",
        file_type=DocumentType.TXT,
        content_hash="same-content-hash",
        chunk_count=1,
    )
    alice_doc = Document(
        id="source-id",
        content="private text",
        metadata=meta,
        chunks=[
            DocumentChunk(
                id="source-chunk", document_id="source-id", content="private text"
            )
        ],
    )
    bob_doc = alice_doc.model_copy(deep=True)
    store.assign_owner(alice_doc, "alice")
    store.assign_owner(bob_doc, "bob")
    store.add(alice_doc, owner_id="alice")
    store.add(bob_doc, owner_id="bob")

    assert alice_doc.id != bob_doc.id
    assert store.get(alice_doc.id, owner_id="alice") is not None
    assert store.get(alice_doc.id, owner_id="bob") is None
    assert len(store.list_all(owner_id="alice")) == 1
    assert len(store.list_all(owner_id="bob")) == 1



