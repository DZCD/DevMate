"""Tests for the RAG module."""

import tempfile
from pathlib import Path

from devmate.rag import RAGEngine


def test_rag_engine_initialization() -> None:
    """Test RAG engine can be initialized."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = RAGEngine(persist_directory=tmpdir)
        assert engine.get_doc_count() == 0


def test_rag_engine_ingest_documents() -> None:
    """Test document ingestion."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test document
        docs_dir = Path(tmpdir) / "docs"
        docs_dir.mkdir()
        (docs_dir / "test.md").write_text(
            "# Test Document\n\nThis is a test document.\n\n"
            "## Section 1\n\nSome content here.",
            encoding="utf-8",
        )

        engine = RAGEngine(persist_directory=tmpdir)
        count = engine.ingest_documents(docs_dir)
        assert count > 0
        assert engine.get_doc_count() > 0


def test_rag_engine_search() -> None:
    """Test document search."""
    with tempfile.TemporaryDirectory() as tmpdir:
        docs_dir = Path(tmpdir) / "docs"
        docs_dir.mkdir()
        (docs_dir / "python.md").write_text(
            "# Python Guidelines\n\n"
            "Use type hints for all function signatures.\n\n"
            "## Formatting\n\n"
            "Follow PEP 8 style guide for all Python code.",
            encoding="utf-8",
        )

        engine = RAGEngine(persist_directory=tmpdir)
        engine.ingest_documents(docs_dir)

        results = engine.search("PEP 8 formatting")
        assert len(results) > 0
        assert any("PEP 8" in doc.page_content for doc in results)


def test_rag_engine_empty_directory() -> None:
    """Test that empty directory is handled gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        docs_dir = Path(tmpdir) / "docs"
        docs_dir.mkdir()
        engine = RAGEngine(persist_directory=tmpdir)
        count = engine.ingest_documents(docs_dir)
        assert count == 0


def test_rag_engine_nonexistent_directory() -> None:
    """Test that nonexistent directory returns 0."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = RAGEngine(persist_directory=tmpdir)
        count = engine.ingest_documents("/nonexistent/path")
        assert count == 0
