"""RAG (Retrieval-Augmented Generation) module for DevMate.

Handles document ingestion, embedding, and retrieval using ChromaDB.
"""

import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

logger = logging.getLogger(__name__)


class RAGEngine:
    """RAG engine for document retrieval."""

    def __init__(
        self,
        persist_directory: str = ".chroma_db",
        collection_name: str = "devmate_docs",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model_name: str = "text-embedding-3-small",
        openai_api_key: str | None = None,
        openai_api_base: str | None = None,
    ) -> None:
        """Initialize the RAG engine.

        Args:
            persist_directory: Directory for ChromaDB persistence.
            collection_name: Name of the ChromaDB collection.
            chunk_size: Size of document chunks.
            chunk_overlap: Overlap between document chunks.
            embedding_model_name: Name of the OpenAI embedding model.
            openai_api_key: API key for the OpenAI-compatible embedding service.
            openai_api_base: Base URL for the OpenAI-compatible embedding service.
        """
        self._persist_directory = persist_directory
        self._collection_name = collection_name
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._embedding_model_name = embedding_model_name

        self._client = chromadb.PersistentClient(path=persist_directory)

        # Set up embedding function for ChromaDB when API credentials are provided
        if openai_api_key:
            embedding_kwargs: dict[str, Any] = {
                "model_name": embedding_model_name,
                "api_key": openai_api_key,
            }
            if openai_api_base:
                embedding_kwargs["api_base"] = openai_api_base
            embedding_function = OpenAIEmbeddingFunction(**embedding_kwargs)
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                embedding_function=embedding_function,
            )
        else:
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
            )
        logger.info(
            "RAG engine initialized (collection=%s, persist_dir=%s)",
            collection_name,
            persist_directory,
        )

    def ingest_documents(self, docs_directory: str | Path) -> int:
        """Ingest all markdown documents from a directory.

        Args:
            docs_directory: Path to the directory containing markdown files.

        Returns:
            Number of documents ingested.
        """
        docs_path = Path(docs_directory)
        if not docs_path.exists():
            logger.warning("Docs directory does not exist: %s", docs_path)
            return 0

        md_files = list(docs_path.glob("**/*.md"))
        logger.info("Found %d markdown files in %s", len(md_files), docs_path)

        total_chunks = 0
        for md_file in md_files:
            file_mtime = str(md_file.stat().st_mtime)

            # Check if file has already been ingested with the same mtime
            existing = self._collection.get(
                where={"source": str(md_file)},
                include=["metadatas"],
            )
            if existing and existing["metadatas"] and existing["metadatas"][0]:
                existing_mtime = existing["metadatas"][0].get("file_mtime")
                if existing_mtime == file_mtime:
                    logger.info("Skipping unchanged file: %s", md_file)
                    continue

            chunks = self._ingest_file(md_file, file_mtime=file_mtime)
            total_chunks += chunks

        logger.info("Ingested %d total chunks", total_chunks)
        return total_chunks

    def _ingest_file(self, file_path: Path, file_mtime: str = "") -> int:
        """Ingest a single markdown file.

        Args:
            file_path: Path to the markdown file.
            file_mtime: File modification timestamp string.

        Returns:
            Number of chunks ingested.
        """
        logger.info("Ingesting file: %s", file_path)
        content = file_path.read_text(encoding="utf-8")

        if not content.strip():
            logger.warning("Empty file: %s", file_path)
            return 0

        # Split by markdown headers first
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on
        )
        md_docs = md_splitter.split_text(content)

        # Further split with recursive character splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )
        chunks = text_splitter.split_documents(md_docs)

        if not chunks:
            return 0

        # Add to ChromaDB
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, str]] = []

        for i, chunk in enumerate(chunks):
            doc_id = f"{file_path.stem}::{i}"
            ids.append(doc_id)
            documents.append(chunk.page_content)
            metadata: dict[str, str] = {
                "source": str(file_path),
                "chunk_index": str(i),
            }
            if file_mtime:
                metadata["file_mtime"] = file_mtime
            metadata.update(chunk.metadata)
            metadatas.append(metadata)

        self._collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

        logger.info("Ingested %d chunks from %s", len(chunks), file_path.name)
        return len(chunks)

    def search(self, query: str, n_results: int = 5) -> list[Document]:
        """Search the knowledge base.

        Args:
            query: The search query.
            n_results: Maximum number of results to return.

        Returns:
            A list of relevant documents.
        """
        if self._collection.count() == 0:
            logger.warning("Knowledge base is empty")
            return []

        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=min(n_results, self._collection.count()),
            )

            documents: list[Document] = []
            if results and results.get("documents"):
                for i, doc_text in enumerate(results["documents"][0]):
                    metadata = {}
                    if results.get("metadatas"):
                        metadata = results["metadatas"][0][i] or {}
                    documents.append(
                        Document(
                            page_content=doc_text,
                            metadata=metadata,
                        )
                    )

            logger.info(
                "Search returned %d results for query: %s",
                len(documents),
                query,
            )
            return documents

        except Exception as exc:
            logger.error("Search failed: %s", exc, exc_info=True)
            return []

    def get_doc_count(self) -> int:
        """Return the number of documents in the collection."""
        return self._collection.count()


def create_search_tool(rag_engine: RAGEngine) -> Any:
    """Create a LangChain tool for searching the knowledge base.

    Args:
        rag_engine: The RAG engine instance.

    Returns:
        A LangChain tool function.
    """

    @tool
    def search_knowledge_base(query: str) -> str:
        """Search the local knowledge base for relevant documentation and guidelines.

        Use this tool when you need to find information about internal coding
        standards, project guidelines, architecture decisions, or any documentation
        that has been indexed in the local knowledge base.

        Args:
            query: The search query describing what information you need.
        """
        docs = rag_engine.search(query, n_results=5)
        if not docs:
            return "No relevant documents found in the knowledge base."

        results: list[str] = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            header = doc.metadata.get("Header 1", "")
            header_2 = doc.metadata.get("Header 2", "")
            section = ""
            if header:
                section = f"[{header}"
                if header_2:
                    section += f" > {header_2}"
                section += "] "
            results.append(
                f"--- Document {i} ---\n"
                f"Source: {source}\n"
                f"Section: {section}\n"
                f"Content:\n{doc.page_content}"
            )

        return "\n\n".join(results)

    return search_knowledge_base
