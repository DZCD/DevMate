"""Integration tests for DevMate.

Tests cross-module interactions:
- MCP Server HTTP endpoints (Starlette ASGI test client)
- RAG document ingestion -> retrieval pipeline
- Agent + MCP + RAG end-to-end flow (mocked LLM)
- File tools + Skills cross-module workflow
- Config -> module initialization flow
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from tests.helpers import write_minimal_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. MCP Server integration tests
# ---------------------------------------------------------------------------


class TestMCPServerIntegration:
    """Test MCP Server HTTP endpoints via httpx.AsyncClient."""

    @pytest.fixture
    def mcp_app(self):
        """Create a test MCP app with a fake Tavily key."""
        from mcp_server.server import create_mcp_app

        return create_mcp_app(
            tavily_api_key="tvly-fake-test-key",
            max_results=3,
            route="/mcp",
        )

    async def test_health_endpoint(self, mcp_app) -> None:
        """Test the /health endpoint returns 200."""
        transport = ASGITransport(app=mcp_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "devmate-mcp-server"


# ---------------------------------------------------------------------------
# 2. RAG pipeline integration tests
# ---------------------------------------------------------------------------


class TestRAGPipeline:
    """Test the full RAG document ingestion -> retrieval pipeline."""

    def test_ingest_and_search_pipeline(self, tmp_path) -> None:
        """Test documents can be ingested and then retrieved via search."""
        from devmate.rag import RAGEngine, create_search_tool

        # Create test documents
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()

        (docs_dir / "coding_standards.md").write_text(
            "# Coding Standards\n\n"
            "All Python code must follow PEP 8.\n\n"
            "## Type Hints\n\n"
            "Use type hints for all function parameters and return values.\n\n"
            "## Error Handling\n\n"
            "Use custom exception classes for domain errors.",
            encoding="utf-8",
        )

        (docs_dir / "architecture.md").write_text(
            "# Architecture Overview\n\n"
            "The system follows a microservices architecture.\n\n"
            "## Services\n\n"
            "- API Gateway\n- User Service\n- Auth Service\n\n"
            "## Database\n\n"
            "PostgreSQL for relational data, Redis for caching.",
            encoding="utf-8",
        )

        # Ingest
        persist_dir = tmp_path / ".chroma_db"
        engine = RAGEngine(persist_directory=str(persist_dir))
        chunk_count = engine.ingest_documents(docs_dir)

        assert chunk_count > 0
        assert engine.get_doc_count() > 0

        # Search for coding standards
        results = engine.search("PEP 8 type hints")
        assert len(results) > 0
        assert any("PEP 8" in doc.page_content for doc in results)

        # Search for architecture
        results = engine.search("microservices database")
        assert len(results) > 0

        # Search tool integration
        search_tool = create_search_tool(engine)
        tool_output = search_tool.invoke({"query": "PEP 8"})
        assert "PEP 8" in tool_output

    def test_ingest_empty_file_skipped(self, tmp_path) -> None:
        """Test that empty markdown files are gracefully skipped."""
        from devmate.rag import RAGEngine

        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "empty.md").write_text("", encoding="utf-8")
        (docs_dir / "whitespace.md").write_text("   \n\n  \n", encoding="utf-8")

        engine = RAGEngine(persist_directory=str(tmp_path / ".chroma_db"))
        count = engine.ingest_documents(docs_dir)
        assert count == 0

    def test_rag_persistence_across_instances(self, tmp_path) -> None:
        """Test that data persists across RAGEngine instances."""
        from devmate.rag import RAGEngine

        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "persistent.md").write_text(
            "# Persistent Data\n\nThis data should survive across instances.",
            encoding="utf-8",
        )

        persist_dir = tmp_path / ".chroma_db"

        # First instance: ingest
        engine1 = RAGEngine(persist_directory=str(persist_dir))
        count1 = engine1.ingest_documents(docs_dir)
        assert count1 > 0

        # Second instance: should still have data
        engine2 = RAGEngine(persist_directory=str(persist_dir))
        assert engine2.get_doc_count() > 0

        results = engine2.search("persistent data")
        assert len(results) > 0


# ---------------------------------------------------------------------------
# 3. Agent integration tests (with mocked LLM and MCP)
# ---------------------------------------------------------------------------


class TestAgentIntegration:
    """Test Agent with mocked external dependencies."""

    def test_agent_initialization(self, tmp_path) -> None:
        """Test DevMateAgent can be created with a config file."""
        from devmate.agent import DevMateAgent

        config_file = write_minimal_config(tmp_path)
        agent = DevMateAgent(config_path=str(config_file), workspace=str(tmp_path))

        assert agent._config is not None
        assert agent._llm is None

    @patch("devmate.agent.RAGEngine")
    @patch("devmate.agent.SkillsManager")
    @patch(
        "langchain_mcp_adapters.client.MultiServerMCPClient",
        new_callable=MagicMock,
    )
    @patch("devmate.agent.OpenAICompatibleAdapter")
    async def test_agent_initialize_with_mocks(
        self,
        mock_llm_cls,
        mock_mcp_client_cls,
        mock_skills_cls,
        mock_rag_cls,
        tmp_path,
    ) -> None:
        """Test agent initialization with all external deps mocked."""
        from devmate.agent import DevMateAgent

        config_file = write_minimal_config(tmp_path)

        # Setup mocks
        mock_llm_instance = MagicMock()
        mock_llm_cls.return_value = mock_llm_instance

        mock_rag_instance = MagicMock()
        mock_rag_instance.ingest_documents.return_value = 0
        mock_rag_instance.get_doc_count.return_value = 0
        mock_rag_cls.return_value = mock_rag_instance

        mock_skills_instance = MagicMock()
        mock_skills_instance.load_skills.return_value = 0
        mock_skills_instance.get_skill_meta.return_value = ""
        mock_skills_instance.create_tools.return_value = []
        mock_skills_cls.return_value = mock_skills_instance

        # MCP client mock
        mock_mcp_instance = MagicMock()
        mock_mcp_instance.get_tools = AsyncMock(return_value=[])
        mock_mcp_client_cls.return_value = mock_mcp_instance

        agent = DevMateAgent(config_path=str(config_file), workspace=str(tmp_path))
        await agent.initialize()

        # Verify components were created
        mock_llm_cls.assert_called_once()
        mock_rag_cls.assert_called_once()
        mock_skills_cls.assert_called_once()

        # Verify tools include file tools
        tool_names = [t.name for t in agent._tools]
        assert "read" in tool_names
        assert "write" in tool_names
        assert "edit" in tool_names
        assert "glob" in tool_names
        assert "grep" in tool_names
        assert "bash" in tool_names
        assert "create_file" in tool_names
        assert "list_directory" in tool_names
        assert "search_knowledge_base" in tool_names

    @patch("devmate.agent.RAGEngine")
    @patch("devmate.agent.SkillsManager")
    @patch(
        "langchain_mcp_adapters.client.MultiServerMCPClient",
        new_callable=MagicMock,
    )
    @patch("devmate.agent.OpenAICompatibleAdapter")
    async def test_agent_mcp_failure_continues(
        self,
        mock_llm_cls,
        mock_mcp_client_cls,
        mock_skills_cls,
        mock_rag_cls,
        tmp_path,
    ) -> None:
        """Test agent continues when MCP connection fails."""
        from devmate.agent import DevMateAgent

        config_file = write_minimal_config(tmp_path)

        mock_llm_cls.return_value = MagicMock()
        mock_rag_instance = MagicMock()
        mock_rag_instance.ingest_documents.return_value = 0
        mock_rag_cls.return_value = mock_rag_instance
        mock_skills_instance = MagicMock()
        mock_skills_instance.load_skills.return_value = 0
        mock_skills_instance.get_skill_meta.return_value = ""
        mock_skills_instance.create_tools.return_value = []
        mock_skills_cls.return_value = mock_skills_instance

        # MCP raises exception
        mock_mcp_client_cls.side_effect = ConnectionRefusedError("Connection refused")

        agent = DevMateAgent(config_path=str(config_file), workspace=str(tmp_path))
        # Should not raise even though MCP fails
        await agent.initialize()

        # Agent should still be functional
        assert agent._tool_registry is not None

    @patch("devmate.agent.RAGEngine")
    @patch("devmate.agent.SkillsManager")
    @patch(
        "langchain_mcp_adapters.client.MultiServerMCPClient",
        new_callable=MagicMock,
    )
    @patch("devmate.agent.OpenAICompatibleAdapter")
    async def test_agent_run_with_mock(
        self,
        mock_llm_cls,
        mock_mcp_client_cls,
        mock_skills_cls,
        mock_rag_cls,
        tmp_path,
    ) -> None:
        """Test agent run method with mocked LLM."""
        from devmate.agent import DevMateAgent
        from devmate.llm import LLMResponse, TextBlock

        config_file = write_minimal_config(tmp_path)

        # Mock LLM
        mock_llm_cls.return_value = MagicMock()

        # Mock RAG
        mock_rag_instance = MagicMock()
        mock_rag_instance.ingest_documents.return_value = 0
        mock_rag_cls.return_value = mock_rag_instance

        # Mock Skills
        mock_skills_instance = MagicMock()
        mock_skills_instance.load_skills.return_value = 0
        mock_skills_instance.get_skill_meta.return_value = ""
        mock_skills_instance.create_tools.return_value = []
        mock_skills_cls.return_value = mock_skills_instance

        # Mock MCP
        mock_mcp_instance = MagicMock()
        mock_mcp_instance.get_tools = AsyncMock(return_value=[])
        mock_mcp_client_cls.return_value = mock_mcp_instance

        agent = DevMateAgent(config_path=str(config_file), workspace=str(tmp_path))
        await agent.initialize()

        # Mock the LLM's chat method
        mock_llm_instance = mock_llm_cls.return_value
        mock_llm_instance.chat = AsyncMock(
            return_value=LLMResponse(
                content=[TextBlock(text="Mocked agent response.")],
                finish_reason="stop",
            )
        )

        result = await agent.run("Hello, DevMate!")
        assert result == "Mocked agent response."

    @patch("devmate.agent.RAGEngine")
    @patch(
        "langchain_mcp_adapters.client.MultiServerMCPClient",
        new_callable=MagicMock,
    )
    @patch("devmate.agent.OpenAICompatibleAdapter")
    async def test_agent_skills_injected_in_prompt(
        self,
        mock_llm_cls,
        mock_mcp_client_cls,
        mock_rag_cls,
        tmp_path,
    ) -> None:
        """Test that available skills XML is injected into the system prompt."""
        from devmate.agent import DevMateAgent

        # Create skills directory in tmp_path and reference it in config
        skills_dir = tmp_path / ".skills"
        skills_dir.mkdir()
        skill_dir = skills_dir / "test_skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: test_skill\ndescription: Test\n---\n# Test\n",
            encoding="utf-8",
        )

        # Use absolute path for skills directory in config
        config_content = f"""
[model]
base_url = "https://example.com/api"
api_key = "test-key"
model_name = "test-model"

[search]
tavily_api_key = "tvly-test"
max_results = 3

[langsmith]
enabled = false

[skills]
directory = "{skills_dir}"

[rag]
docs_directory = "{tmp_path / "docs"}"
chroma_persist_directory = "{tmp_path / ".chroma_db"}"

[mcp_server]
host = "localhost"
port = 18001
route = "/mcp"
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(config_content, encoding="utf-8")

        mock_llm_cls.return_value = MagicMock()
        mock_rag_instance = MagicMock()
        mock_rag_instance.ingest_documents.return_value = 0
        mock_rag_cls.return_value = mock_rag_instance

        mock_mcp_instance = MagicMock()
        mock_mcp_instance.get_tools = AsyncMock(return_value=[])
        mock_mcp_client_cls.return_value = mock_mcp_instance

        # Don't mock SkillsManager — use real one with our test skills
        agent = DevMateAgent(config_path=str(config_file), workspace=str(tmp_path))
        await agent.initialize()

        # Verify skills were loaded and meta XML contains our skill
        assert agent._skills_manager is not None
        meta_xml = agent._skills_manager.get_skill_meta()
        assert "<available_skills>" in meta_xml
        assert "test_skill" in meta_xml

        # Verify the system prompt contains skills
        assert "test_skill" in agent._system_prompt


# ---------------------------------------------------------------------------
# 4. File tools + Skills cross-module integration
# ---------------------------------------------------------------------------


class TestFileToolsSkillsIntegration:
    """Test file tools and skills working together."""

    def test_create_skill_folder_and_load(self, tmp_path) -> None:
        """Test creating a skill folder via file tools and loading it."""
        from devmate.file_tools import create_file_tools
        from devmate.skills import SkillsManager

        workspace = tmp_path / "workspace"
        workspace.mkdir()

        # Create skill folder using file tools
        tools = create_file_tools(workspace=workspace)
        create_file_tool = next(t for t in tools if t.name == "create_file")

        skill_content = """---
name: "react_component"
description: "How to create React components"
trigger_keywords: ["react", "component", "frontend"]
---

# React Component Pattern

Use functional components with hooks.
"""
        skills_dir = workspace / ".skills" / "react_component"
        create_file_tool.invoke(
            {
                "file_path": str(skills_dir / "SKILL.md"),
                "content": skill_content,
            }
        )

        # Load the skill
        manager = SkillsManager(skills_dir=workspace / ".skills")
        count = manager.load_skills()
        assert count == 1

        skill = manager.get_skill("react_component")
        assert skill is not None
        assert "React" in skill.description

    def test_write_skill_and_execute(self, tmp_path) -> None:
        """Test writing a skill, loading it, and executing it."""
        from devmate.file_tools import create_file_tools
        from devmate.skills import SkillsManager

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        skills_dir = workspace / ".skills"

        tools = create_file_tools(workspace=workspace)
        create_file_tool = next(t for t in tools if t.name == "create_file")
        write_tool = next(t for t in tools if t.name == "write")

        # Create then update the skill file
        initial_content = '---\nname: "web_api"\ndescription: "placeholder"\n---\n'
        skill_folder = skills_dir / "web_api"
        create_file_tool.invoke(
            {
                "file_path": str(skill_folder / "SKILL.md"),
                "content": initial_content,
            }
        )

        updated_content = """---
name: "web_api"
description: "REST API design patterns"
trigger_keywords: ["api", "rest", "endpoint", "http"]
---

# REST API Design

Follow RESTful conventions for all API endpoints.
"""
        write_tool.invoke(
            {
                "file_path": str(skill_folder / "SKILL.md"),
                "content": updated_content,
            }
        )

        # Load and execute
        manager = SkillsManager(skills_dir=skills_dir)
        manager.load_skills()

        result = manager.execute_skill("web_api")
        assert "REST API Design" in result

        # Also test legacy keyword matching
        matches = manager.find_matching_skills("I need a REST API endpoint")
        assert len(matches) == 1
        assert matches[0].name == "web_api"


# ---------------------------------------------------------------------------
# 5. Config + RAG + Skills end-to-end
# ---------------------------------------------------------------------------


class TestConfigToModulesIntegration:
    """Test that configuration properly flows to all modules."""

    def test_config_drives_rag_initialization(self, tmp_path) -> None:
        """Test that RAG config values are used by RAGEngine."""
        from devmate.config import get_rag_config, load_config
        from devmate.rag import RAGEngine

        config_content = """
[model]
base_url = "https://example.com"
api_key = "test"

[rag]
docs_directory = "test_docs"
chroma_persist_directory = "test_chroma"
chunk_size = 500
chunk_overlap = 100

[mcp_server]
host = "localhost"
port = 8001
route = "/mcp"
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(config_content, encoding="utf-8")

        config = load_config(str(config_file))
        rag_config = get_rag_config(config)

        # Config values flow correctly
        assert rag_config["chunk_size"] == 500
        assert rag_config["chunk_overlap"] == 100

        # Create docs and ingest
        docs_dir = tmp_path / "test_docs"
        docs_dir.mkdir()
        (docs_dir / "guide.md").write_text(
            "# Guide\n\nSome content here.", encoding="utf-8"
        )

        persist_dir = tmp_path / "test_chroma"
        engine = RAGEngine(
            persist_directory=str(persist_dir),
            chunk_size=rag_config["chunk_size"],
            chunk_overlap=rag_config["chunk_overlap"],
        )
        count = engine.ingest_documents(docs_dir)
        assert count > 0

    def test_config_drives_skills_initialization(self, tmp_path) -> None:
        """Test that skills config values are used by SkillsManager."""
        from devmate.config import get_skills_config, load_config
        from devmate.skills import SkillsManager

        config_content = """
[model]
base_url = "https://example.com"
api_key = "test"

[skills]
directory = "custom_skills"

[mcp_server]
host = "localhost"
port = 8001
route = "/mcp"
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(config_content, encoding="utf-8")

        config = load_config(str(config_file))
        skills_config = get_skills_config(config)
        assert skills_config["directory"] == "custom_skills"

        # Create custom skills directory with folder structure
        custom_dir = tmp_path / "custom_skills"
        custom_dir.mkdir()
        skill_dir = custom_dir / "skill1"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            '---\nname: "skill1"\ndescription: "test"\n---\n# Skill 1\n',
            encoding="utf-8",
        )

        manager = SkillsManager(skills_dir=custom_dir)
        count = manager.load_skills()
        assert count == 1
