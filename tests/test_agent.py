"""Tests for the Agent module."""

from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage


class TestAgentBasics:
    """Basic agent tests."""

    def _create_minimal_config(self, tmp_path) -> str:
        """Create a minimal config.toml for testing."""
        config_content = """
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
directory = ".skills"

[rag]
docs_directory = "docs"
chroma_persist_directory = ".chroma_db"

[mcp_server]
host = "localhost"
port = 18001
route = "/mcp"
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(config_content, encoding="utf-8")
        return str(config_file)

    def test_agent_module_version(self) -> None:
        """Test that __init__.py exports version."""
        import devmate

        assert hasattr(devmate, "__version__")
        assert isinstance(devmate.__version__, str)

    def test_create_agent_func_returns_instance(self, tmp_path) -> None:
        """Test create_agent_func factory function."""
        from devmate.agent import DevMateAgent, create_agent_func

        config_path = self._create_minimal_config(tmp_path)
        agent = create_agent_func(config_path=config_path, workspace=str(tmp_path))

        assert isinstance(agent, DevMateAgent)
        assert agent._config is not None
        assert agent._agent is None
        assert agent._llm is None
        assert agent._rag_engine is None
        assert agent._skills_manager is None
        assert agent._tools == []

    async def test_initialize_sets_up_components(self, tmp_path) -> None:
        """Test that initialize() sets up all components."""
        from devmate.agent import DevMateAgent

        config_path = self._create_minimal_config(tmp_path)

        with (
            patch("devmate.agent.ChatAnthropic") as mock_llm_cls,
            patch("devmate.agent.RAGEngine") as mock_rag_cls,
            patch("devmate.agent.SkillsManager") as mock_skills_cls,
            patch(
                "langchain_mcp_adapters.client.MultiServerMCPClient",
                new_callable=MagicMock,
            ) as mock_mcp_cls,
        ):
            mock_rag_instance = MagicMock()
            mock_rag_instance.ingest_documents.return_value = 5
            mock_rag_cls.return_value = mock_rag_instance

            mock_skills_instance = MagicMock()
            mock_skills_instance.load_skills.return_value = 2
            mock_skills_instance.create_tools.return_value = []
            mock_skills_cls.return_value = mock_skills_instance

            mock_mcp_instance = MagicMock()
            mock_mcp_instance.get_tools = AsyncMock(return_value=[])
            mock_mcp_cls.return_value = mock_mcp_instance

            agent = DevMateAgent(config_path=config_path, workspace=str(tmp_path))
            await agent.initialize()

            # Verify all components were initialized
            assert agent._llm is not None
            assert agent._rag_engine is not None
            assert agent._skills_manager is not None
            assert agent._agent is not None
            mock_llm_cls.assert_called_once()
            mock_rag_cls.assert_called_once()
            mock_skills_cls.assert_called_once()

    async def test_initialize_continues_on_mcp_failure(self, tmp_path) -> None:
        """Test agent initialization continues when MCP connection fails."""
        from devmate.agent import DevMateAgent

        config_path = self._create_minimal_config(tmp_path)

        with (
            patch("devmate.agent.ChatAnthropic"),
            patch("devmate.agent.RAGEngine") as mock_rag_cls,
            patch("devmate.agent.SkillsManager") as mock_skills_cls,
            patch(
                "langchain_mcp_adapters.client.MultiServerMCPClient",
                new_callable=MagicMock,
            ) as mock_mcp_cls,
        ):
            mock_rag_instance = MagicMock()
            mock_rag_instance.ingest_documents.return_value = 0
            mock_rag_cls.return_value = mock_rag_instance

            mock_skills_instance = MagicMock()
            mock_skills_instance.load_skills.return_value = 0
            mock_skills_instance.create_tools.return_value = []
            mock_skills_cls.return_value = mock_skills_instance

            mock_mcp_cls.side_effect = ConnectionRefusedError("refused")

            agent = DevMateAgent(config_path=config_path, workspace=str(tmp_path))
            await agent.initialize()

            assert agent._agent is not None
            assert agent._mcp_client is None

    async def test_run_returns_response(self, tmp_path) -> None:
        """Test run() returns agent response."""
        from devmate.agent import DevMateAgent

        config_path = self._create_minimal_config(tmp_path)

        with (
            patch("devmate.agent.ChatAnthropic"),
            patch("devmate.agent.RAGEngine") as mock_rag_cls,
            patch("devmate.agent.SkillsManager") as mock_skills_cls,
            patch(
                "langchain_mcp_adapters.client.MultiServerMCPClient",
                new_callable=MagicMock,
            ),
        ):
            mock_rag_instance = MagicMock()
            mock_rag_instance.ingest_documents.return_value = 0
            mock_rag_cls.return_value = mock_rag_instance

            mock_skills_instance = MagicMock()
            mock_skills_instance.load_skills.return_value = 0
            mock_skills_instance.create_tools.return_value = []
            mock_skills_cls.return_value = mock_skills_instance

            agent = DevMateAgent(config_path=config_path, workspace=str(tmp_path))
            await agent.initialize()

            # Mock the compiled graph's ainvoke
            ai_msg = AIMessage(content="Hello from DevMate!")
            agent._agent.ainvoke = AsyncMock(return_value={"messages": [ai_msg]})

            result = await agent.run("Hello!")
            assert result == "Hello from DevMate!"

    async def test_run_handles_exception(self, tmp_path) -> None:
        """Test run() returns error message on exception."""
        from devmate.agent import DevMateAgent

        config_path = self._create_minimal_config(tmp_path)

        with (
            patch("devmate.agent.ChatAnthropic"),
            patch("devmate.agent.RAGEngine") as mock_rag_cls,
            patch("devmate.agent.SkillsManager") as mock_skills_cls,
            patch(
                "langchain_mcp_adapters.client.MultiServerMCPClient",
                new_callable=MagicMock,
            ),
        ):
            mock_rag_instance = MagicMock()
            mock_rag_instance.ingest_documents.return_value = 0
            mock_rag_cls.return_value = mock_rag_instance

            mock_skills_instance = MagicMock()
            mock_skills_instance.load_skills.return_value = 0
            mock_skills_instance.create_tools.return_value = []
            mock_skills_cls.return_value = mock_skills_instance

            agent = DevMateAgent(config_path=config_path, workspace=str(tmp_path))
            await agent.initialize()

            agent._agent.ainvoke = AsyncMock(side_effect=RuntimeError("LLM error"))

            result = await agent.run("test")
            assert "Error" in result
            assert "LLM error" in result

    async def test_cleanup_completes(self, tmp_path) -> None:
        """Test cleanup() completes without errors."""
        from devmate.agent import DevMateAgent

        config_path = self._create_minimal_config(tmp_path)
        agent = DevMateAgent(config_path=config_path, workspace=str(tmp_path))
        await agent.cleanup()
