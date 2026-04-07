"""Tests for the Agent module."""

from unittest.mock import AsyncMock, MagicMock, patch

from devmate.llm import LLMResponse, TextBlock
from tests.helpers import build_agent, write_minimal_config


class TestAgentBasics:
    """Basic agent tests."""

    def test_agent_module_version(self) -> None:
        """Test that __init__.py exports version."""
        import devmate

        assert hasattr(devmate, "__version__")
        assert isinstance(devmate.__version__, str)

    def test_create_agent_func_returns_instance(self, tmp_path) -> None:
        """Test create_agent_func factory function."""
        from devmate.agent import DevMateAgent, create_agent_func

        config_path = str(write_minimal_config(tmp_path))
        agent = create_agent_func(config_path=config_path, workspace=str(tmp_path))

        assert isinstance(agent, DevMateAgent)
        assert agent._config is not None
        assert agent._llm is None
        assert agent._storage is None
        assert agent._tool_registry is None

    async def test_initialize_sets_up_components(self, tmp_path) -> None:
        """Test that initialize() sets up all components."""
        from devmate.agent import DevMateAgent

        config_path = str(write_minimal_config(tmp_path))

        with (
            patch("devmate.agent.RAGEngine") as mock_rag_cls,
            patch("devmate.agent.SkillsManager") as mock_skills_cls,
            patch("devmate.agent.OpenAICompatibleAdapter") as mock_llm_cls,
        ):
            mock_rag_instance = MagicMock()
            mock_rag_instance.ingest_documents.return_value = 5
            mock_rag_cls.return_value = mock_rag_instance

            mock_skills_instance = MagicMock()
            mock_skills_instance.load_skills.return_value = 2
            mock_skills_instance.create_tools.return_value = []
            mock_skills_instance.get_skill_meta.return_value = ""
            mock_skills_cls.return_value = mock_skills_instance

            agent = DevMateAgent(config_path=config_path, workspace=str(tmp_path))
            await agent.initialize()

            # Verify all components were initialized
            assert agent._llm is not None
            assert agent._rag_engine is not None
            assert agent._skills_manager is not None
            assert agent._tool_registry is not None
            assert len(agent._tools) > 0
            mock_llm_cls.assert_called_once()
            mock_rag_cls.assert_called_once()
            mock_skills_cls.assert_called_once()

    async def test_initialize_continues_on_rag_failure(self, tmp_path) -> None:
        """Test agent initialization continues when RAG init fails."""
        from devmate.agent import DevMateAgent

        config_path = str(write_minimal_config(tmp_path))

        with (
            patch("devmate.agent.RAGEngine") as mock_rag_cls,
            patch("devmate.agent.SkillsManager") as mock_skills_cls,
            patch("devmate.agent.OpenAICompatibleAdapter"),
        ):
            # First call raises (with API key), second call succeeds (fallback)
            mock_rag_instance_fallback = MagicMock()
            mock_rag_instance_fallback.ingest_documents.return_value = 0
            mock_rag_cls.side_effect = [
                RuntimeError("RAG error"),
                mock_rag_instance_fallback,
            ]

            mock_skills_instance = MagicMock()
            mock_skills_instance.load_skills.return_value = 0
            mock_skills_instance.create_tools.return_value = []
            mock_skills_instance.get_skill_meta.return_value = ""
            mock_skills_cls.return_value = mock_skills_instance

            agent = DevMateAgent(config_path=config_path, workspace=str(tmp_path))
            await agent.initialize()

            assert agent._tool_registry is not None

    async def test_run_returns_response(self, tmp_path) -> None:
        """Test run() returns agent response."""
        agent = await build_agent(tmp_path)

        mock_response = LLMResponse(
            content=[TextBlock(text="Hello from DevMate!")],
            finish_reason="stop",
        )
        agent._llm.chat = AsyncMock(return_value=mock_response)

        result = await agent.run("Hello!")
        assert "Hello from DevMate!" in result

    async def test_run_handles_exception(self, tmp_path) -> None:
        """Test run() returns error message on exception."""
        agent = await build_agent(tmp_path)

        agent._llm.chat = AsyncMock(side_effect=RuntimeError("LLM error"))

        result = await agent.run("test")
        assert "Error" in result
        assert "LLM error" in result

    async def test_cleanup_completes(self, tmp_path) -> None:
        """Test cleanup() completes without errors."""
        from devmate.agent import DevMateAgent

        config_path = str(write_minimal_config(tmp_path))
        agent = DevMateAgent(config_path=config_path, workspace=str(tmp_path))
        await agent.cleanup()
