"""End-to-end test script for DevMate.

This script validates the complete agent workflow:
1. Agent initialization (config, RAG, skills, MCP)
2. Agent calls search tools
3. Agent calls RAG retrieval
4. Agent generates files (html/css/js)
5. Code quality verification

Run with:
    python -m pytest tests/test_e2e.py -v -s

Or as a standalone script:
    python tests/test_e2e.py
"""

import logging
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------


def _run_standalone() -> int:
    """Run the e2e checks as a standalone script (no pytest)."""
    import asyncio

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    exit_code = 0

    async def _run_checks() -> None:
        nonlocal exit_code

        # --- Check 1: Config loading ---
        logger.info("[E2E] Check 1: Config loading")
        try:
            from devmate.config import load_config

            config = load_config()
            assert config.get("model", {}).get("api_key"), "Missing API key"
            assert config.get("search", {}).get("tavily_api_key"), "Missing Tavily key"
            logger.info("[E2E] Config loaded OK")
        except Exception as exc:
            logger.error("[E2E] FAIL: Config loading: %s", exc)
            exit_code = 1
            return

        # --- Check 2: RAG ingestion and retrieval ---
        logger.info("[E2E] Check 2: RAG pipeline")
        try:
            from devmate.rag import RAGEngine, create_search_tool

            with tempfile.TemporaryDirectory() as tmpdir:
                docs_dir = Path(tmpdir) / "docs"
                docs_dir.mkdir()

                (docs_dir / "guide.md").write_text(
                    "# Hiking Website Guide\n\n"
                    "## HTML Structure\n\n"
                    "Use semantic HTML5 elements for accessibility.\n\n"
                    "## CSS Styling\n\n"
                    "Use responsive design with flexbox and grid.\n\n"
                    "## JavaScript\n\n"
                    "Use modern ES6+ syntax.",
                    encoding="utf-8",
                )

                persist_dir = Path(tmpdir) / ".chroma_db"
                engine = RAGEngine(persist_directory=str(persist_dir))
                count = engine.ingest_documents(docs_dir)
                assert count > 0, "No chunks ingested"

                results = engine.search("HTML5 accessibility")
                assert len(results) > 0, "Search returned no results"
                assert any("semantic" in d.page_content.lower() for d in results), (
                    "Expected content not found in search results"
                )

                # Verify search tool works
                tool = create_search_tool(engine)
                tool_output = tool.invoke({"query": "CSS responsive design"})
                assert "flexbox" in tool_output or "grid" in tool_output, (
                    "Search tool output missing expected content"
                )

            logger.info("[E2E] RAG pipeline OK (ingested %d chunks)", count)
        except Exception as exc:
            logger.error("[E2E] FAIL: RAG pipeline: %s", exc)
            exit_code = 1

        # --- Check 3: Skills system ---
        logger.info("[E2E] Check 3: Skills system")
        try:
            from devmate.skills import SkillsManager

            with tempfile.TemporaryDirectory() as tmpdir:
                skills_dir = Path(tmpdir) / ".skills"
                skills_dir.mkdir()

                (skills_dir / "web_dev.md").write_text(
                    "---\n"
                    'name: "web_development"\n'
                    'description: "Web development patterns"\n'
                    'trigger_keywords: ["website", "web", "html", "css", "js"]\n'
                    "---\n\n"
                    "# Web Development\n\n"
                    "Use modern frameworks like React or Vue.\n",
                    encoding="utf-8",
                )

                manager = SkillsManager(skills_dir=skills_dir)
                count = manager.load_skills()
                assert count == 1, f"Expected 1 skill, got {count}"

                matches = manager.find_matching_skills("build a website")
                assert len(matches) == 1, "Keyword match failed"

            logger.info("[E2E] Skills system OK")
        except Exception as exc:
            logger.error("[E2E] FAIL: Skills system: %s", exc)
            exit_code = 1

        # --- Check 4: File tools ---
        logger.info("[E2E] Check 4: File tools")
        try:
            from devmate.file_tools import create_file_tools

            with tempfile.TemporaryDirectory() as tmpdir:
                workspace = Path(tmpdir) / "workspace"
                workspace.mkdir()
                tools = create_file_tools(workspace=workspace)

                create_file = next(t for t in tools if t.name == "create_file")
                write_file = next(t for t in tools if t.name == "write_file")
                list_dir = next(t for t in tools if t.name == "list_directory")

                # Create files
                create_file.invoke(
                    {
                        "file_path": "index.html",
                        "content": (
                            "<!DOCTYPE html><html><body>"
                            "<h1>Hiking Trails</h1></body></html>"
                        ),
                    }
                )
                assert (workspace / "index.html").exists(), "HTML file not created"

                create_file.invoke(
                    {
                        "file_path": "css/style.css",
                        "content": "body { font-family: sans-serif; margin: 0; }",
                    }
                )
                assert (workspace / "css" / "style.css").exists(), (
                    "CSS file not created"
                )

                create_file.invoke(
                    {
                        "file_path": "js/app.js",
                        "content": "console.log('Hiking app loaded');",
                    }
                )
                assert (workspace / "js" / "app.js").exists(), "JS file not created"

                # Update file
                write_file.invoke(
                    {
                        "file_path": "index.html",
                        "content": (
                            "<!DOCTYPE html><html><body>"
                            "<h1>Nearby Hiking Routes</h1></body></html>"
                        ),
                    }
                )
                content = (workspace / "index.html").read_text()
                assert "Nearby Hiking Routes" in content, "Write file content mismatch"

                # List directory
                listing = list_dir.invoke({"dir_path": "."})
                assert "index.html" in listing, "index.html not in directory listing"
                assert "css" in listing, "css dir not in directory listing"
                assert "js" in listing, "js dir not in directory listing"

            logger.info("[E2E] File tools OK")
        except Exception as exc:
            logger.error("[E2E] FAIL: File tools: %s", exc)
            exit_code = 1

        # --- Check 5: MCP Server ---
        logger.info("[E2E] Check 5: MCP Server")
        try:
            from mcp_server.server import create_mcp_app

            app = create_mcp_app(
                tavily_api_key="test-key",
                max_results=3,
                route="/mcp",
            )
            assert app is not None, "MCP app creation failed"

            from httpx import ASGITransport, AsyncClient

            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.get("/health")
            assert resp.status_code == 200, f"Health check failed: {resp.status_code}"
            assert resp.json()["status"] == "ok"

            logger.info("[E2E] MCP Server OK")
        except Exception as exc:
            logger.error("[E2E] FAIL: MCP Server: %s", exc)
            exit_code = 1

        # --- Check 6: Code quality ---
        logger.info("[E2E] Check 6: Code quality (PEP 8)")
        try:
            import subprocess

            project_root = Path(__file__).resolve().parent.parent

            # Check for print() statements in source code
            result = subprocess.run(
                ["grep", "-rn", "print(", "src/devmate/", "mcp_server/"],
                capture_output=True,
                text=True,
                cwd=str(project_root),
            )
            if result.returncode == 0:
                logger.error("[E2E] FAIL: Found print() statements:\n%s", result.stdout)
                exit_code = 1
            else:
                logger.info("[E2E] No print() statements found in source")

            # Run ruff format check
            result = subprocess.run(
                ["ruff", "format", "--check", "src/", "mcp_server/", "tests/"],
                capture_output=True,
                text=True,
                cwd=str(project_root),
            )
            if result.returncode != 0:
                logger.error("[E2E] FAIL: ruff format check:\n%s", result.stdout)
                exit_code = 1
            else:
                logger.info("[E2E] ruff format check passed")

            # Run ruff lint
            result = subprocess.run(
                ["ruff", "check", "--select", "E,W", "src/", "mcp_server/", "tests/"],
                capture_output=True,
                text=True,
                cwd=str(project_root),
            )
            if result.returncode != 0:
                logger.error("[E2E] FAIL: ruff lint check:\n%s", result.stdout)
                exit_code = 1
            else:
                logger.info("[E2E] ruff lint check passed")

        except Exception as exc:
            logger.error("[E2E] FAIL: Code quality check: %s", exc)
            exit_code = 1

        # --- Summary ---
        if exit_code == 0:
            logger.info("[E2E] All checks PASSED")
        else:
            logger.error("[E2E] Some checks FAILED")

    asyncio.run(_run_checks())
    return exit_code


if __name__ == "__main__":
    sys.exit(_run_standalone())


# ---------------------------------------------------------------------------
# Pytest-based e2e tests (for CI integration)
# ---------------------------------------------------------------------------


class TestEndToEndWorkflow:
    """Pytest-based end-to-end tests."""

    def test_config_to_rag_to_search_pipeline(self, tmp_path) -> None:
        """Full pipeline: config -> RAG ingest -> search."""
        from devmate.rag import RAGEngine, create_search_tool

        # Setup: create docs with hiking website content
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()

        (docs_dir / "hiking_guide.md").write_text(
            "# Hiking Trail Website Guide\n\n"
            "## HTML Structure\n\n"
            "Use semantic HTML5 elements.\n\n"
            "## CSS Styling\n\n"
            "Use responsive flexbox layouts.\n\n"
            "## JavaScript Interactivity\n\n"
            "Use ES6 modules for clean code organization.\n",
            encoding="utf-8",
        )

        # Ingest
        engine = RAGEngine(persist_directory=str(tmp_path / ".chroma_db"))
        chunk_count = engine.ingest_documents(docs_dir)
        assert chunk_count > 0

        # Search
        results = engine.search("responsive CSS flexbox")
        assert len(results) > 0

        # Via tool
        tool = create_search_tool(engine)
        output = tool.invoke({"query": "HTML5 semantic elements"})
        assert "semantic" in output

    def test_skills_match_hiking_query(self, tmp_path) -> None:
        """Test that skills are loaded from folder structure and can be queried."""
        from devmate.skills import SkillsManager

        skills_dir = tmp_path / ".skills"
        skills_dir.mkdir()

        # Create skill in folder structure: .skills/web_project/SKILL.md
        web_skill_dir = skills_dir / "web_project"
        web_skill_dir.mkdir()
        (web_skill_dir / "SKILL.md").write_text(
            "---\n"
            'name: "web_project"\n'
            'description: "Creating web projects with HTML, CSS, and JavaScript"\n'
            'trigger_keywords: ["website", "project", "build", "create"]\n'
            "---\n\n"
            "# Web Project Template\n\n"
            "Use HTML, CSS, and JavaScript to build the project.\n",
            encoding="utf-8",
        )

        manager = SkillsManager(skills_dir=skills_dir)
        manager.load_skills()

        # Verify skill is loaded by name
        skill = manager.get_skill("web_project")
        assert skill is not None
        assert "web projects" in skill.description.lower()

        # The e2e prompt is about building a hiking website
        matches = manager.find_matching_skills(
            "build a website showing nearby hiking routes"
        )
        assert len(matches) == 1
        assert matches[0].name == "web_project"

    def test_file_generation_workflow(self, tmp_path) -> None:
        """Test creating html/css/js files for the hiking website."""
        from devmate.file_tools import create_file_tools

        workspace = tmp_path / "output"
        workspace.mkdir()
        tools = create_file_tools(workspace=workspace)

        create_file = next(t for t in tools if t.name == "create_file")
        list_dir = next(t for t in tools if t.name == "list_directory")

        # Create project files
        create_file.invoke(
            {
                "file_path": "index.html",
                "content": (
                    "<!DOCTYPE html>\n"
                    "<html lang='en'>\n"
                    "<head><meta charset='utf-8'><title>Nearby Hiking Routes</title>"
                    '<link rel="stylesheet" href="css/style.css"></head>\n'
                    "<body><h1>Nearby Hiking Routes</h1><div id='app'></div>"
                    '<script src="js/app.js"></script></body>\n'
                    "</html>"
                ),
            }
        )

        create_file.invoke(
            {
                "file_path": "css/style.css",
                "content": (
                    "body { font-family: 'Segoe UI', sans-serif; margin: 0; "
                    "padding: 20px; background: #f5f5f5; }\n"
                    "h1 { color: #2d5016; }\n"
                    ".trail-card { border: 1px solid #ddd; border-radius: 8px; "
                    "padding: 16px; margin: 12px 0; }\n"
                ),
            }
        )

        create_file.invoke(
            {
                "file_path": "js/app.js",
                "content": (
                    "'use strict';\n"
                    "const trails = [\n"
                    '  { name: "Mountain View Trail", distance: "5.2 km" },\n'
                    '  { name: "Lakeside Path", distance: "3.1 km" },\n'
                    "];\n"
                    "function renderTrails() {\n"
                    '  const app = document.getElementById("app");\n'
                    "  trails.forEach(t => {\n"
                    "    const card = document.createElement('div');\n"
                    "    card.className = 'trail-card';\n"
                    "    card.textContent = `${t.name} - ${t.distance}`;\n"
                    "    app.appendChild(card);\n"
                    "  });\n"
                    "}\n"
                    "renderTrails();\n"
                ),
            }
        )

        # Verify all files exist
        assert (workspace / "index.html").exists()
        assert (workspace / "css" / "style.css").exists()
        assert (workspace / "js" / "app.js").exists()

        # Verify content quality
        html_content = (workspace / "index.html").read_text()
        assert "<!DOCTYPE html>" in html_content
        assert "Hiking" in html_content or "hiking" in html_content

        css_content = (workspace / "css" / "style.css").read_text()
        assert "font-family" in css_content
        assert "margin" in css_content

        js_content = (workspace / "js" / "app.js").read_text()
        assert "'use strict'" in js_content
        assert "const" in js_content or "function" in js_content

        # Verify directory listing
        listing = list_dir.invoke({"dir_path": "."})
        assert "index.html" in listing
        assert "css" in listing
        assert "js" in listing

    def test_no_print_statements_in_source(self) -> None:
        """Verify no print() statements in source code.

        Note: We grep only .py files (not .pyc) and exclude lines that are
        inside string literals (e.g. the SYSTEM_PROMPT which mentions
        print() in documentation text). A simple heuristic: only flag lines
        that start with whitespace followed by ``print(``.
        """
        import subprocess

        project_root = Path(__file__).resolve().parent.parent

        # Use grep with --include to skip binary/cache files, and match
        # lines that look like actual Python print() calls (indented code)
        result = subprocess.run(
            [
                "grep",
                "-rn",
                "--include=*.py",
                r"^\s*print\s*\(",
                "src/devmate/",
                "mcp_server/",
            ],
            capture_output=True,
            text=True,
            cwd=str(project_root),
        )
        assert result.returncode != 0, (
            f"Found print() statements in source code:\n{result.stdout}"
        )
