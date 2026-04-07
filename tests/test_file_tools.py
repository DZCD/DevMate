"""Tests for the file_tools module."""

from devmate.file_tools import create_file_tools


def test_create_file_creates_new_file(tmp_path) -> None:
    """Test create_file tool creates a new file with content."""
    tools = create_file_tools(workspace=tmp_path)
    create_file = next(t for t in tools if t.name == "create_file")

    result = create_file.invoke(
        {
            "file_path": "new_dir/hello.py",
            "content": "print('hello')\n",
        }
    )

    assert "Successfully created" in result
    created = tmp_path / "new_dir" / "hello.py"
    assert created.exists()
    assert created.read_text(encoding="utf-8") == "print('hello')\n"


def test_create_file_rejects_overwrite_by_default(tmp_path) -> None:
    """Test create_file refuses to overwrite existing files."""
    tools = create_file_tools(workspace=tmp_path)
    create_file = next(t for t in tools if t.name == "create_file")

    # Create a file first
    (tmp_path / "existing.txt").write_text("original", encoding="utf-8")

    result = create_file.invoke(
        {
            "file_path": "existing.txt",
            "content": "overwritten",
        }
    )

    assert "already exists" in result
    assert (tmp_path / "existing.txt").read_text(encoding="utf-8") == "original"


def test_create_file_overwrite_when_flagged(tmp_path) -> None:
    """Test create_file overwrites when overwrite=True."""
    tools = create_file_tools(workspace=tmp_path)
    create_file = next(t for t in tools if t.name == "create_file")

    (tmp_path / "existing.txt").write_text("original", encoding="utf-8")

    result = create_file.invoke(
        {
            "file_path": "existing.txt",
            "content": "overwritten",
            "overwrite": True,
        }
    )

    assert "Successfully" in result
    assert (tmp_path / "existing.txt").read_text(encoding="utf-8") == "overwritten"


def test_create_file_rejects_path_outside_workspace(tmp_path) -> None:
    """Test create_file rejects paths outside workspace."""
    tools = create_file_tools(workspace=tmp_path)
    create_file = next(t for t in tools if t.name == "create_file")

    result = create_file.invoke(
        {
            "file_path": "/etc/passwd",
            "content": "malicious",
        }
    )

    assert "outside the workspace" in result


def test_write_file_updates_existing_file(tmp_path) -> None:
    """Test write_file updates an existing file."""
    tools = create_file_tools(workspace=tmp_path)
    write_file = next(t for t in tools if t.name == "write_file")

    target = tmp_path / "update.txt"
    target.write_text("old content", encoding="utf-8")

    result = write_file.invoke(
        {
            "file_path": "update.txt",
            "content": "new content",
        }
    )

    assert "Successfully wrote" in result
    assert target.read_text(encoding="utf-8") == "new content"


def test_write_file_rejects_nonexistent_file(tmp_path) -> None:
    """Test write_file refuses to create new files."""
    tools = create_file_tools(workspace=tmp_path)
    write_file = next(t for t in tools if t.name == "write_file")

    result = write_file.invoke(
        {
            "file_path": "nonexistent.txt",
            "content": "content",
        }
    )

    assert "does not exist" in result


def test_write_file_rejects_path_outside_workspace(tmp_path) -> None:
    """Test write_file rejects paths outside workspace."""
    tools = create_file_tools(workspace=tmp_path)
    write_file = next(t for t in tools if t.name == "write_file")

    result = write_file.invoke(
        {
            "file_path": "/tmp/something.txt",
            "content": "content",
        }
    )

    assert "outside the workspace" in result


def test_list_directory_default(tmp_path) -> None:
    """Test list_directory shows workspace contents."""
    # Create some files
    (tmp_path / "file1.txt").write_text("a", encoding="utf-8")
    (tmp_path / "file2.py").write_text("b", encoding="utf-8")
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "nested.txt").write_text("c", encoding="utf-8")

    tools = create_file_tools(workspace=tmp_path)
    list_dir = next(t for t in tools if t.name == "list_directory")

    result = list_dir.invoke({"dir_path": "."})

    assert "file1.txt" in result
    assert "file2.py" in result
    assert "subdir" in result
    assert "[DIR]" in result
    assert "[FILE]" in result


def test_list_directory_subdirectory(tmp_path) -> None:
    """Test listing a subdirectory."""
    tools = create_file_tools(workspace=tmp_path)
    list_dir = next(t for t in tools if t.name == "list_directory")

    result = list_dir.invoke({"dir_path": "subdir"})
    assert "does not exist" in result  # doesn't exist yet


def test_list_directory_nonexistent(tmp_path) -> None:
    """Test listing a nonexistent directory."""
    tools = create_file_tools(workspace=tmp_path)
    list_dir = next(t for t in tools if t.name == "list_directory")

    result = list_dir.invoke({"dir_path": "no_such_dir"})
    assert "does not exist" in result


def test_list_directory_not_a_directory(tmp_path) -> None:
    """Test listing a file path (not a directory)."""
    tools = create_file_tools(workspace=tmp_path)
    list_dir = next(t for t in tools if t.name == "list_directory")

    (tmp_path / "notafile.txt").write_text("data", encoding="utf-8")

    result = list_dir.invoke({"dir_path": "notafile.txt"})
    assert "not a directory" in result


def test_list_directory_empty(tmp_path) -> None:
    """Test listing an empty directory."""
    empty = tmp_path / "empty_dir"
    empty.mkdir()

    tools = create_file_tools(workspace=tmp_path)
    list_dir = next(t for t in tools if t.name == "list_directory")

    result = list_dir.invoke({"dir_path": "empty_dir"})
    assert "empty" in result.lower()


def test_tools_count() -> None:
    """Test that create_file_tools returns exactly 3 tools."""
    tools = create_file_tools(workspace="/tmp")
    tool_names = [t.name for t in tools]
    assert len(tools) == 3
    assert "create_file" in tool_names
    assert "write_file" in tool_names
    assert "list_directory" in tool_names
