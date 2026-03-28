"""tests/tool_infra_tests/test_export_functionality.py

Tests for BaseToolOutput export functionality."""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import List

import pytest
from pydantic import Field

from bio_programming_tools.utils.tool_io import BaseToolOutput

logger = logging.getLogger(__name__)


# ── Helper functions ─────────────────────────────────────────────────────────


def validate_export_output(export_path: Path) -> bool:
    """
    Validate that an export output exists and is not empty.

    This helper checks whether the exported output (file or directory) exists
    and contains actual data. Used in tests to verify export functionality.

    Args:
        export_path: Path to the exported file or directory

    Returns:
        True if export exists and is not empty, False otherwise

    Examples:
        >>> # Test file export
        >>> result.export(name="output", export_path=tmp_dir, file_format="csv")
        >>> assert validate_export_output(Path(tmp_dir) / "output.csv")

        >>> # Test directory export
        >>> result.export(name="output", export_path=tmp_dir)
        >>> assert validate_export_output(Path(tmp_dir) / "output")
    """
    if not export_path.exists():
        logger.warning(f"Export path {export_path} does not exist")
        return False

    if export_path.is_file():
        # For files, check that size > 0
        size = export_path.stat().st_size
        if size == 0:
            logger.warning(f"Export path exists and is a file, but the file is empty (size {size} bytes)")
            return False

        logger.debug(f"Valid export output file found at {export_path} with size {size} bytes")
        return True

    elif export_path.is_dir():
        # For directories, check that at least one non-empty file exists
        files = list(export_path.rglob("*"))
        non_empty_files = [f for f in files if f.is_file() and f.stat().st_size > 0]
        if len(non_empty_files) == 0:
            logger.warning(f"Export path exists and is a directory, but no non-empty files found")
            return False

        logger.debug(f"Valid export output directory found at {export_path} with {len(non_empty_files)} non-empty files")
        return True

    logger.warning(f"Export path exists, but is not a file or directory: {export_path}")
    return False


def validate_output(output: BaseToolOutput, check_export: bool = True):
    """
    Validate tool output and test export functionality.

    This is a standardized helper for testing tool outputs. It checks that:
    1. The tool execution was successful (success=True)
    2. The export functionality works correctly (if check_export=True)

    Args:
        output: The tool output to validate
        check_export: Whether to validate export functionality (default: True).
                     Set to False for outputs with no data to export (e.g., no hits).

    Raises:
        AssertionError: If the output is invalid or export fails

    Examples:
        >>> result = some_tool(inputs, config)
        >>> validate_output(result)  # Checks success and export

        >>> result_no_hits = some_tool(inputs, config)
        >>> validate_output(result_no_hits, check_export=False)  # Only checks success
    """
    # Check tool execution was successful
    assert output.success is True, f"Tool execution failed: {output}"

    # Test export functionality with a temporary directory
    if check_export:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output.export(name="test_output", export_path=tmp_dir)
            export_path = Path(tmp_dir) / "test_output"
            tool_name = output.tool_id if output.tool_id else "unknown tool"

            # Pattern 1: Directory export (path is a directory with files inside)
            if validate_export_output(export_path):
                return

            # Pattern 2: Single-file export (path with file extension appended)
            ext_path = export_path.with_suffix(f".{output.output_format_default}")
            if validate_export_output(ext_path):
                return

            # Pattern 3: Sibling-file export (multiple files in parent dir with name prefix)
            parent_dir = export_path.parent
            sibling_files = [
                f for f in parent_dir.glob(f"{export_path.stem}*")
                if f.is_file() and f.stat().st_size > 0
            ]
            if sibling_files:
                return

            assert False, f"Export validation failed for {tool_name}"


# ── Mock tool outputs ────────────────────────────────────────────────────────


class MockToolOutputBase(BaseToolOutput):
    """
    Base class for mock tool outputs in tests.

    This class provides no-op implementations of the export abstract methods,
    allowing test mock classes to inherit from BaseToolOutput without having
    to implement export functionality.

    Test classes that need to test actual export functionality should override
    these methods.
    """

    @property
    def output_format_options(self) -> List[str]:
        return []

    @property
    def output_format_default(self) -> str:
        return ""

    def _export_output(self, export_path: Path, file_format: str):
        """No-op export for mock outputs."""
        pass


class MockToolOutput(BaseToolOutput):
    """Mock tool output for testing export functionality."""

    data: List[str] = Field(
        default_factory=list,
        description="Mock data for testing"
    )

    @property
    def output_format_options(self) -> List[str]:
        return ["txt", "json", "csv"]

    @property
    def output_format_default(self) -> str:
        return "txt"

    def _export_output(self, export_path: Path, file_format: str):
        """Export mock data in specified format."""
        path = Path(export_path).with_suffix(f".{file_format}")
        path.parent.mkdir(parents=True, exist_ok=True)

        if file_format == "txt":
            with open(path, "w") as f:
                for item in self.data:
                    f.write(f"{item}\n")

        elif file_format == "json":
            with open(path, "w") as f:
                json.dump({"data": self.data}, f, indent=2)

        elif file_format == "csv":
            with open(path, "w") as f:
                f.write("value\n")
                for item in self.data:
                    f.write(f"{item}\n")


# ── validate_output helper ───────────────────────────────────────────────────


def test_validate_output_successful():
    """Test validate_output with successful tool output."""
    output = MockToolOutput(
        success=True,
        data=["item1", "item2", "item3"]
    )

    # Should not raise any assertions
    validate_output(output)


def test_validate_output_failed_execution():
    """Test validate_output fails when tool execution failed."""
    output = MockToolOutput(
        success=False,
        data=[]
    )

    with pytest.raises(AssertionError, match="Tool execution failed"):
        validate_output(output)


def test_validate_output_with_empty_export():
    """Test validate_output fails when export creates empty output."""
    class _EmptyExportOutput(BaseToolOutput):
        @property
        def output_format_options(self) -> List[str]:
            return ["txt"]

        @property
        def output_format_default(self) -> str:
            return "txt"

        def _export_output(self, export_path: Path, file_format: str):
            # Create empty file
            path = Path(export_path).with_suffix(f".{file_format}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()  # Empty file

    output = _EmptyExportOutput(success=True)

    with pytest.raises(AssertionError, match="Export validation failed"):
        validate_output(output)


# ── validate_export_output helper ────────────────────────────────────────────


def test_validate_nonexistent_path():
    """Test validation fails for nonexistent path."""
    result = validate_export_output(Path("/nonexistent/path"))
    assert result is False


def test_validate_empty_file(tmp_path):
    """Test validation fails for empty file."""
    empty_file = tmp_path / "empty.txt"
    empty_file.touch()

    result = validate_export_output(empty_file)
    assert result is False


def test_validate_non_empty_file(tmp_path):
    """Test validation succeeds for non-empty file."""
    file = tmp_path / "data.txt"
    file.write_text("test data")

    result = validate_export_output(file)
    assert result is True


def test_validate_empty_directory(tmp_path):
    """Test validation fails for empty directory."""
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()

    result = validate_export_output(empty_dir)
    assert result is False


def test_validate_directory_with_empty_file(tmp_path):
    """Test validation fails for directory with only empty files."""
    dir_with_empty = tmp_path / "dir"
    dir_with_empty.mkdir()
    (dir_with_empty / "empty.txt").touch()

    result = validate_export_output(dir_with_empty)
    assert result is False


def test_validate_directory_with_data(tmp_path):
    """Test validation succeeds for directory with non-empty files."""
    dir_with_data = tmp_path / "dir"
    dir_with_data.mkdir()
    (dir_with_data / "data.txt").write_text("test data")

    result = validate_export_output(dir_with_data)
    assert result is True


def test_validate_nested_directory_structure(tmp_path):
    """Test validation succeeds for nested directory with data."""
    nested = tmp_path / "parent" / "child"
    nested.mkdir(parents=True)
    (nested / "data.txt").write_text("nested data")

    result = validate_export_output(tmp_path / "parent")
    assert result is True


# ── BaseToolOutput.export() ─────────────────────────────────────────────────


def test_export_with_default_format(tmp_path):
    """Test exporting with default format."""
    output = MockToolOutput(
        success=True,
        data=["item1", "item2", "item3"]
    )

    output.export(name="test_output", export_path=tmp_path)

    # Should create test_output.txt (default format is txt)
    exported = tmp_path / "test_output.txt"
    assert validate_export_output(exported)


def test_export_with_custom_format(tmp_path):
    """Test exporting with specific format."""
    output = MockToolOutput(
        success=True,
        data=["item1", "item2", "item3"]
    )

    output.export(name="test_output", export_path=tmp_path, file_format="json")

    exported = tmp_path / "test_output.json"
    assert validate_export_output(exported)

    # Verify JSON content
    with open(exported) as f:
        data = json.load(f)
    assert "data" in data
    assert len(data["data"]) == 3


def test_export_all_supported_formats(tmp_path):
    """Test exporting with all supported formats."""
    output = MockToolOutput(
        success=True,
        data=["item1", "item2"]
    )

    for fmt in output.output_format_options:
        export_dir = tmp_path / fmt
        export_dir.mkdir()
        output.export(name="test_output", export_path=export_dir, file_format=fmt)

        exported = export_dir / f"test_output.{fmt}"
        assert validate_export_output(exported), f"Export failed for format: {fmt}"


def test_export_invalid_format_raises_error(tmp_path):
    """Test that invalid format raises ValueError."""
    output = MockToolOutput(
        success=True,
        data=["item1"]
    )

    with pytest.raises(ValueError, match="Invalid file format"):
        output.export(name="test_output", export_path=tmp_path, file_format="invalid")


def test_export_without_export_path(tmp_path):
    """Test export without specifying export_path uses cwd."""
    output = MockToolOutput(
        success=True,
        data=["item1"]
    )

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        output.export(name="test_output")

        exported = tmp_path / "test_output.txt"
        assert validate_export_output(exported)
    finally:
        os.chdir(original_cwd)


def test_export_creates_parent_directories(tmp_path):
    """Test that export creates parent directories if they don't exist."""
    output = MockToolOutput(
        success=True,
        data=["item1"]
    )

    nested_path = tmp_path / "parent" / "child"
    # Don't create the directory - export should handle it

    output.export(name="test_output", export_path=nested_path)

    exported = nested_path / "test_output.txt"
    assert validate_export_output(exported)


def test_export_empty_data(tmp_path):
    """Test exporting with empty data still creates output."""
    output = MockToolOutput(
        success=True,
        data=[]
    )

    output.export(name="test_output", export_path=tmp_path)

    # Export should exist even with empty data (single-file export creates test_output.txt)
    exported = tmp_path / "test_output.txt"
    assert exported.exists()


# ── Multi-file exports ──────────────────────────────────────────────────────


class _MultiFileOutput(BaseToolOutput):
    """Mock output that creates multiple files."""

    files: dict = Field(default_factory=dict)

    @property
    def output_format_options(self) -> List[str]:
        return ["multi"]

    @property
    def output_format_default(self) -> str:
        return "multi"

    def _export_output(self, export_path: Path, file_format: str):
        path = Path(export_path)
        if not path.is_dir():
            path.mkdir(parents=True)

        for filename, content in self.files.items():
            (path / filename).write_text(content)


def test_multi_file_export(tmp_path):
    """Test validation works with multi-file exports."""
    output = _MultiFileOutput(
        success=True,
        files={
            "file1.txt": "content1",
            "file2.txt": "content2",
            "file3.txt": "content3"
        }
    )

    output.export(name="test_output", export_path=tmp_path)

    exported = tmp_path / "test_output"
    assert validate_export_output(exported)

    # Verify all files exist
    assert (exported / "file1.txt").exists()
    assert (exported / "file2.txt").exists()
    assert (exported / "file3.txt").exists()
