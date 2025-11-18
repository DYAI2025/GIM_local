#!/usr/bin/env python3
"""
Tests for Pydantic models and validation.
"""
import pytest
from pydantic import ValidationError
from models import ToolRequest, ToolResult, ConfigValidator


class TestToolRequest:
    """Test ToolRequest model validation."""

    def test_valid_list_dir_request(self):
        """Test valid list_dir tool request."""
        req = ToolRequest(tool="list_dir", args={"path": "."})
        assert req.tool == "list_dir"
        assert req.args["path"] == "."

    def test_valid_list_dir_no_args(self):
        """Test list_dir with no args (optional path)."""
        req = ToolRequest(tool="list_dir", args={})
        assert req.tool == "list_dir"
        assert req.args == {}

    def test_valid_read_file_request(self):
        """Test valid read_file tool request."""
        req = ToolRequest(tool="read_file", args={"path": "test.txt"})
        assert req.tool == "read_file"
        assert req.args["path"] == "test.txt"

    def test_invalid_read_file_missing_path(self):
        """Test read_file fails without path."""
        with pytest.raises(ValidationError) as exc_info:
            ToolRequest(tool="read_file", args={})
        assert "read_file requires" in str(exc_info.value)

    def test_valid_write_file_request(self):
        """Test valid write_file tool request."""
        req = ToolRequest(
            tool="write_file",
            args={"path": "output.txt", "content": "Hello World"}
        )
        assert req.tool == "write_file"
        assert req.args["path"] == "output.txt"
        assert req.args["content"] == "Hello World"

    def test_invalid_write_file_missing_content(self):
        """Test write_file fails without content."""
        with pytest.raises(ValidationError) as exc_info:
            ToolRequest(tool="write_file", args={"path": "test.txt"})
        assert "write_file requires" in str(exc_info.value)

    def test_invalid_tool_name(self):
        """Test invalid tool name is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ToolRequest(tool="delete_file", args={})
        # Pydantic will reject this due to Literal type constraint
        assert "tool" in str(exc_info.value).lower()

    def test_invalid_args_type(self):
        """Test non-dict args are rejected."""
        with pytest.raises(ValidationError):
            ToolRequest(tool="list_dir", args="not_a_dict")  # type: ignore


class TestToolResult:
    """Test ToolResult model."""

    def test_success_result(self):
        """Test successful tool result."""
        result = ToolResult(
            success=True,
            data={"items": ["file1.txt", "file2.txt"]}
        )
        assert result.success is True
        assert result.error is None
        assert len(result.data["items"]) == 2

    def test_error_result(self):
        """Test error tool result."""
        result = ToolResult(
            success=False,
            error="File not found"
        )
        assert result.success is False
        assert result.error == "File not found"
        assert result.data == {}


class TestConfigValidator:
    """Test configuration validation."""

    def test_default_config(self):
        """Test config with all defaults."""
        config = ConfigValidator()
        assert config.ollama_url == "http://127.0.0.1:11434"
        assert config.ollama_model == "llama3.1:8b"
        assert config.sample_rate == 16000
        assert config.vad_aggr == 2

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ConfigValidator(
            ollama_url="https://ollama.example.com",
            ollama_model="llama2:13b",
            sample_rate=44100,
            vad_aggr=3
        )
        assert config.ollama_url == "https://ollama.example.com"
        assert config.ollama_model == "llama2:13b"
        assert config.sample_rate == 44100
        assert config.vad_aggr == 3

    def test_invalid_url_scheme(self):
        """Test URL must start with http:// or https://."""
        with pytest.raises(ValidationError) as exc_info:
            ConfigValidator(ollama_url="ftp://example.com")
        assert "must start with http://" in str(exc_info.value)

    def test_invalid_vad_aggr_too_high(self):
        """Test VAD aggressiveness must be <= 3."""
        with pytest.raises(ValidationError) as exc_info:
            ConfigValidator(vad_aggr=4)
        assert "vad_aggr" in str(exc_info.value).lower()

    def test_invalid_vad_aggr_negative(self):
        """Test VAD aggressiveness must be >= 0."""
        with pytest.raises(ValidationError) as exc_info:
            ConfigValidator(vad_aggr=-1)
        assert "vad_aggr" in str(exc_info.value).lower()

    def test_invalid_timeout_zero(self):
        """Test timeout must be > 0."""
        with pytest.raises(ValidationError) as exc_info:
            ConfigValidator(ollama_start_timeout=0)
        assert "ollama_start_timeout" in str(exc_info.value).lower()

    def test_invalid_retry_max_zero(self):
        """Test retry_max must be > 0."""
        with pytest.raises(ValidationError) as exc_info:
            ConfigValidator(ollama_retry_max=0)
        assert "ollama_retry_max" in str(exc_info.value).lower()

    def test_url_trailing_slash_removed(self):
        """Test URL trailing slash is stripped."""
        config = ConfigValidator(ollama_url="http://localhost:11434/")
        assert config.ollama_url == "http://localhost:11434"
