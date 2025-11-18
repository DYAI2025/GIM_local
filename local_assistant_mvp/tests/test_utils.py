#!/usr/bin/env python3
"""
Tests for utility functions.
"""
import pytest
import pathlib
import tempfile
import shutil
from utils import (
    setup_workspace,
    safe_path,
    list_dir,
    read_file,
    write_file,
    save_wav
)


class TestWorkspace:
    """Test workspace setup and path safety."""

    def test_setup_workspace(self, tmp_path):
        """Test workspace creation."""
        workspace = setup_workspace(str(tmp_path / "test_workspace"))
        assert workspace.exists()
        assert workspace.is_dir()

    def test_setup_workspace_already_exists(self, tmp_path):
        """Test workspace setup when dir already exists."""
        workspace_path = tmp_path / "existing"
        workspace_path.mkdir()

        workspace = setup_workspace(str(workspace_path))
        assert workspace.exists()
        assert workspace == workspace_path

    def test_safe_path_valid(self, tmp_path):
        """Test safe_path accepts paths within workspace."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        result = safe_path("subdir/file.txt", workspace)
        assert str(result).startswith(str(workspace))

    def test_safe_path_absolute_within_workspace(self, tmp_path):
        """Test safe_path with absolute path inside workspace."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        subdir = workspace / "subdir"

        result = safe_path(str(subdir), workspace)
        assert str(result).startswith(str(workspace))

    def test_safe_path_rejects_escape_attempt(self, tmp_path):
        """Test safe_path rejects path traversal attacks."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        with pytest.raises(PermissionError, match="outside sandbox"):
            safe_path("../../etc/passwd", workspace)

    def test_safe_path_rejects_absolute_outside(self, tmp_path):
        """Test safe_path rejects absolute path outside workspace."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        with pytest.raises(PermissionError, match="outside sandbox"):
            safe_path("/etc/passwd", workspace)


class TestListDir:
    """Test directory listing function."""

    def test_list_empty_directory(self, tmp_path):
        """Test listing an empty directory."""
        result = list_dir(".", tmp_path)

        assert "cwd" in result
        assert "items" in result
        assert result["items"] == []

    def test_list_directory_with_files(self, tmp_path):
        """Test listing directory with files and subdirs."""
        # Create test structure
        (tmp_path / "file1.txt").write_text("content1")
        (tmp_path / "file2.txt").write_text("content2")
        (tmp_path / "subdir").mkdir()

        result = list_dir(".", tmp_path)

        assert len(result["items"]) == 3
        assert result["items"][0]["name"] == "file1.txt"
        assert result["items"][0]["is_dir"] is False
        assert result["items"][2]["name"] == "subdir"
        assert result["items"][2]["is_dir"] is True

    def test_list_nonexistent_directory(self, tmp_path):
        """Test listing non-existent directory returns error."""
        result = list_dir("nonexistent", tmp_path)

        assert "error" in result
        assert "does not exist" in result["error"]

    def test_list_file_instead_of_directory(self, tmp_path):
        """Test listing a file path returns error."""
        (tmp_path / "file.txt").write_text("content")

        result = list_dir("file.txt", tmp_path)

        assert "error" in result
        assert "Not a directory" in result["error"]


class TestReadFile:
    """Test file reading function."""

    def test_read_existing_file(self, tmp_path):
        """Test reading an existing file."""
        test_file = tmp_path / "test.txt"
        test_content = "Hello, World!\nLine 2"
        test_file.write_text(test_content)

        result = read_file("test.txt", tmp_path)

        assert "content" in result
        assert result["content"] == test_content
        assert "path" in result

    def test_read_nonexistent_file(self, tmp_path):
        """Test reading non-existent file returns error."""
        result = read_file("nonexistent.txt", tmp_path)

        assert "error" in result
        assert "not a file" in result["error"]

    def test_read_directory_instead_of_file(self, tmp_path):
        """Test reading a directory path returns error."""
        (tmp_path / "subdir").mkdir()

        result = read_file("subdir", tmp_path)

        assert "error" in result
        assert "not a file" in result["error"]

    def test_read_file_too_large(self, tmp_path):
        """Test reading file larger than max_size returns error."""
        large_file = tmp_path / "large.txt"
        large_file.write_text("x" * (600 * 1024))  # 600KB

        result = read_file("large.txt", tmp_path, max_size=512 * 1024)

        assert "error" in result
        assert "too large" in result["error"]

    def test_read_empty_file(self, tmp_path):
        """Test reading empty file."""
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")

        result = read_file("empty.txt", tmp_path)

        assert "content" in result
        assert result["content"] == ""


class TestWriteFile:
    """Test file writing function."""

    def test_write_new_file(self, tmp_path):
        """Test writing a new file."""
        result = write_file("new.txt", "Test content", tmp_path)

        assert "written" in result
        assert "bytes" in result
        assert result["bytes"] > 0

        # Verify file was created
        written_file = tmp_path / "new.txt"
        assert written_file.exists()
        assert written_file.read_text() == "Test content"

    def test_write_file_in_subdirectory(self, tmp_path):
        """Test writing file in non-existent subdirectory (should create it)."""
        result = write_file("subdir/nested/file.txt", "Nested content", tmp_path)

        assert "written" in result

        # Verify subdirectories and file were created
        written_file = tmp_path / "subdir" / "nested" / "file.txt"
        assert written_file.exists()
        assert written_file.read_text() == "Nested content"

    def test_overwrite_existing_file(self, tmp_path):
        """Test overwriting an existing file."""
        existing_file = tmp_path / "existing.txt"
        existing_file.write_text("Old content")

        result = write_file("existing.txt", "New content", tmp_path)

        assert "written" in result
        assert existing_file.read_text() == "New content"

    def test_write_empty_file(self, tmp_path):
        """Test writing empty content."""
        result = write_file("empty.txt", "", tmp_path)

        assert "written" in result
        assert result["bytes"] == 0

        written_file = tmp_path / "empty.txt"
        assert written_file.exists()
        assert written_file.read_text() == ""

    def test_write_unicode_content(self, tmp_path):
        """Test writing Unicode content."""
        unicode_content = "Hello 世界! 🌍"

        result = write_file("unicode.txt", unicode_content, tmp_path)

        assert "written" in result

        written_file = tmp_path / "unicode.txt"
        assert written_file.read_text() == unicode_content


class TestSaveWav:
    """Test WAV file saving function."""

    def test_save_wav_creates_file(self, tmp_path):
        """Test saving PCM16 data as WAV file."""
        # Create simple audio data (1 second of silence at 16kHz)
        import numpy as np
        silence = np.zeros(16000, dtype=np.int16)
        pcm16 = silence.tobytes()

        output_path = tmp_path / "test.wav"
        result = save_wav(pcm16, 16000, output_path)

        assert result == output_path
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_save_wav_with_audio_data(self, tmp_path):
        """Test saving actual audio waveform."""
        import numpy as np

        # Generate 1 second 440Hz tone
        sample_rate = 16000
        frequency = 440
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        waveform = np.sin(2 * np.pi * frequency * t) * 32767
        pcm16 = waveform.astype(np.int16).tobytes()

        output_path = tmp_path / "tone.wav"
        result = save_wav(pcm16, sample_rate, output_path)

        # Verify WAV file structure
        import wave
        with wave.open(str(result), 'rb') as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == sample_rate
            assert wf.getnframes() == len(waveform)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
