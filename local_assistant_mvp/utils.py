#!/usr/bin/env python3
"""
Utility functions for logging, file operations, and TTS.
"""
import os
import logging
import pathlib
import wave
import subprocess
from typing import Dict, Any, List
import numpy as np
import sounddevice as sd

# Setup logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_workspace(workspace_dir: str) -> pathlib.Path:
    """Create and return workspace directory."""
    workspace = pathlib.Path(os.path.expanduser(workspace_dir)).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    logger.info(f"Workspace initialized at: {workspace}")
    return workspace


def safe_path(p: str, workspace: pathlib.Path) -> pathlib.Path:
    """
    Resolve and ensure path is within workspace sandbox.

    Args:
        p: Requested path (relative or absolute)
        workspace: Workspace root directory

    Returns:
        Resolved path within workspace

    Raises:
        PermissionError: If path escapes workspace
    """
    target = (workspace / p).resolve()
    if not str(target).startswith(str(workspace)):
        logger.warning(f"Attempted path escape: {p} -> {target}")
        raise PermissionError("Path outside sandbox.")
    return target


def list_dir(path: str, workspace: pathlib.Path) -> Dict[str, Any]:
    """
    List directory contents within workspace.

    Args:
        path: Directory path relative to workspace
        workspace: Workspace root

    Returns:
        Dict with cwd and items list
    """
    try:
        target = safe_path(path, workspace)

        if not target.exists():
            logger.warning(f"Directory does not exist: {target}")
            return {"error": "Directory does not exist", "cwd": str(target)}

        if not target.is_dir():
            logger.warning(f"Path is not a directory: {target}")
            return {"error": "Not a directory", "cwd": str(target)}

        items = []
        for entry in sorted(target.iterdir()):
            items.append({
                "name": entry.name,
                "is_dir": entry.is_dir(),
                "size": entry.stat().st_size if entry.is_file() else None
            })

        logger.debug(f"Listed directory {target} with {len(items)} items")
        return {"cwd": str(target), "items": items}

    except PermissionError as e:
        logger.error(f"Permission denied for path: {path}", exc_info=True)
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Error listing directory {path}: {e}", exc_info=True)
        return {"error": f"Unexpected error: {e}"}


def read_file(path: str, workspace: pathlib.Path, max_size: int = 512 * 1024) -> Dict[str, Any]:
    """
    Read file contents within workspace.

    Args:
        path: File path relative to workspace
        workspace: Workspace root
        max_size: Maximum file size in bytes (default 512KB)

    Returns:
        Dict with path and content, or error
    """
    try:
        target = safe_path(path, workspace)

        if not target.is_file():
            logger.warning(f"Not a file: {target}")
            return {"error": "not a file"}

        if target.stat().st_size > max_size:
            logger.warning(f"File too large: {target} ({target.stat().st_size} bytes)")
            return {"error": f"file too large (>{max_size // 1024}KB)"}

        content = target.read_text(encoding="utf-8", errors="replace")
        logger.debug(f"Read file {target} ({len(content)} chars)")
        return {"path": str(target), "content": content}

    except PermissionError as e:
        logger.error(f"Permission denied for file: {path}", exc_info=True)
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Error reading file {path}: {e}", exc_info=True)
        return {"error": f"Unexpected error: {e}"}


def write_file(path: str, content: str, workspace: pathlib.Path) -> Dict[str, Any]:
    """
    Write content to file within workspace.

    Args:
        path: File path relative to workspace
        content: Content to write
        workspace: Workspace root

    Returns:
        Dict with written path and byte count
    """
    try:
        target = safe_path(path, workspace)
        target.parent.mkdir(parents=True, exist_ok=True)

        target.write_text(content, encoding="utf-8")
        byte_count = len(content.encode('utf-8'))

        logger.info(f"Wrote file {target} ({byte_count} bytes)")
        return {"written": str(target), "bytes": byte_count}

    except PermissionError as e:
        logger.error(f"Permission denied for file: {path}", exc_info=True)
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Error writing file {path}: {e}", exc_info=True)
        return {"error": f"Unexpected error: {e}"}


def speak(text: str, piper_path: str, piper_voice: str, workspace: pathlib.Path) -> None:
    """
    Convert text to speech using Piper TTS.

    Args:
        text: Text to speak
        piper_path: Path to piper binary
        piper_voice: Path to voice model
        workspace: Workspace for temp files
    """
    if not (piper_path and piper_voice):
        logger.debug("TTS not configured, skipping speech output")
        return

    piper_path_expanded = os.path.expanduser(piper_path)
    if not os.path.exists(piper_path_expanded):
        logger.warning(f"Piper binary not found at: {piper_path_expanded}")
        return

    tmp_wav = workspace / "_tts_out.wav"

    try:
        proc = subprocess.run(
            [piper_path_expanded, "-m", os.path.expanduser(piper_voice), "-f", str(tmp_wav)],
            input=text.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30
        )

        if proc.returncode != 0:
            logger.warning(f"Piper TTS failed with code {proc.returncode}: {proc.stderr.decode()}")
            return

        # Play wav
        with wave.open(str(tmp_wav), 'rb') as wf:
            sr = wf.getframerate()
            ch = wf.getnchannels()
            frames = wf.readframes(wf.getnframes())

        data = np.frombuffer(frames, dtype=np.int16)
        if ch > 1:
            data = data.reshape(-1, ch).mean(axis=1).astype(np.int16)

        sd.play(data, sr)
        sd.wait()

        logger.debug(f"Spoke {len(text)} characters")

    except subprocess.TimeoutExpired:
        logger.error("Piper TTS timed out after 30 seconds")
    except Exception as e:
        logger.error(f"Error during TTS: {e}", exc_info=True)
    finally:
        if tmp_wav.exists():
            try:
                tmp_wav.unlink()
            except:
                pass


def save_wav(pcm16: bytes, samplerate: int, path: pathlib.Path) -> pathlib.Path:
    """
    Save PCM16 audio data as WAV file.

    Args:
        pcm16: Raw PCM16 audio bytes
        samplerate: Sample rate in Hz
        path: Output path

    Returns:
        Path to saved file
    """
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(pcm16)

    logger.debug(f"Saved WAV file: {path} ({len(pcm16)} bytes)")
    return path
