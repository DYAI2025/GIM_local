#!/usr/bin/env python3
"""
Pydantic models for type-safe tool handling and configuration validation.
"""
from typing import Dict, Any, Literal
from pydantic import BaseModel, Field, validator, ConfigDict


class ToolRequest(BaseModel):
    """Validates tool requests from LLM."""
    model_config = ConfigDict(strict=True)

    tool: Literal["list_dir", "read_file", "write_file"] = Field(
        ...,
        description="Name of the tool to execute"
    )
    args: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments for the tool"
    )

    @validator('args')
    def validate_args(cls, v, values):
        """Validate arguments based on tool type."""
        tool = values.get('tool')

        if tool == 'list_dir':
            # path is optional, defaults to "."
            if 'path' in v and not isinstance(v['path'], str):
                raise ValueError('path must be a string')

        elif tool == 'read_file':
            # path is required
            if 'path' not in v:
                raise ValueError('read_file requires "path" argument')
            if not isinstance(v['path'], str):
                raise ValueError('path must be a string')

        elif tool == 'write_file':
            # path and content are required
            if 'path' not in v or 'content' not in v:
                raise ValueError('write_file requires "path" and "content" arguments')
            if not isinstance(v['path'], str) or not isinstance(v['content'], str):
                raise ValueError('path and content must be strings')

        return v


class ToolResult(BaseModel):
    """Standardized tool result format."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    success: bool = Field(..., description="Whether the operation succeeded")
    data: Dict[str, Any] = Field(default_factory=dict, description="Result data")
    error: str | None = Field(None, description="Error message if failed")


class ConfigValidator(BaseModel):
    """Validates environment configuration at startup."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    ollama_url: str = Field(default="http://127.0.0.1:11434")
    ollama_model: str = Field(default="llama3.1:8b")
    ollama_autostart: bool = Field(default=True)
    ollama_auto_pull: bool = Field(default=False)
    ollama_start_timeout: float = Field(default=30.0, gt=0)
    ollama_retry_max: int = Field(default=10, gt=0)

    asr_model: str = Field(default="small-int8")
    sample_rate: int = Field(default=16000, gt=0)
    vad_aggr: int = Field(default=2, ge=0, le=3)

    workspace_dir: str = Field(default="~/ai_workspace")
    piper_path: str = Field(default="")
    piper_voice: str = Field(default="")

    @validator('ollama_url')
    def validate_url(cls, v):
        """Ensure URL is well-formed."""
        if not v.startswith(('http://', 'https://')):
            raise ValueError('OLLAMA_URL must start with http:// or https://')
        return v.rstrip('/')

    @validator('vad_aggr')
    def validate_vad_aggr(cls, v):
        """VAD aggressiveness must be 0-3."""
        if not 0 <= v <= 3:
            raise ValueError('VAD_AGGR must be between 0 and 3')
        return v
