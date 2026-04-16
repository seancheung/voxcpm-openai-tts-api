from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


ResponseFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]


class SpeechRequest(BaseModel):
    """OpenAI-compatible `/v1/audio/speech` request (ultimate-cloning mode)."""

    model: Optional[str] = Field(default=None, description="Accepted for OpenAI compatibility; ignored.")
    input: str = Field(..., description="Text to synthesize.")
    voice: str = Field(..., description="Voice id matching a file pair in the voices directory.")
    response_format: ResponseFormat = Field(default="mp3")
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="Accepted for OpenAI compatibility; ignored (VoxCPM has no speed control).")

    cfg_value: Optional[float] = Field(default=None, ge=0.1, le=10.0)
    inference_timesteps: Optional[int] = Field(default=None, ge=1, le=100)
    denoise: Optional[bool] = Field(default=None)
    normalize: Optional[bool] = Field(default=None)


class ReferenceRequest(BaseModel):
    """`/v1/audio/reference` request — VoxCPM2's controllable cloning (reference-isolated, ignores .txt)."""

    input: str = Field(..., description="Text to synthesize.")
    voice: str = Field(..., description="Voice id; only the .wav is used, the .txt is ignored.")
    response_format: ResponseFormat = Field(default="mp3")

    cfg_value: Optional[float] = Field(default=None, ge=0.1, le=10.0)
    inference_timesteps: Optional[int] = Field(default=None, ge=1, le=100)
    denoise: Optional[bool] = Field(default=None)
    normalize: Optional[bool] = Field(default=None)


class DesignRequest(BaseModel):
    """`/v1/audio/design` request — VoxCPM's voice-design mode (no reference audio)."""

    input: str = Field(..., description="Text to synthesize.")
    instruct: str = Field(..., description="Voice design attributes, e.g. 'young woman, warm, soft'.")
    response_format: ResponseFormat = Field(default="mp3")

    cfg_value: Optional[float] = Field(default=None, ge=0.1, le=10.0)
    inference_timesteps: Optional[int] = Field(default=None, ge=1, le=100)
    normalize: Optional[bool] = Field(default=None)


class VoiceInfo(BaseModel):
    id: str
    preview_url: str
    prompt_text: str


class VoiceList(BaseModel):
    object: Literal["list"] = "list"
    data: list[VoiceInfo]


class HealthResponse(BaseModel):
    status: Literal["ok", "loading", "error"]
    model: str
    device: Optional[str] = None
    sample_rate: Optional[int] = None
