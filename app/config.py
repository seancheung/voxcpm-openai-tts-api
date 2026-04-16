from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", case_sensitive=False, extra="ignore")

    voxcpm_model: str = Field(default="openbmb/VoxCPM2")
    voxcpm_device: Literal["auto", "cuda", "mps", "cpu"] = Field(default="auto")
    voxcpm_cuda_index: int = Field(default=0)
    voxcpm_cache_dir: Optional[str] = Field(default=None)
    voxcpm_optimize: bool = Field(default=True)
    voxcpm_load_denoiser: bool = Field(default=False)
    voxcpm_zipenhancer_model: str = Field(
        default="iic/speech_zipenhancer_ans_multiloss_16k_base"
    )

    voxcpm_cfg_value: float = Field(default=2.0, ge=0.1, le=10.0)
    voxcpm_inference_timesteps: int = Field(default=10, ge=1, le=100)
    voxcpm_denoise: bool = Field(default=False)
    voxcpm_normalize: bool = Field(default=False)
    voxcpm_retry_badcase: bool = Field(default=True)
    voxcpm_retry_badcase_max_times: int = Field(default=3, ge=0, le=10)

    voxcpm_voices_dir: str = Field(default="/voices")

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    log_level: str = Field(default="info")
    max_input_chars: int = Field(default=8000)
    default_response_format: Literal[
        "mp3", "opus", "aac", "flac", "wav", "pcm"
    ] = Field(default="mp3")

    @property
    def voices_path(self) -> Path:
        return Path(self.voxcpm_voices_dir)

    @property
    def resolved_device(self) -> str:
        import torch

        if self.voxcpm_device == "auto":
            if torch.cuda.is_available():
                return f"cuda:{self.voxcpm_cuda_index}"
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and mps.is_available():
                return "mps"
            return "cpu"
        if self.voxcpm_device == "cuda":
            return f"cuda:{self.voxcpm_cuda_index}"
        return self.voxcpm_device


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
