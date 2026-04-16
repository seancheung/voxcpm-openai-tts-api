from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

import numpy as np
from voxcpm import VoxCPM

log = logging.getLogger(__name__)


class TTSEngine:
    def __init__(self, settings):
        self.settings = settings

        device = settings.resolved_device

        if settings.voxcpm_cache_dir:
            os.environ.setdefault("HF_HOME", settings.voxcpm_cache_dir)

        log.info(
            "loading VoxCPM model=%s device=%s optimize=%s denoiser=%s",
            settings.voxcpm_model,
            device,
            settings.voxcpm_optimize,
            settings.voxcpm_load_denoiser,
        )
        self.model = VoxCPM.from_pretrained(
            hf_model_id=settings.voxcpm_model,
            load_denoiser=settings.voxcpm_load_denoiser,
            zipenhancer_model_id=settings.voxcpm_zipenhancer_model,
            cache_dir=settings.voxcpm_cache_dir,
            optimize=settings.voxcpm_optimize,
            device=device,
        )
        self.device = device
        self.sample_rate = int(self.model.tts_model.sample_rate)
        self._lock = asyncio.Lock()

    def _gen_kwargs(
        self,
        *,
        cfg_value: Optional[float],
        inference_timesteps: Optional[int],
        denoise: Optional[bool],
        normalize: Optional[bool],
    ) -> dict:
        s = self.settings
        return dict(
            cfg_value=cfg_value if cfg_value is not None else s.voxcpm_cfg_value,
            inference_timesteps=(
                inference_timesteps
                if inference_timesteps is not None
                else s.voxcpm_inference_timesteps
            ),
            denoise=denoise if denoise is not None else s.voxcpm_denoise,
            normalize=normalize if normalize is not None else s.voxcpm_normalize,
            retry_badcase=s.voxcpm_retry_badcase,
            retry_badcase_max_times=s.voxcpm_retry_badcase_max_times,
        )

    @staticmethod
    def _to_float32(wav) -> np.ndarray:
        arr = np.asarray(wav)
        return np.ascontiguousarray(arr.astype(np.float32, copy=False))

    # ------------------------------------------------------------------
    # inference entrypoints
    # ------------------------------------------------------------------
    async def synthesize_clone(
        self,
        text: str,
        *,
        prompt_wav: str,
        prompt_text: str,
        cfg_value: Optional[float] = None,
        inference_timesteps: Optional[int] = None,
        denoise: Optional[bool] = None,
        normalize: Optional[bool] = None,
    ) -> np.ndarray:
        kwargs = self._gen_kwargs(
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            denoise=denoise,
            normalize=normalize,
        )
        async with self._lock:
            wav = await asyncio.to_thread(
                self.model.generate,
                text=text,
                prompt_wav_path=prompt_wav,
                prompt_text=prompt_text,
                **kwargs,
            )
        return self._to_float32(wav)

    async def synthesize_reference(
        self,
        text: str,
        *,
        reference_wav: str,
        cfg_value: Optional[float] = None,
        inference_timesteps: Optional[int] = None,
        denoise: Optional[bool] = None,
        normalize: Optional[bool] = None,
    ) -> np.ndarray:
        kwargs = self._gen_kwargs(
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            denoise=denoise,
            normalize=normalize,
        )
        async with self._lock:
            wav = await asyncio.to_thread(
                self.model.generate,
                text=text,
                reference_wav_path=reference_wav,
                **kwargs,
            )
        return self._to_float32(wav)

    async def synthesize_design(
        self,
        text: str,
        *,
        instruct: str,
        cfg_value: Optional[float] = None,
        inference_timesteps: Optional[int] = None,
        normalize: Optional[bool] = None,
    ) -> np.ndarray:
        kwargs = self._gen_kwargs(
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            denoise=None,
            normalize=normalize,
        )
        final_text = f"({instruct}){text}"
        async with self._lock:
            wav = await asyncio.to_thread(
                self.model.generate,
                text=final_text,
                **kwargs,
            )
        return self._to_float32(wav)
