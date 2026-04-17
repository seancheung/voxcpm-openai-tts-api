# VoxCPM OpenAI-TTS API

**English** · [中文](./README.zh.md)

An [OpenAI TTS](https://platform.openai.com/docs/api-reference/audio/createSpeech)-compatible HTTP service wrapping [VoxCPM](https://github.com/OpenBMB/VoxCPM) — a tokenizer-free, context-aware multilingual TTS model (30 languages incl. 9 Chinese dialects, 48 kHz output) — with zero-shot voice cloning driven by files dropped into a mounted directory.

## Features

- **OpenAI TTS compatible** — `POST /v1/audio/speech` with the same request shape as the OpenAI SDK
- **Voice cloning (ultimate)** — each voice is a `xxx.wav` + `xxx.txt` pair in a mounted directory; the model continues the reference to faithfully reproduce timbre, rhythm, and emotion
- **Voice cloning (reference-isolated)** — extra `POST /v1/audio/reference` endpoint for VoxCPM2's controllable cloning mode (uses only the reference audio, ignores the transcript)
- **Voice design** — extra `POST /v1/audio/design` endpoint that creates a voice from a natural-language description (e.g. `"young woman, warm, soft"`), no reference audio needed
- **2 images** — `cuda` and `cpu`
- **Model weights downloaded at runtime** — nothing heavy baked into the image; HuggingFace cache is mounted for reuse
- **Multiple output formats** — `mp3`, `opus`, `aac`, `flac`, `wav`, `pcm`

## Available images

| Image | Device |
|---|---|
| `ghcr.io/seancheung/voxcpm-openai-tts-api:cuda-latest` | CUDA 12.4 |
| `ghcr.io/seancheung/voxcpm-openai-tts-api:latest`      | CPU |

Images are built for `linux/amd64`.

## Quick start

### 1. Prepare the voices directory

```
voices/
├── alice.wav     # reference audio, mono, 16kHz+, ~3-20s recommended
├── alice.txt     # UTF-8 text: the exact transcript of alice.wav
├── bob.wav
└── bob.txt
```

**Rules**: a voice is valid only when both files with the same stem exist; the stem is the voice id; unpaired or extra files are ignored. Voices are used by both `/v1/audio/speech` (ultimate cloning, uses wav + txt) and `/v1/audio/reference` (reference-isolated cloning, uses wav only). The `/v1/audio/design` endpoint does not need the `voices/` directory.

### 2. Run the container

GPU (recommended):

```bash
docker run --rm -p 8000:8000 --gpus all \
  -v $PWD/voices:/voices:ro \
  -v $PWD/hf_cache:/root/.cache/huggingface \
  ghcr.io/seancheung/voxcpm-openai-tts-api:cuda-latest
```

CPU:

```bash
docker run --rm -p 8000:8000 \
  -v $PWD/voices:/voices:ro \
  -v $PWD/hf_cache:/root/.cache/huggingface \
  ghcr.io/seancheung/voxcpm-openai-tts-api:latest
```

Model weights (≈4 GB for VoxCPM2) are pulled from HuggingFace on first start. Mounting `/root/.cache/huggingface` persists them across container restarts.

> **GPU prerequisites**: NVIDIA driver + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) on Linux. On Windows use Docker Desktop + WSL2 + NVIDIA Windows driver; no host CUDA toolkit required. VoxCPM2 needs ≈8 GB VRAM.

### 3. docker-compose

See [`docker/docker-compose.example.yml`](./docker/docker-compose.example.yml).

## API usage

The service listens on port `8000` by default.

### GET `/v1/audio/voices`

List all usable voices.

```bash
curl -s http://localhost:8000/v1/audio/voices | jq
```

Response:

```json
{
  "object": "list",
  "data": [
    {
      "id": "alice",
      "preview_url": "http://localhost:8000/v1/audio/voices/preview?id=alice",
      "prompt_text": "Hello, this is a reference audio sample."
    }
  ]
}
```

### GET `/v1/audio/voices/preview?id={id}`

Returns the raw reference wav (`audio/wav`), suitable for a browser `<audio>` element.

### POST `/v1/audio/speech`

OpenAI TTS-compatible endpoint — **ultimate cloning** mode. The voice's `wav` and `txt` are both passed to VoxCPM so the model continues the reference speaker, reproducing timbre, rhythm, and emotional nuance.

```bash
curl -s http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "voxcpm",
    "input": "Hello world, this is a test.",
    "voice": "alice",
    "response_format": "mp3"
  }' \
  -o out.mp3
```

Request fields:

| Field | Type | Description |
|---|---|---|
| `model` | string | Accepted but ignored (for OpenAI SDK compatibility) |
| `input` | string | Text to synthesize, up to 8000 characters |
| `voice` | string | Voice id — must match an entry from `/v1/audio/voices` |
| `response_format` | string | `mp3` (default) / `opus` / `aac` / `flac` / `wav` / `pcm` |
| `speed` | float | Accepted for OpenAI SDK compatibility but **ignored** — VoxCPM has no speed control |
| `cfg_value` | float | Optional classifier-free guidance scale (`0.1 - 10.0`, default `2.0`) |
| `inference_timesteps` | int | Optional LocDiT sampling steps (`1 - 100`, default `10`); higher is slower but slightly better |
| `denoise` | bool | Optional, denoise the reference audio before use (requires `VOXCPM_LOAD_DENOISER=true`) |
| `normalize` | bool | Optional text normalization before synthesis |

Output audio is mono 48 kHz for VoxCPM2 (44.1 kHz for `openbmb/VoxCPM-0.5B`); `pcm` is raw s16le.

### POST `/v1/audio/reference`

VoxCPM2-only, **reference-isolated cloning** — the voice's wav is used purely as a timbre reference (the `.txt` is ignored). Compared with `/v1/audio/speech`, this mode decouples the reference's prosody/content from the generation and is better suited to style-controlled cloning.

```bash
curl -s http://localhost:8000/v1/audio/reference \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "Speaking in a calm, professional tone.",
    "voice": "alice",
    "response_format": "mp3"
  }' \
  -o out_ref.mp3
```

Request fields: same as `/v1/audio/speech` minus `model` and `speed`.

### POST `/v1/audio/design`

Non-standard endpoint that exposes VoxCPM's **voice design** mode — no reference audio needed; describe the target voice with an `instruct` string.

```bash
curl -s http://localhost:8000/v1/audio/design \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "British narrator here.",
    "instruct": "male, low pitch, british accent",
    "response_format": "mp3"
  }' \
  -o out_design.mp3
```

Request fields:

| Field | Type | Description |
|---|---|---|
| `input` | string | Text to synthesize |
| `instruct` | string | Voice attributes, e.g. `"young woman, warm, soft"`. Injected as a `(description)` prefix in front of the text before VoxCPM generates. |
| `response_format` | string | Same as `/speech` |
| `cfg_value` / `inference_timesteps` / `normalize` | — | Same semantics as `/speech` |

### Using the OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-noop")

with client.audio.speech.with_streaming_response.create(
    model="voxcpm",
    voice="alice",
    input="Hello world",
    response_format="mp3",
) as resp:
    resp.stream_to_file("out.mp3")
```

Extensions (`cfg_value`, `inference_timesteps`, `denoise`, `normalize`) can be passed through `extra_body={...}`.

### GET `/healthz`

Returns model name, device, sample rate and status for health checks.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `VOXCPM_MODEL` | `openbmb/VoxCPM2` | HuggingFace repo id or local path. Set to `openbmb/VoxCPM-0.5B` to use the older, smaller model. |
| `VOXCPM_DEVICE` | `auto` | `auto` → CUDA > MPS > CPU. Or `cuda` / `mps` / `cpu` |
| `VOXCPM_CUDA_INDEX` | `0` | Selects `cuda:N` when device is `cuda` or `auto` |
| `VOXCPM_CACHE_DIR` | — | Sets `HF_HOME` and VoxCPM's snapshot cache directory before model load |
| `VOXCPM_OPTIMIZE` | `true` | Enable `torch.compile` and warm-up at startup |
| `VOXCPM_LOAD_DENOISER` | `false` | Load ModelScope ZipEnhancer for prompt-audio denoising; adds ~250 MB download |
| `VOXCPM_ZIPENHANCER_MODEL` | `iic/speech_zipenhancer_ans_multiloss_16k_base` | ModelScope id or local path for the denoiser |
| `VOXCPM_CFG_VALUE` | `2.0` | Default classifier-free guidance scale (`0.1 - 10.0`) |
| `VOXCPM_INFERENCE_TIMESTEPS` | `10` | Default LocDiT sampling steps (`1 - 100`) |
| `VOXCPM_DENOISE` | `false` | Default value of the `denoise` request field |
| `VOXCPM_NORMALIZE` | `false` | Default value of the `normalize` request field |
| `VOXCPM_RETRY_BADCASE` | `true` | Retry synthesis if the audio/text ratio looks wrong |
| `VOXCPM_RETRY_BADCASE_MAX_TIMES` | `3` | Max retry attempts |
| `VOXCPM_VOICES_DIR` | `/voices` | Voices directory |
| `MAX_INPUT_CHARS` | `8000` | Upper bound for the `input` field |
| `DEFAULT_RESPONSE_FORMAT` | `mp3` | |
| `HOST` | `0.0.0.0` | |
| `PORT` | `8000` | |
| `LOG_LEVEL` | `info` | |

## Building images locally

Initialize the submodule first (the workflow does this automatically).

```bash
git submodule update --init --recursive

# CUDA image
docker buildx build -f docker/Dockerfile.cuda \
  -t voxcpm-openai-tts-api:cuda .

# CPU image
docker buildx build -f docker/Dockerfile.cpu \
  -t voxcpm-openai-tts-api:cpu .
```

## Caveats

- **`speed` is a no-op.** VoxCPM has no native speed control, but the field is kept in the schema so that OpenAI's Python SDK default request body (which always sends `speed=1.0`) does not 422. If you need tempo control, post-process the returned audio.
- **No built-in OpenAI voice names** (`alloy`, `echo`, `fable`, …). VoxCPM is zero-shot; to get a stable voice under those names, just drop `alloy.wav` + `alloy.txt` into `voices/`.
- **Concurrency**: a single VoxCPM instance is not thread-safe; the service serializes inference with an asyncio Lock. Scale out by running more containers behind a load balancer.
- **Long text**: requests whose `input` exceeds `MAX_INPUT_CHARS` (default 8000) return 413. VoxCPM itself handles long text in chunks internally.
- **Streaming is not supported** on the HTTP layer — the endpoint returns the complete audio when generation finishes. (VoxCPM does support streaming internally; exposing it here is future work.)
- **No reference-audio preprocessing cache** — each request re-reads the voice's wav. If you hit the same voice at high RPS, add a reverse-proxy cache or pre-generate clips.
- **Denoiser is off by default** to keep the image small and startup fast. Set `VOXCPM_LOAD_DENOISER=true` to enable it, which also lets per-request `denoise: true` do real work.
- **Voice design syntax**: VoxCPM triggers voice design by a `(description)` prefix on the text. This service does the wrapping for you when you hit `/v1/audio/design`; if you pass `(...)` directly to `/v1/audio/speech`, behaviour is undefined.
- **No built-in auth** — deploy behind a reverse proxy (Nginx, Cloudflare, etc.) if you need token-based access control.
- **VoxCPM2 vs VoxCPM-0.5B**: the default is VoxCPM2 (2B params, 48 kHz, 30 languages). `/v1/audio/reference` only works with VoxCPM2. Override with `VOXCPM_MODEL=openbmb/VoxCPM-0.5B` for the smaller model (6 GB VRAM, 44.1 kHz, Chinese+English).

## Project layout

```
.
├── VoxCPM/                     # read-only submodule, never modified
├── app/                        # FastAPI application
│   ├── server.py
│   ├── engine.py               # model loading + inference
│   ├── voices.py               # voices directory scanner
│   ├── audio.py                # multi-format encoder
│   ├── config.py
│   └── schemas.py
├── docker/
│   ├── Dockerfile.cuda
│   ├── Dockerfile.cpu
│   ├── requirements.api.txt
│   ├── entrypoint.sh
│   └── docker-compose.example.yml
├── .github/workflows/
│   └── build-images.yml        # cuda + cpu matrix build
└── README.md
```

## Acknowledgements

Built on top of [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM) (Apache 2.0).
