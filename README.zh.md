# VoxCPM OpenAI-TTS API

[English](./README.md) · **中文**

一个 [OpenAI TTS](https://platform.openai.com/docs/api-reference/audio/createSpeech) 兼容的 HTTP 服务，对 [VoxCPM](https://github.com/OpenBMB/VoxCPM)（无 tokenizer、上下文感知的多语言 TTS 模型，支持 30 种语言和 9 种中文方言，输出 48 kHz 高保真音频）进行封装，支持从挂载目录零样本克隆音色。

## 特性

- **OpenAI TTS 兼容**：`POST /v1/audio/speech`，请求体格式与 OpenAI SDK 一致
- **音色克隆（续写模式）**：挂载 `voices/` 目录下的 `xxx.wav` + `xxx.txt` 对，模型会续写参考音频以最大程度还原音色、韵律与情绪
- **音色克隆（参考隔离）**：额外提供 `POST /v1/audio/reference`，使用 VoxCPM2 的 controllable cloning 模式——只用参考音频做音色引导，忽略文本
- **音色设计**：额外提供 `POST /v1/audio/design`，通过自然语言描述（如 `"young woman, warm, soft"`）创造全新音色，无需参考音频
- **2 个镜像**：`cuda` 与 `cpu`
- **模型运行时下载**：不打包进镜像，HuggingFace 缓存目录挂载后可复用
- **多种输出格式**：`mp3`、`opus`、`aac`、`flac`、`wav`、`pcm`

## 可用镜像

| 镜像 | 设备 |
|---|---|
| `ghcr.io/seancheung/voxcpm-openai-tts-api:cuda-latest` | CUDA 12.4 |
| `ghcr.io/seancheung/voxcpm-openai-tts-api:latest`      | CPU |

镜像仅构建 `linux/amd64`。

## 快速开始

### 1. 准备音色目录

```
voices/
├── alice.wav     # 参考音频，单声道，16kHz 以上，推荐 3-20 秒
├── alice.txt     # UTF-8 纯文本，内容为 alice.wav 中说出的原文
├── bob.wav
└── bob.txt
```

**规则**：必须同时存在同名的 `.wav` 和 `.txt` 才会被识别为有效音色；文件名（不含后缀）即音色 id；多余或缺对的文件会被忽略。`/v1/audio/speech`（续写克隆，使用 wav + txt）和 `/v1/audio/reference`（参考隔离克隆，只使用 wav）共享同一音色目录；`/v1/audio/design` 端点不需要 `voices/`。

### 2. 运行容器

GPU 版本（推荐）：

```bash
docker run --rm -p 8000:8000 --gpus all \
  -v $PWD/voices:/voices:ro \
  -v $PWD/cache:/root/.cache \
  ghcr.io/seancheung/voxcpm-openai-tts-api:cuda-latest
```

CPU 版本：

```bash
docker run --rm -p 8000:8000 \
  -v $PWD/voices:/voices:ro \
  -v $PWD/cache:/root/.cache \
  ghcr.io/seancheung/voxcpm-openai-tts-api:latest
```

首次启动会从 HuggingFace 下载模型权重（VoxCPM2 约 4 GB）。挂载 `/root/.cache` 可让权重在容器重启后复用。

> **GPU 要求**：宿主机需安装 NVIDIA 驱动与 [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)。Windows 需 Docker Desktop + WSL2 + NVIDIA Windows 驱动。VoxCPM2 约需 8 GB 显存。

### 3. docker-compose

参考 [`docker/docker-compose.example.yml`](./docker/docker-compose.example.yml)。

## API 用法

服务默认监听 `8000` 端口。

### GET `/v1/audio/voices`

列出所有可用音色。

```bash
curl -s http://localhost:8000/v1/audio/voices | jq
```

返回：

```json
{
  "object": "list",
  "data": [
    {
      "id": "alice",
      "preview_url": "http://localhost:8000/v1/audio/voices/preview?id=alice",
      "prompt_text": "你好，这是一段参考音频。"
    }
  ]
}
```

### GET `/v1/audio/voices/preview?id={id}`

返回参考音频本体（`audio/wav`），可用于浏览器 `<audio>` 试听。

### POST `/v1/audio/speech`

OpenAI TTS 兼容接口——**续写克隆**模式。音色的 `.wav` 与 `.txt` 会一并作为 prompt 传给 VoxCPM，由模型续写参考说话人，最大程度还原音色、韵律与情绪。

```bash
curl -s http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "voxcpm",
    "input": "你好世界，这是一段测试语音。",
    "voice": "alice",
    "response_format": "mp3"
  }' \
  -o out.mp3
```

请求字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `model` | string | 接受但忽略（为了与 OpenAI SDK 兼容） |
| `input` | string | 要合成的文本，最长 8000 字符 |
| `voice` | string | 音色 id，必须匹配 `/v1/audio/voices` 中的某一项 |
| `response_format` | string | `mp3`（默认） / `opus` / `aac` / `flac` / `wav` / `pcm` |
| `speed` | float | 为 OpenAI SDK 兼容而保留，**实际忽略**——VoxCPM 无语速控制 |
| `cfg_value` | float | 可选 classifier-free guidance（`0.1 - 10.0`，默认 `2.0`） |
| `inference_timesteps` | int | 可选 LocDiT 采样步数（`1 - 100`，默认 `10`），越高越慢但效果略好 |
| `denoise` | bool | 可选，对参考音频降噪（需开启 `VOXCPM_LOAD_DENOISER=true`） |
| `normalize` | bool | 可选，合成前做文本规范化 |

输出音频为单声道 48 kHz（VoxCPM2）或 44.1 kHz（`openbmb/VoxCPM-0.5B`）；`pcm` 为裸 s16le 数据。

### POST `/v1/audio/reference`

VoxCPM2 专属——**参考隔离**克隆。音色的 wav 仅作为音色参考使用，`.txt` 被忽略。相比 `/v1/audio/speech`，该模式将参考的韵律/内容从生成中解耦，更适合风格受控的克隆。

```bash
curl -s http://localhost:8000/v1/audio/reference \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "以冷静、专业的语气朗读这段文本。",
    "voice": "alice",
    "response_format": "mp3"
  }' \
  -o out_ref.mp3
```

请求字段：同 `/v1/audio/speech`，去掉 `model` 和 `speed`。

### POST `/v1/audio/design`

非标准端点，暴露 VoxCPM 的 **voice design**（语音设计）模式——无需参考音频，通过 `instruct` 字符串描述目标音色。

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

请求字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `input` | string | 要合成的文本 |
| `instruct` | string | 音色属性，如 `"young woman, warm, soft"`；合成前会以 `(描述)` 前缀方式注入到文本前 |
| `response_format` | string | 同 `/speech` |
| `cfg_value` / `inference_timesteps` / `normalize` | — | 语义同 `/speech` |

### 使用 OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-noop")

with client.audio.speech.with_streaming_response.create(
    model="voxcpm",
    voice="alice",
    input="你好世界",
    response_format="mp3",
) as resp:
    resp.stream_to_file("out.mp3")
```

`cfg_value`、`inference_timesteps`、`denoise`、`normalize` 等扩展字段可通过 `extra_body={...}` 传入。

### GET `/healthz`

返回模型名、设备、采样率与状态，用于健康检查。

## 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `VOXCPM_MODEL` | `openbmb/VoxCPM2` | HuggingFace 仓库 id 或本地路径。设为 `openbmb/VoxCPM-0.5B` 可切换到旧的小模型。 |
| `VOXCPM_DEVICE` | `auto` | `auto` 按 CUDA > MPS > CPU 优先级。也可强制 `cuda` / `mps` / `cpu` |
| `VOXCPM_CUDA_INDEX` | `0` | `cuda` / `auto` 时选择的 `cuda:N` |
| `VOXCPM_CACHE_DIR` | — | 加载模型前写入 `HF_HOME`，同时作为 VoxCPM 快照下载目录 |
| `VOXCPM_OPTIMIZE` | `true` | 启动时 `torch.compile` + 预热 |
| `VOXCPM_LOAD_DENOISER` | `false` | 加载 ModelScope ZipEnhancer 做参考音频降噪，需额外下载约 250 MB |
| `VOXCPM_ZIPENHANCER_MODEL` | `iic/speech_zipenhancer_ans_multiloss_16k_base` | ModelScope id 或本地路径 |
| `VOXCPM_CFG_VALUE` | `2.0` | 默认 classifier-free guidance 强度（`0.1 - 10.0`） |
| `VOXCPM_INFERENCE_TIMESTEPS` | `10` | 默认 LocDiT 采样步数（`1 - 100`） |
| `VOXCPM_DENOISE` | `false` | 请求 `denoise` 字段的默认值 |
| `VOXCPM_NORMALIZE` | `false` | 请求 `normalize` 字段的默认值 |
| `VOXCPM_RETRY_BADCASE` | `true` | 音频/文本比例异常时是否重试 |
| `VOXCPM_RETRY_BADCASE_MAX_TIMES` | `3` | 最大重试次数 |
| `VOXCPM_VOICES_DIR` | `/voices` | 音色目录 |
| `MAX_INPUT_CHARS` | `8000` | `input` 字段上限 |
| `DEFAULT_RESPONSE_FORMAT` | `mp3` | |
| `HOST` | `0.0.0.0` | |
| `PORT` | `8000` | |
| `LOG_LEVEL` | `info` | |

## 本地构建镜像

构建前需先初始化 submodule（workflow 已处理）。

```bash
git submodule update --init --recursive

# CUDA 镜像
docker buildx build -f docker/Dockerfile.cuda \
  -t voxcpm-openai-tts-api:cuda .

# CPU 镜像
docker buildx build -f docker/Dockerfile.cpu \
  -t voxcpm-openai-tts-api:cpu .
```

## 局限 / 注意事项

- **`speed` 字段是 no-op**：VoxCPM 无原生语速控制；保留该字段只为让 OpenAI Python SDK 的默认请求体（总会带 `speed=1.0`）不被 422。若需调速请对返回音频做后处理。
- **不做 OpenAI 固定音色名映射**（`alloy`、`echo`、`fable` 等）。VoxCPM 本身是零样本，没有内置音色；若想通过这些名字调用稳定的声音，直接在 `voices/` 放同名 `.wav` + `.txt` 即可。
- **并发**：VoxCPM 单实例非线程安全，服务内部用 asyncio Lock 串行化。并发请求依赖横向扩容（多容器 + 负载均衡）。
- **长文本**：超过 `MAX_INPUT_CHARS`（默认 8000）返回 413。VoxCPM 内部自身会做分句处理。
- **不支持 HTTP 层流式返回**：生成完成后一次性返回。（VoxCPM 本身支持流式，服务层目前未暴露。）
- **不做参考音频缓存**：每次请求都会重新读取 wav 文件。如果同一音色 QPS 很高，请在反向代理层做缓存或预生成音频片段。
- **降噪器默认关闭**：为了镜像更小、启动更快。设 `VOXCPM_LOAD_DENOISER=true` 后才会真的启用每请求的 `denoise: true`。
- **voice design 语法**：VoxCPM 通过文本前加 `(描述)` 前缀触发 voice design。服务会在 `/v1/audio/design` 时自动包装；如果你直接把 `(...)` 写进 `/v1/audio/speech` 的 `input` 里，行为未定义。
- **无内置鉴权**：如需 token 访问控制，请在反向代理层（Nginx、Cloudflare 等）做。
- **VoxCPM2 vs VoxCPM-0.5B**：默认是 VoxCPM2（2B 参数、48 kHz、30 种语言）。`/v1/audio/reference` 只在 VoxCPM2 上可用。设 `VOXCPM_MODEL=openbmb/VoxCPM-0.5B` 可切到小模型（6 GB 显存、44.1 kHz、中英双语）。

## 目录结构

```
.
├── VoxCPM/                     # 只读 submodule，不修改
├── app/                        # FastAPI 应用
│   ├── server.py
│   ├── engine.py               # 模型加载 + 推理
│   ├── voices.py               # 音色扫描
│   ├── audio.py                # 多格式编码
│   ├── config.py
│   └── schemas.py
├── docker/
│   ├── Dockerfile.cuda
│   ├── Dockerfile.cpu
│   ├── requirements.api.txt
│   ├── entrypoint.sh
│   └── docker-compose.example.yml
├── .github/workflows/
│   └── build-images.yml        # cuda + cpu 矩阵构建
└── README.md
```

## 致谢

基于 [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM)（Apache 2.0）。
