# Depression Agent

可运行的抑郁症随访助手骨架，包含 FastAPI API、Gradio UI、以及若干服务适配层。当前所有 ASR/TTS/LLM 均为 Stub 实现，方便后续替换为真实服务。

## 目录结构

```
apps/
  api/
    main.py            # FastAPI 应用，提供 /health /metrics 及 DM 路由
    router_dm.py       # 抑郁随访问答与报告构建接口
  ui-gradio/
    app.py             # 简单的 Gradio 多模态 UI
packages/
  common/config.py     # 环境变量配置
services/
  orchestrator/        # LangGraph 最小管线
  audio/               # ASR Stub
  tts/                 # TTS Stub
  llm/                 # LLM JSON 客户端 (Stub + 可切换 OpenAI 兼容接口)
  risk/                # 风险评估引擎
  report/              # PDF 报告生成
  store/               # Redis 仓储封装
scripts/
  run_api.sh           # 启动 FastAPI
  run_ui.sh            # 启动 Gradio UI
```

## 快速开始

```bash
pip install -r requirements.txt  # 如已存在
cp .env.example .env
./scripts/run_api.sh
./scripts/run_ui.sh
```

默认 API 监听 `0.0.0.0:8080`，UI 监听 `0.0.0.0:7860`。

### API 验收

1. `GET /health` 返回 `{ "ok": true }`
2. `POST /dm/step` 仅支持文本输入，返回下一轮问句与进度信息
3. `POST /report/build` 基于当前对话生成本地 PDF 并返回 `file://` 路径

### API 测试示例

> 所有请求体字段已统一为 `sid`，旧的 `session_id` 不再使用。

```bash
# 获取首问
curl -X POST "http://127.0.0.1:8080/dm/step" \
  -H "Content-Type: application/json" \
  -d '{"sid":"demo-session","role":"user"}'

# 追加一次对话轮次
curl -X POST "http://127.0.0.1:8080/dm/step" \
  -H "Content-Type: application/json" \
  -d '{"sid":"demo-session","role":"user","text":"最近睡得不太好"}'

# 生成报告
curl -X POST "http://127.0.0.1:8080/report/build" \
  -H "Content-Type: application/json" \
  -d '{"sid":"demo-session"}'
```

### 组件说明

- **ASR 适配器**：`services/audio/asr_adapter.py` 统一封装 Stub 与听悟实现，默认回退到文本回声模式，仅在提供听悟凭据时启用真实识别。
- **TTS Stub**：`services/tts/tts_adapter.py` 仅记录日志。可在此处集成 CoSyVoice 或其他语音合成服务，并在 `.env` 中配置所需的供应商凭据。
- **LLM Stub**：`services/llm/json_client.py` 默认基于关键词返回结构化结果；如在 `.env` 中配置 `DEEPSEEK_API_BASE` 与 `DEEPSEEK_API_KEY`，会尝试调用兼容 `/chat/completions` 的 JSON-only 接口，失败后自动回退到 Stub。
- **LangGraph Orchestrator**：`services/orchestrator/langgraph_min.py` 实现了最小 ask → collect_audio → llm_analyze → clarify/risk_check → advance_or_finish → summarize 的流程，最多触发两次澄清，并在检测到高风险时立即打断。
- **风险引擎**：`services/risk/engine.py` 使用强触发关键词识别高风险事件，并写入 `risk:events`。
- **PDF 报告**：`services/report/build.py` 使用 Jinja2 + WeasyPrint 生成临时 PDF 文件并返回本地路径。
- **数据存储**：`services/store/repository.py` 封装 Redis 访问，默认回退到内存实现（无 Redis 环境时方便调试）。存储键包括 `session`、`score`、`transcripts`、`risk:events`、`oss:{sid}` 等。

## 替换为真实服务

- **听悟 ASR**：`services/audio/asr_adapter.py` 会在检测到听悟凭据时自动启用 `services/audio/tingwu_client.py`，只需在 `.env` 中配置 `ALIBABA_CLOUD_ACCESS_KEY_ID`、`ALIBABA_CLOUD_ACCESS_KEY_SECRET`、`ALIBABA_TINGWU_APPKEY`（或 `TINGWU_APPKEY`）、`TINGWU_APP_ID` 即可。
- **CoSyVoice TTS**：在 `services/tts/tts_adapter.py` 中调用真实语音合成接口，返回或缓存生成的语音资源。
- **真实 LLM**：在 `.env` 配置 `DEEPSEEK_API_BASE`（可选）、`DEEPSEEK_API_KEY`，即可通过 OpenAI 兼容接口返回 JSON，或直接修改 `services/llm/json_client.py` 以适配其他供应商。

## 听悟实时识别工作流

`services/audio/tingwu_client.py` 封装了 OpenAPI + Realtime SDK 的完整流程：创建实时任务、通过 NLS SDK 建会话推流 16 kHz 单声道音频，并在任务结束时停止服务返回整段文本。

- **必需凭据**：
  - `ALIBABA_CLOUD_ACCESS_KEY_ID`
  - `ALIBABA_CLOUD_ACCESS_KEY_SECRET`
  - `ALIBABA_TINGWU_APPKEY`（或兼容字段 `TINGWU_APPKEY`）
  - `TINGWU_APP_ID`
- **推荐转码命令**：`ffmpeg -y -i input.wav -ac 1 -ar 16000 output.wav`

## 注意事项

- 禁止引入向量检索或 RAG 依赖，所有证据均基于最近分段直接传递。
- `.env.example` 提供了所有必要的环境变量，请根据实际部署环境调整。

## 开发工具

- 清理指定会话缓存：`python scripts/cleanup_session.py --sid <SESSION_ID>`
- （慎用）清空当前 Redis 数据库：`python scripts/cleanup_session.py --all` 并按提示输入 `FLUSH`

```bash
python /scripts/cleanup_session.py --sid demo1
python /scripts/cleanup_session.py --all
```

## Redis 配置与调优

- 配置 `/etc/redis/redis.conf` 时建议开启 `appendonly yes`，保留快照 `save 900 1`，如需限制内存可设置 `maxmemory-policy allkeys-lru`。
- Redis 连接串示例（含密码）：`REDIS_URL=redis://:StrongPass@localhost:6379/0`。
- 备份命令：`redis-cli SAVE` 或 `redis-cli BGREWRITEAOF`。
- 常用排错命令：`redis-cli INFO`、`redis-cli SLOWLOG GET`、`redis-cli MONITOR`（开发期使用）。
