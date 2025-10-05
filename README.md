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

# 测试 TTS：生成占位语音
curl -X POST "http://127.0.0.1:8080/tts/say" \
  -H "Content-Type: application/json" \
  -d '{"sid":"demo-session","text":"您好，这是语音播报测试。"}'

# 生成报告
curl -X POST "http://127.0.0.1:8080/report/build" \
  -H "Content-Type: application/json" \
  -d '{"sid":"demo-session"}'
```

在 Gradio 评估页中，每轮对话会自动播放 /dm/step 返回的 `tts_url` 对应语音。

### 组件说明

- **ASR Stub**：`services/audio/asr_adapter.py` 将文本直接映射为单个分段。后续如需接入阿里云听悟（TingWu）等服务，可在此替换实现，并在 `.env` 中配置 `ALIBABA_CLOUD_ACCESS_KEY_ID`、`ALIBABA_CLOUD_ACCESS_KEY_SECRET`、`TINGWU_APPKEY` 等凭据。
- **TTS Stub**：`services/tts/tts_adapter.py` 会生成 16 kHz 单声道的占位 WAV 文件并返回 `file://` 路径，后续可替换为 CoSyVoice 或其他语音合成服务。
- **LLM Stub**：`services/llm/json_client.py` 默认基于关键词返回结构化结果；如在 `.env` 中配置 `DEEPSEEK_API_BASE` 与 `DEEPSEEK_API_KEY`，会尝试调用兼容 `/chat/completions` 的 JSON-only 接口，失败后自动回退到 Stub。
- **LangGraph Orchestrator**：`services/orchestrator/langgraph_min.py` 实现了最小 ask → collect_audio → llm_analyze → clarify/risk_check → advance_or_finish → summarize 的流程，最多触发两次澄清，并在检测到高风险时立即打断。
- **风险引擎**：`services/risk/engine.py` 使用强触发关键词识别高风险事件，并写入 `risk:events`。
- **PDF 报告**：`services/report/build.py` 使用 Jinja2 + WeasyPrint 生成临时 PDF 文件并返回本地路径。
- **数据存储**：`services/store/repository.py` 封装 Redis 访问，默认回退到内存实现（无 Redis 环境时方便调试）。存储键包括 `session`、`score`、`transcripts`、`risk:events`、`oss:{sid}` 等。

## 替换为真实服务

- **TingWu ASR**：`services/audio/asr_adapter.py` 通过 `services/audio/tingwu_client.py` 调用阿里听悟 CreateTask → WebSocket 推流 → GetTaskInfo 流程，自动回退到 Stub。
- **CoSyVoice TTS**：在 `services/tts/tts_adapter.py` 中调用真实语音合成接口，返回或缓存生成的语音资源。
- **真实 LLM**：在 `.env` 配置 `DEEPSEEK_API_BASE`（可选）、`DEEPSEEK_API_KEY`，即可通过 OpenAI 兼容接口返回 JSON，或直接修改 `services/llm/json_client.py` 以适配其他供应商。

## 接入听悟实时识别（CreateTask→WS 推流→GetTaskInfo）

- **必要参数**：`ALIBABA_CLOUD_ACCESS_KEY_ID`、`ALIBABA_CLOUD_ACCESS_KEY_SECRET`、`TINGWU_APPKEY`（或 `ALIBABA_TINGWU_APPKEY`）、`TINGWU_REGION`（默认 `cn-beijing`）、`TINGWU_SAMPLE_RATE`（采样率，建议 `16000`）、`TINGWU_FORMAT`（音频格式，如 `pcm`）。
- **实时流程（项目已封装在 `services/audio/tingwu_client.py`）**：
  1. 调用 `CreateTask` 获取会话 `record_id` 与推流地址。
  2. 按官方文档使用 WebSocket 推流入口（默认 `wss://tingwu.aliyuncs.com/ws/v1`）发送 `start` → 音频帧 → `stop` 控制消息，推送 16 kHz 单声道音频。
  3. 解析增量回包并在识别完成后调用 `GetTaskInfo` 拉取最终结果（离线兜底可重试该接口）。
- **采样率与格式**：实时识别推荐 16 kHz/单声道，`TINGWU_SAMPLE_RATE=16000`、`TINGWU_FORMAT=pcm`。若使用 8 kHz 或其他压缩格式需与推流参数保持一致。
- **限额提示**：根据官方文档，`CreateTask` QPS ≈ 20，`GetTaskInfo` QPS ≈ 100；请在批量处理或并发调用时做好限速与退避策略。

## 注意事项

- 阶段一无需接入 DeepSeek 或 CoSyVoice 真接口即可跑通 ASR→问答→报告的闭环，相关环境变量预留即可。
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
