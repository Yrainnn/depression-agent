# Depression Agent

Depression Agent 是一个针对抑郁症随访场景打造的多模态智能体框架，集成了对话编排、音频识别、语音合成、风险评估与报告生成能力。项目以可插拔的方式封装关键服务，便于在原型验证阶段快速迭代，同时为生产部署预留了稳定的扩展接口。

## 核心特性

- **智能体工作流**：基于 LangGraph 构建问诊 → 澄清 → 风险识别 → 总结的闭环流程，支持多轮上下文管理与任务状态追踪。
- **前端实时转写**：Gradio UI 已对接阿里听悟（TingWu）实时转写结果，并修复了前端展示链路，可即时同步用户语音内容。
- **灵活的模型接入**：默认提供 ASR/TTS/LLM Stub，允许按需切换到听悟、CoSyVoice、DeepSeek 等真实服务，失败时自动回退以保障体验。
- **风险与报告能力**：内置高危关键词识别引擎与 PDF 报告生成器，可输出结构化量表评分、风险事件及随访总结。
- **可观测性与调试**：提供健康检查、Prometheus 指标埋点、Redis 缓存抽象以及多种清理脚本，支持开发及上线阶段的运维需求。

## 目录概览

```
apps/
  api/
    main.py            # FastAPI 应用，包含 /health /metrics 及 DM 路由
    router_dm.py       # 抑郁随访问答与报告构建接口
  ui-gradio/
    app.py             # 多模态 Gradio 前端，含实时转写展示
packages/
  common/config.py     # 环境变量与配置解析
services/
  orchestrator/        # LangGraph 最小智能体管线
  audio/               # ASR Stub 与听悟接入实现
  tts/                 # TTS Stub，预留真实语音合成接口
  llm/                 # LLM JSON 客户端（Stub + OpenAI 兼容接口）
  risk/                # 风险评估引擎
  report/              # PDF 报告生成
  store/               # Redis 仓储封装
scripts/
  run_api.sh           # 启动 FastAPI
  run_ui.sh            # 启动 Gradio UI
```

## 快速启动

```bash
pip install -r requirements.txt
cp .env.example .env
./scripts/run_api.sh
./scripts/run_ui.sh
```

默认 API 监听 `0.0.0.0:8080`，UI 监听 `0.0.0.0:7860`。

### API 基础验证

1. `GET /health` → `{ "ok": true }`
2. `POST /dm/step` → 基于文本输入返回下一轮问句与进度信息
3. `POST /report/build` → 基于当前对话生成本地 PDF，并返回 `file://` 路径

### API 调试示例

> 请求字段统一使用 `sid`，旧的 `session_id` 已废弃。

```bash
# 获取首轮问句
curl -X POST "http://127.0.0.1:8080/dm/step" \
  -H "Content-Type: application/json" \
  -d '{"sid":"demo-session","role":"user"}'

# 追加一次对话轮
curl -X POST "http://127.0.0.1:8080/dm/step" \
  -H "Content-Type: application/json" \
  -d '{"sid":"demo-session","role":"user","text":"最近睡得不太好"}'

# 生成占位语音
curl -X POST "http://127.0.0.1:8080/tts/say" \
  -H "Content-Type: application/json" \
  -d '{"sid":"demo-session","text":"您好，这是语音播报测试。"}'

# 构建报告
curl -X POST "http://127.0.0.1:8080/report/build" \
  -H "Content-Type: application/json" \
  -d '{"sid":"demo-session"}'
```

Gradio 前端会在评估页自动播放 `/dm/step` 返回的 `tts_url`，并呈现实时转写文本。

## 组件说明

- **ASR（TingWu + Stub）**：`services/audio/asr_adapter.py` 默认将文本映射为单个转写分段，可切换为听悟实时接口。`.env` 中需配置 `ALIBABA_CLOUD_ACCESS_KEY_ID`、`ALIBABA_CLOUD_ACCESS_KEY_SECRET`、`TINGWU_APPKEY` 等凭据。
- **TTS Stub**：`services/tts/tts_adapter.py` 生成 16 kHz 单声道占位 WAV 文件并返回本地路径，可替换为 CoSyVoice 或其他语音合成服务。
- **LLM 客户端**：`services/llm/json_client.py` 采用关键词规则生成结构化结果；若设置 `DEEPSEEK_API_BASE`、`DEEPSEEK_API_KEY`，将尝试调用兼容 `/chat/completions` 的 JSON-only 接口（已取消额外文本清洗流程），失败后回退到 Stub。
- **LangGraph Orchestrator**：`services/orchestrator/langgraph_min.py` 实现问诊到总结的最小有向图流程，最多触发两次澄清，并在识别高风险时终止对话。
- **风险评估**：`services/risk/engine.py` 识别高危关键词并写入 `risk:events`。
- **报告生成**：`services/report/build.py` 使用 Jinja2 + WeasyPrint 生成临时 PDF。
- **数据存储**：`services/store/repository.py` 封装 Redis 访问，默认回退至内存实现，适合本地调试。

## 配置真实服务

- **听悟实时识别**：`services/audio/tingwu_client.py` 封装 CreateTask → WebSocket 推流 → GetTaskInfo 流程，自动处理鉴权与重试。关键参数包括 `TINGWU_REGION`（默认 `cn-beijing`）、`TINGWU_SAMPLE_RATE`（推荐 `16000`）、`TINGWU_FORMAT`（如 `pcm`）。
- **语音合成服务**：在 `services/tts/tts_adapter.py` 中对接实际供应商，返回可访问的音频资源或缓存路径。
- **大语言模型**：配置 `DEEPSEEK_API_BASE`（支持 `https://api.deepseek.com` 或 `https://api.deepseek.com/v1`）与 `DEEPSEEK_API_KEY` 后，将通过 OpenAI 兼容接口返回 JSON 结果；也可修改 `services/llm/json_client.py` 以适配其他厂商。

### DeepSeek JSON 评分与澄清问

开启真实 LLM 后，系统会在问卷流程中：

- 输出 HAMD-17 结构化 JSON（含 `clarify_need` 指示）
- 对缺失信息条目生成中文澄清问，失败时回退到内置模板

相关提示词集中于 `services/llm/prompts.py`，可通过以下环境变量覆盖：`PROMPT_HAMD17_PATH`、`PROMPT_DIAGNOSIS_PATH`、`PROMPT_MDD_JUDGMENT_PATH`、`PROMPT_CLARIFY_CN_PATH`。

## 注意事项

- 初始阶段无需接入真实 DeepSeek 或 CoSyVoice 服务即可跑通 ASR → 问答 → 报告流程。
- 项目禁止引入向量检索或 RAG 依赖，所有证据基于当前会话上下文传递。
- `.env.example` 提供了所有必要配置项，请根据部署环境调整。

## 运维与调试工具

- 清理指定会话缓存：`python scripts/cleanup_session.py --sid <SESSION_ID>`
- 清空 Redis 数据库（慎用）：`python scripts/cleanup_session.py --all` 并按照提示输入 `FLUSH`

```bash
python scripts/cleanup_session.py --sid demo1
python scripts/cleanup_session.py --all
```

## Redis 配置建议

- 在 `/etc/redis/redis.conf` 中启用 `appendonly yes`，保留 `save 900 1` 快照；如需限制内存，可设置 `maxmemory-policy allkeys-lru`。
- 连接串示例（含密码）：`REDIS_URL=redis://:StrongPass@localhost:6379/0`。
- 常用备份命令：`redis-cli SAVE` 或 `redis-cli BGREWRITEAOF`。
- 常用排错命令：`redis-cli INFO`、`redis-cli SLOWLOG GET`、`redis-cli MONITOR`（开发期使用）。
