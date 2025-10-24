# 项目架构速览

本文档面向评委，帮助快速理解 **Depression Agent** 的主要模块、数据流与扩展接口。整体框架围绕「LangGraph 流程管控 + 结构化 Prompt 工程 + 临床可审计服务」展开，按 **API 编排层 → 服务能力层 → 存储与配置层 → 前端交互层 → 运维测试** 组织。

## 1. 运行时角色

| 角色 | 关键目录 / 文件 | 职责摘要 |
| --- | --- | --- |
| API 服务 | `apps/api/main.py`, `apps/api/router_dm.py`, `apps/api/router_asr_tingwu.py` | 基于 FastAPI 提供健康检查、Prometheus 指标、会话驱动 (`/dm/step`)、报告生成、ASR 转写与 TTS 等 REST 接口，是所有客户端的统一入口。 |
| 会话编排 | `services/orchestrator/langgraph_min.py`, `services/orchestrator/questions_hamd17.py` | LangGraph 风格的最小化调度器，执行 HAMD-17 问诊节点、澄清分支、风险守卫与流程收敛，按需触发 LLM、ASR、TTS、报告与数字人合成。 |
| 语言模型管道 | `services/llm/json_client.py`, `services/llm/prompts.py` | 封装 DeepSeek JSON-only 通道，加载/覆写 Prompt 模板，具备重试、熔断与澄清补问策略，输出结构化评分与流程控制指令。 |
| 语音与多模态 | `services/audio/tingwu_client.py`, `services/audio/tingwu_async_client.py`, `services/audio/asr_adapter.py`, `services/tts/tts_adapter.py`, `services/digital_human/` | 适配阿里通义听悟（实时/离线）ASR、语音合成 Stub 及数字人生成链路，支持本地文件上传与 WebSocket 推流。 |
| 风险与报告 | `services/risk/engine.py`, `services/report/build.py` | 风险引擎基于关键词/规则识别高危语句；报告模块组合 Jinja2 + WeasyPrint 生成 PDF，并输出量表明细。 |
| 对象存储 | `services/oss/client.py`, `services/oss/uploader.py` | 报告 PDF / TTS 音频上传 OSS 并生成公网 URL，若配置缺失则回退到本地文件 URI。 |
| 会话存储 | `services/store/repository.py` | 优先写入 Redis（含密码注入、TTL 管理），在 Redis 不可用时自动回退内存字典，统一维护进度、转写、风险与分数。 |
| 通用配置 | `packages/common/config.py` | Pydantic Settings 解析 `.env`，集中管理 DeepSeek、TingWu、OSS、Redis 等运行参数，供服务层按需引用。 |
| 前端 | `apps/ui-gradio/app.py` | Gradio 应用承载实时转写、流程进度、TTS 播报与报告链接展示，实现会话录入与反馈闭环。 |

## 2. 端到端流程

1. **会话入口**：客户端调用 `/dm/step`，`ConversationRepository` 根据 `sid` 装载上下文，`LangGraphMini` 返回下一问句与进度；必要时调用 ASR/TTS/Digital Human 子模块。
2. **多模态采集**：语音通过 `/upload/audio` 或 TingWu WebSocket 进入 `TingwuClientASR`，文本与转写结果统一写入仓储用于 LLM 证据聚合。
3. **结构化推理**：`DeepSeekJSONClient` 根据 `prompts.py` 模板生成评分、澄清需求或流程决策，异常时触发熔断回退。
4. **风险守卫**：`services/risk/engine.py` 每轮检查敏感关键词，必要时修改 LangGraph 状态（如提前终止、安全确认提示）。
5. **报告与制品**：`services/report/build.py` 汇总量表得分生成 PDF，`OSSUploader` 负责上传；`TTSAdapter`/`digital_human.service` 生成语音与视频制品返回前端。
6. **可观测性**：`/health` 与 `/metrics` 由 `apps/api/main.py` 暴露，支持 Prometheus 抓取；`/debug/redis` 快速确认 Redis 可用性。

## 3. 目录概览

```text
apps/                 # API 与 Gradio 前端入口
  api/
  ui-gradio/
services/             # 业务能力层（LLM、ASR、TTS、LangGraph、报告、风险、OSS、存储、数字人）
packages/common/      # 配置解析
scripts/              # 启动/清理工具（run_api.sh、run_ui.sh、cleanup_session.py）
services/__init__.py  # 服务模块聚合入口
requirements.txt      # 统一依赖
```

## 4. 测试与运维支持

- `tests/` 目录覆盖 LangGraph 澄清流程、DeepSeek JSON 客户端、报告生成、TTS 适配器等关键路径，保障核心链路可回归。
- `scripts/cleanup_session.py` 提供 Redis 或内存仓储的会话清理；`run_api.sh` / `run_ui.sh` 简化本地调试。

## 5. 扩展策略

- **服务可插拔**：ASR、TTS、OSS、LLM 均以接口/适配器模式封装，便于替换云厂商或离线方案。
- **Prompt 资产化**：环境变量可覆写默认 Prompt 路径，实现多版本模板 A/B 测试。
- **容灾回退**：Redis、DeepSeek、OSS 等外部依赖失效时均内置告警与回退逻辑，维持临床会话连续性。

> 如需更细粒度的业务流程，可配合 `README.md` 中的 Mermaid 流程图与接口示例一起参考。
