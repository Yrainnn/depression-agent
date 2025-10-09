# Depression Agent

**Depression Agent** 是一个聚焦抑郁症随访的多模态智能体框架，目标是在临床质控标准下，提供**可控的问诊流程**、**可靠的风险提示**、以及**可复用的评估报告**。框架基于 **LangGraph** 与结构化提示工程实现：**问诊 → 澄清 → 量化评分 → 风险确认 → 总结与报告** 的闭环，并通过可插拔服务快速接入真实 **ASR（通义听悟）**、**TTS** 与 **LLM** 能力。

---

## ✨ 核心亮点

- **可控流程编排**  
  `services/orchestrator` 使用 LangGraph 显式建模节点与边，约束澄清次数、风险终止条件与报告触发时机，确保全链路符合医疗随访 SOP。

- **提示工程资产化**  
  `services/llm/prompts.py` 将 HAMD-17 评估、澄清问生成、诊断总结等提示模板模块化，支持 `.env` 覆写；配套 **DeepSeek JSON-only 通道** 与故障回退策略，保证结构化输出稳定。

- **多模态协同**  
  Gradio 前端整合 **通义听悟** 转写、TTS 播报与会话进度展示；后端在音频、文本、报告之间以统一 `sid` 贯穿，实现跨模态追踪与溯源。

- **可观测与运维友好**  
  API 暴露 `/health`、`/metrics`，提供 Redis 抽象与清理脚本，测试覆盖关键风险节点与报告链路，便于持续交付与问题定位。

---

## 📁 仓库结构

```text
apps/
  api/
    main.py                   # FastAPI 入口，含 /health、/metrics、/dm、/report 等路由
    router_dm.py              # 对话管理与报告构建 REST 接口
    router_asr_tingwu.py      # ✅ 通义听悟 ASR：上传、转写等 REST 接口
  ui-gradio/
    app.py                    # 多模态前端，集成 Tingwu 转写与随访流程控制
packages/
  common/
    config.py                 # 环境变量解析与配置对象
requirements.txt              # 统一依赖声明
scripts/
  cleanup_session.py          # Redis/内存会话清理工具
  run_api.sh                  # 启动 FastAPI 服务
  run_ui.sh                   # 启动 Gradio 前端
services/
  audio/                      # ASR 接入（通义听悟 SDK / WebSocket / 文件回放）
    tingwu_client.py          # 文件回放式实时会话（NLS SDK），支持分帧推流 .wav/.pcm
    tingwu_async_client.py    # 异步创建/停止实时任务的封装
  llm/                        # JSON-only LLM 客户端与提示模板
  orchestrator/               # LangGraph 流程控制实现
  report/                     # Jinja2 + WeasyPrint 报告生成
  risk/                       # 高危事件识别引擎
  store/                      # Redis 仓储封装及内存回退
  tts/                        # 语音合成 Stub，预留真实供应商对接
  oss/                        # ✅ OSS 上传/下载客户端：产出报告与 TTS 音频的公网 URL（下载/播放）
tests/
  test_deepseek_client.py
  test_orchestrator_clarify.py
  test_orchestrator_report.py
  test_report_build_pdf.py
  test_tts_adapter.py
```

注：services/oss 负责将 PDF 报告与 TTS 语音等制品上传到对象存储并返回公网 URL，以便前端直接下载/播放。

🔁 工作流总览
语音输入 → TingWu 转写 → LangGraph 流程控制 → LLM 结构化推理
    → 风险评估（触发预警/提前终止） → 报告生成 → OSS 公网链接 → 前端播放/下载


会话初始化：前端或外部系统调用 POST /dm/step；services/store 基于 sid 装载/创建上下文。

采集与澄清：前端录音（或上传）→ TingWu 转写；流程根据节点状态自动进入澄清分支，限次控制。

结构化推理：services/llm/json_client.py 调用（DeepSeek JSON-only 或回退）产出 HAMD-17 分项、clarify_need 与临床要点。

风险监测：services/risk/engine.py 每轮扫描高危信号，联动流程（如提前终止/人工干预）。

报告生成：POST /report/build 调用 Jinja2 模板渲染临床摘要、风险事件与量表评分，WeasyPrint 输出 PDF。

制品分发（OSS）：services/oss/client.py 上传 PDF 与 TTS 音频，返回公网 URL；前端直接渲染下载/播放。

🚀 快速开始
pip install -r requirements.txt
cp .env.example .env
./scripts/run_api.sh          # 启动 API（默认 0.0.0.0:8080）
./scripts/run_ui.sh           # 启动 Gradio（默认 0.0.0.0:7860）


.env.example 列出全部关键变量，请按部署环境填写。

🔌 API 验证示例
1) 对话流转（DM）
# 获取首轮问句
curl -X POST "http://127.0.0.1:8080/dm/step"   -H "Content-Type: application/json"   -d '{"sid":"demo-session","role":"user"}'

# 追加一轮对话（文本）
curl -X POST "http://127.0.0.1:8080/dm/step"   -H "Content-Type: application/json"   -d '{"sid":"demo-session","role":"user","text":"最近睡得不太好"}'

2) 听悟转写（文件回放/离线）
# 上传音频（WAV/PCM）
SID=tf_demo
AR=$(curl -s -F "sid=$SID" -F "file=@/tmp/sample16k.wav"   http://127.0.0.1:8080/asr/tingwu/upload | jq -r .audio_ref)

# 触发转写（返回最终文本，或异步状态 + 查询接口）
curl -s -X POST http://127.0.0.1:8080/asr/tingwu/transcribe   -H 'Content-Type: application/json'   -d "{"sid":"$SID","audio_ref":"$AR"}" | jq


路由在 apps/api/router_asr_tingwu.py 中实现，便于前端/第三方系统复用。

3) 语音合成（TTS Stub）
curl -X POST "http://127.0.0.1:8080/tts/say"   -H "Content-Type: application/json"   -d '{"sid":"demo-session","text":"您好，这是语音播报测试。"}'

4) 生成报告（PDF → OSS 公网 URL）
curl -X POST "http://127.0.0.1:8080/report/build"   -H "Content-Type: application/json"   -d '{"sid":"demo-session"}'
# 返回: {"report_url":"https://<oss-domain>/reports/report_<sid>.pdf", ...}

🖥️ Gradio 前端要点

录音/上传音频后，前端自动转为 16 kHz 单声道 PCM，调用 Tingwu 完成转写；

转写文本在页面展示，随后进入文本清洗（DeepSeek） → 问答 → 报告；

对话页自动播放后端返回的 tts_url（可为 OSS 公网 URL）；

建议 gr.Chatbot(type="messages") 以兼容后续版本。

⚙️ 配置真实服务
能力	关键文件	环境变量（示例）	说明
听悟实时识别/文件回放	services/audio/tingwu_client.py, tingwu_async_client.py	ALIBABA_CLOUD_ACCESS_KEY_ID, ALIBABA_CLOUD_ACCESS_KEY_SECRET, ALIBABA_TINGWU_APPKEY, TINGWU_REGION, TINGWU_FORMAT=pcm, TINGWU_SAMPLE_RATE=16000	支持创建实时任务→NLS SDK 推流→收结果；也提供异步创建/停止任务封装
LLM（JSON-only）	services/llm/json_client.py, services/llm/prompts.py	DEEPSEEK_API_BASE, DEEPSEEK_API_KEY	OpenAI 兼容 /chat/completions；异常自动回退规则引擎
TTS	services/tts/*	（自定义，如 COSYVOICE_API_KEY）	以 Stub 为基线，可替换为供应商 SDK，返回本地或公网 URL
OSS	services/oss/client.py	OSS_ENDPOINT, OSS_BUCKET, OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET, （可选 OSS_KEY_PREFIX）	统一上传报告 & 音频制品并生成公网 URL，供前端下载/播放

提示模板可通过如下环境变量覆盖：
PROMPT_HAMD17_PATH、PROMPT_DIAGNOSIS_PATH、PROMPT_MDD_JUDGMENT_PATH、PROMPT_CLARIFY_CN_PATH。

✅ 测试与质量保障
pytest


test_orchestrator_clarify.py：验证风险/澄清分支的上限与流转行为。

test_deepseek_client.py：覆盖 JSON-only 提示工程与回退策略。

test_report_build_pdf.py：确保提示输出变化下 PDF 仍可渲染。

test_tts_adapter.py、test_orchestrator_report.py：检查音频与报告链路一致性。

🛠️ 运维与故障排查

清理指定会话缓存：python scripts/cleanup_session.py --sid <SESSION_ID>

清空 Redis（慎用）：python scripts/cleanup_session.py --all

建议 Redis 配置：appendonly yes、save 900 1；内存策略可用 maxmemory-policy allkeys-lru

常用排错命令：redis-cli INFO、redis-cli SLOWLOG GET、redis-cli MONITOR（开发阶段）

🔒 注意事项

项目不依赖 RAG；所有证据来自当前会话上下文，便于审查和复现。

早期部署阶段可全链路使用 Stub 验证，逐步替换为真实服务。

生产建议对接 Prometheus 采集 /metrics，并配置日志采集以监控失败率与端到端耗时。

🧭 里程碑建议（可选）

 前端**实时转录（WebSocket 推流）**字幕流完善（SentenceBegin/Changed/End 即时更新）。

 DeepSeek 文本清洗提示词迭代 & 标注场景上线。

 TTS 供应商接入与缓存策略。

 报告模板国际化与导出渠道（邮件/OSS/CDN）。

License：根据你的项目策略补充（MIT / Apache-2.0 / 私有）。
