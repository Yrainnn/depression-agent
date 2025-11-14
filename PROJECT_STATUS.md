# 项目进度与后续任务

## 当前逻辑检查
- **风险筛查**：`RiskNode` 在进入策略与澄清节点前统一通过 LLM 进行风险评估，标准化输出风险等级、触发词、理由与建议；高危时自动添加升级提示，保障风险路径触发的完备性。【F:services/orchestrator/langgraph_core/nodes/node_risk.py†L1-L47】
- **策略提问**：`StrategyNode` 在生成问题时会读取 `STRATEGY_DESCRIPTIONS` 中的策略元数据，将“策略上下文 + 原模板”拼接后再调用 LLM，使问句既保留原有动态加边逻辑，又包含策略名称、目标、语气等信息。【F:services/orchestrator/langgraph_core/nodes/node_strategy.py†L1-L147】【F:services/orchestrator/prompts/strategy_descriptions.py†L1-L15】
- **会话更新**：`UpdateNode` 会把用户回答写回对话上下文，并调用事实抽取、主题识别、滚动摘要等工具更新患者画像，同时维护 `current_strategy`、`waiting_for_user` 等控制位，确保回合状态转换正确。【F:services/orchestrator/langgraph_core/nodes/node_update.py†L1-L65】
- **模板构建**：`TemplateBuilderAgent` 目前会向 LLM 请求包含 YAML 与策略描述的 JSON，解析后生成题目模板和 `strategy_descriptions.py`，从而为策略节点提供结构化上下文数据；缺省情况下会在测试或离线环境写入空描述并保持兼容性。【F:services/orchestrator/langgraph_core/template_builder_agent.py†L1-L163】
- **LLM 工具链**：`LLMToolBox` 在 DeepSeek 未就绪时自动退回本地 fallback，所有测试用例已覆盖模板构建、策略流转等核心路径，当前全部通过。【F:services/orchestrator/langgraph_core/llm_tools.py†L1-L213】【b9a0e7†L1-L13】

## 后续任务清单
1. **补充策略描述库**：根据临床策略手册完善 `STRATEGY_DESCRIPTIONS` 中其余策略（S1-Sn）的名称、目标、语气等字段，并确保与模板内的策略 ID 对齐。
2. **接入 DeepSeek 生产环境**：为 `DeepSeekJSONClient` 配置真实的 API 网关、模型与鉴权，完善熔断/重试日志，并补充集成测试验证统一的 `chat`/`call_json` 输出格式。【F:services/llm/json_client.py†L1-L229】
3. **完善模板生成链路**：针对真实 LLM 输出编写更健壮的解析与校验（如字段缺失、描述为空的回退策略），并在 `TemplateBuilderAgent` 增加对字段有效性的单元测试覆盖。【F:services/orchestrator/langgraph_core/template_builder_agent.py†L60-L126】
4. **语音合成集成 CosyVoice**：在 `TTSAdapter` 中落地 DashScope/CosyVoice 生产调用链，补充音频格式自适应、错误上报与缓存策略，同时为在线/离线模式设计统一接口。【F:services/tts/tts_adapter.py†L1-L146】
5. **风险节点升级**：结合 DeepSeek 风险模型增加更多细粒度标签（例如自伤计划、被动求死、第三方风险），并在 `RiskNode` 中扩展结构化返回值以支撑后续人工干预策略。【F:services/orchestrator/langgraph_core/nodes/node_risk.py†L21-L47】
6. **Clarify/Strategy 评估链路接入**：为策略与澄清节点增加与 DeepSeek 控制器/澄清模型的端到端测试，确保动态加边、澄清分支与评分节点在真实 LLM 场景下保持一致。【F:services/orchestrator/langgraph_core/nodes/node_strategy.py†L96-L147】【F:services/orchestrator/langgraph_core/nodes/node_update.py†L32-L65】
