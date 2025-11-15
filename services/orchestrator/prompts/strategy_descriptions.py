"""Strategy description templates used to enrich LLM prompts."""

from __future__ import annotations

from typing import Dict, Mapping

StrategyDescription = Mapping[str, str]

STRATEGY_DESCRIPTIONS: Dict[str, StrategyDescription] = {
    "S1": {
        "name": "提供选项与场景化锚定",
        "description": "当患者回答模糊时，医生主动列举症状表现形式或特定生活场景，将评估锚定到可观察的具体情境中，降低患者认知负荷。",
        "tone": "引导性、具体性",
    },
    "S2": {
        "name": "问题簇交叉验证",
        "description": "针对单个评估项目，连续抛出多个关联问题，形成症状维度矩阵进行交叉验证，确保评估准确性。",
        "tone": "系统性、全面性",
    },
    "S3": {
        "name": "三维锚定",
        "description": "对高风险或复杂症状，构建时间、频率、强度三维坐标系，精确锁定症状在病程中的位置。",
        "tone": "精确性、结构性",
    },
    "S4": {
        "name": "比较框架",
        "description": "引入正常-异常、病理性-情境性等参照系，帮助患者定位自身症状的严重程度与性质。",
        "tone": "比较性、正常化",
    },
    "S5": {
        "name": "元认知评估",
        "description": "不仅询问症状有无，更追问患者如何理解症状，评估自知力与病识感，影响诊断分类。",
        "tone": "探索性、反思性",
    },
    "S6": {
        "name": "微观追问",
        "description": "捕捉患者口语中的关键词（如'断断续续'），立即打断进行颗粒度更细的拆解，体现专注倾听。",
        "tone": "细致性、即时性",
    },
    "S7": {
        "name": "性质鉴别",
        "description": "主动区分症状的时间模式（如急性vs慢性、发作性vs持续性），用于诊断亚型鉴别。",
        "tone": "鉴别性、分析性",
    },
    "S8": {
        "name": "回声确认",
        "description": "高频使用回声确认语（如'是吧？'），确保信息准确录入并给予患者修正机会，形成互动闭环。",
        "tone": "确认性、互动性",
    },
    "S9": {
        "name": "弹性时间轴",
        "description": "以'最近两周'为基准锚点，灵活延伸至童年期、远期病程及未来担忧，实现结构化时间框架下的回溯与前瞻。",
        "tone": "灵活性、结构性",
    },
    "S10": {
        "name": "适应性替代",
        "description": "在未使用解释评估目的、量化标尺或深度共情等策略时，采用非对立性追问、举例辅助描述和效率优先提问等替代模式。",
        "tone": "适应性、效率性",
    },
}
