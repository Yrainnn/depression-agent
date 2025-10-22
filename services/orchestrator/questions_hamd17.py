from typing import Dict, List

# 汉密尔顿抑郁量表（HAMD-17）题库示例
HAMD17_QUESTION_BANK: Dict[int, dict] = {
    1: {
        "primary": ["抑郁情绪"],
        "clarify": {
            "frequency": ["这些低落的感觉一周大概有几天出现？"],
            "duration": ["每次大概会持续多久？"],
            "severity": ["对日常生活的影响有多大？"],
            "negation": ["有没有明显好转的时候？"],
        },
    },
    2: {
        "primary": ["有罪感"],
        "clarify": {
            "frequency": ["这种兴趣下降大概多久发生一次？"],
            "duration": ["每次持续多久？"],
            "severity": ["影响到工作或生活了吗？"],
            "negation": ["有没有让您感觉好些的时候？"],
        },
    },
    3: {
        "primary": ["自杀倾向"],
        "clarify": {
            "frequency": ["这种想法一周内出现几次？"],
            "duration": ["每次会持续多久？"],
            "severity": ["强度如何？是否难以控制？"],
            "negation": ["现在还有这种想法吗？是否减轻？"],
            "plan": ["是否有具体计划或准备过相关工具？"],
            "safety": ["现在身边是否有人陪伴、能保证您的安全？"],
        },
    },
    4: {
        "primary": ["入睡困难"],
        "clarify": {
            "frequency": ["这些睡眠问题一周内有几晚？"],
            "duration": ["持续了多长时间？"],
            "severity": ["对白天状态影响多大？"],
            "negation": ["有没有睡得比较好的时候？"],
        },
    },
    5: {
        "primary": ["睡眠维持障碍"],
        "clarify": {
            "frequency": ["这种情况多久发生一次？"],
            "duration": ["一次会持续多久？"],
            "severity": ["是否影响到日常计划？"],
            "negation": ["有没有精力较好的时候？"],
        },
    },
    6: {
        "primary": ["早醒"],
        "clarify": {
            "frequency": ["食欲变化多久出现一次？"],
            "duration": ["这种状态持续了多长时间？"],
            "severity": ["体重或饮食习惯有改变吗？"],
            "negation": ["有没有恢复正常的时候？"],
        },
    },
    7: {
        "primary": ["工作和兴趣"],
        "clarify": {
            "frequency": ["疲倦一周大概有几天？"],
            "duration": ["每次疲倦持续多久？"],
            "severity": ["影响到哪些事情？"],
            "negation": ["有没有精力比较充足的时候？"],
        },
    },
    8: {
        "primary": ["精神运动迟缓"],
        "clarify": {
            "frequency": ["注意力问题多久会出现一次？"],
            "duration": ["一次大约持续多久？"],
            "severity": ["是否影响到工作或学习？"],
            "negation": ["有没有集中得比较好的时候？"],
        },
    },
    9: {
        "primary": ["激越"],
        "clarify": {
            "frequency": ["这种紧张感多久出现一次？"],
            "duration": ["每次持续多久？"],
            "severity": ["是否影响到睡眠或社交？"],
            "negation": ["有没有放松的时候？"],
        },
    },
    10: {
        "primary": ["精神性焦虑"],
        "clarify": {
            "frequency": ["这种不安感多久发生？"],
            "duration": ["一次会持续多久？"],
            "severity": ["别人会注意到这种变化吗？"],
            "negation": ["有没有平静放松的时候？"],
        },
    },
    11: {
        "primary": ["躯体性焦虑"],
        "clarify": {
            "frequency": ["这种迟缓多久出现一次？"],
            "duration": ["每次持续多久？"],
            "severity": ["别人是否指出您反应变慢？"],
            "negation": ["有没有恢复正常的时间？"],
        },
    },
    12: {
        "primary": ["胃肠道"],
        "clarify": {
            "frequency": ["这种担心一周有几天？"],
            "duration": ["每次会持续多久？"],
            "severity": ["是否因此频繁就医或查资料？"],
            "negation": ["有没有不太担心的时候？"],
        },
    },
    13: {
        "primary": ["全身症状"],
        "clarify": {
            "frequency": ["这种感觉多久出现一次？"],
            "duration": ["每次持续多久？"],
            "severity": ["是否影响到自我评价或人际？"],
            "negation": ["有没有觉得自己还不错的时候？"],
        },
    },
    14: {
        "primary": ["性症状"],
        "clarify": {
            "frequency": ["这种反应多久出现一次？"],
            "duration": ["通常持续多久？"],
            "severity": ["影响到生活或休息了吗？"],
            "negation": ["有没有明显缓解的时候？"],
        },
    },
    15: {
        "primary": ["疑病倾向"],
        "clarify": {
            "frequency": ["这种情况多久发生一次？"],
            "duration": ["会持续多久？"],
            "severity": ["会影响到睡眠或安全感吗？"],
            "negation": ["有没有比较放松的时候？"],
        },
    },
    16: {
        "primary": ["体重减轻"],
        "clarify": {
            "frequency": ["这种感受多久出现？"],
            "duration": ["每次持续多久？"],
            "severity": ["是否影响到日常活动？"],
            "negation": ["有没有轻松些的时候？"],
        },
    },
    17: {
        "primary": ["自知力"],
        "clarify": {
            "frequency": ["忘事的情况多久发生一次？"],
            "duration": ["影响持续多久？"],
            "severity": ["是否影响到工作或生活安排？"],
            "negation": ["有没有情况好一些的时候？"],
        },
    },
}


def get_first_item() -> int:
    return 1


def get_next_item(current: int) -> int:
    return current + 1 if current < 17 else -1


MAX_SCORE: Dict[int, int] = {
    1: 4,
    2: 4,
    3: 4,
    4: 2,
    5: 2,
    6: 2,
    7: 4,
    8: 4,
    9: 4,
    10: 4,
    11: 4,
    12: 2,
    13: 2,
    14: 2,
    15: 4,
    16: 2,
    17: 4,
}


def pick_primary(item_id: int) -> str:
    arr = HAMD17_QUESTION_BANK.get(item_id, {}).get(
        "primary", ["请描述该条目相关情况。"]
    )
    return arr[0]


def pick_clarify(item_id: int, gap: str) -> str:
    node = HAMD17_QUESTION_BANK.get(item_id, {}).get("clarify", {})
    arr: List[str] = node.get(gap) or node.get("severity") or ["能再具体一点吗？"]
    return arr[0]
