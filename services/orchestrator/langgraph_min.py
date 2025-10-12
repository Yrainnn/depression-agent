"""
DeepSeek Controller Orchestrator 修正版 v7
特性：
- 自动记录已问问题/条目（防止重复）
- item_id 与 state.index 同步推进
- 澄清次数上限控制
- finish 正常终止并打印诊断日志
"""

import json
import time
from services.llm.json_client import analyze_json
from services.orchestrator.questions_hamd17 import pick_primary


class LangGraphMini:
    """轻量化 DeepSeek 控制器状态机"""

    def __init__(self):
        self.state = None
        self.max_clarify = 2  # 最大澄清次数
        self.start_time = time.time()

    def step(self, state):
        print(f"🧠 进入 step(), 当前题号={state.index}")
        self.state = state

        # 初始化防重集合
        if not hasattr(state, "asked_questions"):
            state.asked_questions = set()
        if not hasattr(state, "asked_items"):
            state.asked_items = set()
        if not hasattr(state, "clarify_count"):
            state.clarify_count = 0

        # ========== 调用 DeepSeek 控制器 ==========
        print(f"🧩 调用 DeepSeek 控制器生成第{state.index}题或澄清问")
        t0 = time.time()
        decision = analyze_json(state)
        print(f"🔥 DeepSeek plan_turn 已执行")
        print(f"⏱️ DeepSeek 调用耗时 {time.time()-t0:.2f} 秒")

        if not decision or not isinstance(decision, dict):
            print("⚠️ DeepSeek 返回无效，启用 fallback。")
            ask_text = pick_primary(state.index)
            state.index += 1
            return ask_text

        action = decision.get("action", "")
        ask_text = decision.get("next_utterance", "").strip()
        item_id = decision.get("current_item_id", None)

        # ========== item_id 同步修复 ==========
        if isinstance(item_id, int):
            if item_id > state.index:
                state.index = item_id
        else:
            item_id = state.index

        # ========== 去重机制 ==========
        if ask_text in state.asked_questions:
            print(f"⚠️ 检测到重复问句：{ask_text}，跳过。")
            state.index += 1
            ask_text = pick_primary(state.index)
        elif item_id in state.asked_items:
            print(f"⚠️ item_id={item_id} 已问过，跳过重复。")
            state.index += 1
            ask_text = pick_primary(state.index)
        else:
            state.asked_questions.add(ask_text)
            state.asked_items.add(item_id)

        # ========== 打印 DeepSeek 返回内容 ==========
        print(f"🤖 DeepSeek 返回: {json.dumps(decision, ensure_ascii=False)}")

        # ========== Clarify / Ask / Finish 判定 ==========
        if action == "clarify":
            state.clarify_count += 1
            print(f"🗣️ DeepSeek 要求继续澄清，第 {state.clarify_count} 次。")
            if state.clarify_count >= self.max_clarify:
                print("⚠️ 达到最大澄清次数上限，强制推进。")
                state.index += 1
                state.clarify_count = 0
                ask_text = pick_primary(state.index)
            else:
                # 同一题澄清不推进 index
                pass

        elif action == "ask":
            print(f"📈 DeepSeek 决策推进至第 {state.index + 1} 题。")
            state.index += 1
            state.clarify_count = 0

        elif action == "finish":
            print("🏁 DeepSeek 判断评估结束，输出终止问句。")
            state.finished = True

        # ========== 返回问句 ==========
        print(f"📢 最终输出问句: {ask_text}")
        return ask_text
