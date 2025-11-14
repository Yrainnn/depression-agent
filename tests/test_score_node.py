import threading
import time

from services.orchestrator.langgraph_core.nodes.node_score import ScoreNode
from services.orchestrator.langgraph_core.state_types import ItemContext, SessionState
from services.orchestrator.langgraph_core.llm_tools import LLM, ScoreItemTool


def test_score_node_runs_in_parallel(monkeypatch):
    state = SessionState(sid="sid-1")
    for idx in range(1, 5):
        state.item_contexts[idx] = ItemContext(
            item_id=idx,
            item_name=f"Item {idx}",
            dialogue=[{"role": "user", "text": f"answer {idx}"}],
            summary=f"Summary {idx}",
            facts={"k": idx},
            themes=[f"theme-{idx}"],
            risks=[],
        )

    threads_used = []
    barrier = threading.Barrier(4)

    def fake_call(tool, payload):
        assert tool is ScoreItemTool
        threads_used.append(threading.current_thread().name)
        try:
            barrier.wait(timeout=1)
        except threading.BrokenBarrierError:
            pass
        time.sleep(0.05)
        return {"score": payload["facts"]["k"]}

    monkeypatch.setattr(LLM, "call", fake_call)

    node = ScoreNode("score", max_workers=4)
    result = node.run(state)

    assert result["analysis"]["total_score"]["sum"] == sum(range(1, 5))
    # 至少启用了三个不同的工作线程
    worker_threads = {name for name in threads_used if "ThreadPoolExecutor" in name}
    assert len(worker_threads) >= 3
