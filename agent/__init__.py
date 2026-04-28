"""Agent 核心 — 构建与运行，三步链路：识别 → 精确查找 → 语义检索"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from config import load_config
from database import ShipDatabase
from tools import build_tools

logger = logging.getLogger(__name__)

# ── System Prompt（含 Skills）──────────────────

SYSTEM_PROMPT = """你是船弦号识别助手。VLM 已完成图像预识别，你根据识别结果决策后续操作。

## 可用工具
- lookup_by_hull_number：通过弦号精确查找数据库
- retrieve_by_description：通过描述语义检索数据库

## 执行链路（必须严格按顺序执行）

### Step 1: 根据 VLM 结果判断情况

**情况 A — 无弦号（hull_number 为空）：**
→ 直接调用 retrieve_by_description 用 description 做语义检索。

**情况 B — 有弦号（无论清晰或模糊）：**
→ 调用 lookup_by_hull_number 精确查找。
  → found=true：返回「库内确定id：{hull_number}，描述：{description}」
  → found=false：调用 retrieve_by_description 语义检索，返回「可能id：{候选列表}」

注意：模糊弦号（clarity="blurry"）时，VLM 已返回 hull_box 坐标，无需二次识别，直接按上述流程处理。

### Step 2: 返回结果
返回格式：「弦号：{hull_number}，描述：{description}，匹配类型：{exact/semantic/none}」

## 禁止
- 不要编造弦号或描述
- 不要跳过任何步骤
- 不要同时调用多个工具
"""

# ── 无 Few-shot 示例（避免误导 Agent）──


class AgentResult:
    """Agent 运行结果，包含结构化信息供 pipeline 使用。"""

    def __init__(
        self,
        hull_number: str = "",
        description: str = "",
        match_type: str = "none",
        semantic_match_ids: list[str] | None = None,
        answer: str = "",
        hull_box: list[float] | None = None,
        clarity: str = "",
    ):
        self.hull_number = hull_number
        self.description = description
        self.match_type = match_type        # "exact" | "semantic" | "none"
        self.semantic_match_ids = semantic_match_ids or []
        self.answer = answer
        self.hull_box = hull_box            # 弦号框相对坐标 [rx1, ry1, rx2, ry2]
        self.clarity = clarity              # "clear" | "blurry" | ""


class ShipHullAgent:
    """船弦号识别 Agent 封装。链路：VLM预识别 → Agent决策（lookup/retrieve）。"""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or load_config()

        app_cfg = self.config.get("app", {})
        llm_cfg = self.config.get("llm", {})

        logging.basicConfig(
            level=getattr(logging, app_cfg.get("log_level", "INFO").upper(), logging.INFO),
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        )

        self.db = ShipDatabase(config=self.config)
        # Agent 模式：不含 recognize_ship（VLM 由 pipeline 预调用），含 lookup + retrieve
        self.tools = build_tools(self.db, include_recognize=False)

        self._llm = ChatOpenAI(
            model=llm_cfg.get("model", "Qwen/Qwen3-VL-4B-AWQ"),
            api_key=llm_cfg.get("api_key", "abc123"),
            base_url=llm_cfg.get("base_url", "http://localhost:7890/v1"),
            temperature=llm_cfg.get("temperature", 0.0),
            max_tokens=llm_cfg.get("max_tokens", 2048),
        )

        self._agent = create_react_agent(
            model=self._llm,
            tools=self.tools,
            prompt=SYSTEM_PROMPT,
        )

    def run(self, query: str) -> str:
        """运行 Agent，返回自然语言回答。"""
        logger.info("收到查询: %s", query[:100])
        try:
            result = self._agent.invoke({"messages": [HumanMessage(content=query)]})
            answer = result["messages"][-1].content
            logger.info("回答: %s", answer[:100])
            return answer
        except Exception as e:
            logger.exception("Agent 执行失败")
            return f"查询执行失败: {e}"

    def run_with_result(self, query: str) -> AgentResult:
        """运行 Agent，返回结构化结果（供 pipeline 使用）。"""
        try:
            result = self._agent.invoke({"messages": [HumanMessage(content=query)]})
            return self._parse_result(result)
        except Exception as e:
            logger.exception("Agent 执行失败")
            return AgentResult(answer=f"Agent 执行失败: {e}")

    @staticmethod
    def _parse_result(result: dict) -> AgentResult:
        """从 Agent 消息历史中提取结构化结果。"""
        msgs = result.get("messages", [])
        hull_number = ""
        description = ""
        match_type = "none"
        semantic_match_ids: list[str] = []
        answer = msgs[-1].content if msgs else ""
        hull_box = None
        clarity = ""

        for msg in msgs:
            if not isinstance(msg, ToolMessage):
                continue
            try:
                data = json.loads(msg.content)
            except (json.JSONDecodeError, TypeError):
                continue

            # recognize_ship 结果（包含 hull_number, description, clarity, hull_box）
            if "hull_number" in data and "description" in data and "found" not in data and "results" not in data:
                hull_number = data.get("hull_number", "")
                description = data.get("description", "")
                clarity = data.get("clarity", "")
                raw_box = data.get("hull_box")
                if isinstance(raw_box, list) and len(raw_box) == 4:
                    try:
                        coords = [float(v) for v in raw_box]
                        if all(0.0 <= c <= 1.0 for c in coords):
                            hull_box = coords
                    except (ValueError, TypeError):
                        pass

            # lookup_by_hull_number（无论 found 与否都保留 hull_number）
            if "found" in data:
                if data.get("found") is True:
                    match_type = "exact"
                    description = data.get("description", description)
                hull_number = data.get("hull_number", hull_number) or hull_number

            # retrieve_by_description 语义匹配
            if "results" in data:
                results = data["results"]
                if results:
                    semantic_match_ids = [
                        r.get("hull_number", "") for r in results if r.get("hull_number")
                    ]
                    if match_type != "exact":
                        match_type = "semantic"

        return AgentResult(
            hull_number=hull_number,
            description=description,
            match_type=match_type,
            semantic_match_ids=semantic_match_ids,
            answer=answer,
            hull_box=hull_box,
            clarity=clarity,
        )

    def run_verbose(self, query: str) -> list[dict]:
        try:
            result = self._agent.invoke({"messages": [HumanMessage(content=query)]})
            trace = []
            for msg in result["messages"]:
                entry = {"type": msg.type, "content": msg.content}
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    entry["tool_calls"] = [
                        {"name": tc["name"], "args": tc["args"]}
                        for tc in msg.tool_calls
                    ]
                if hasattr(msg, "tool_call_id"):
                    entry["tool_call_id"] = msg.tool_call_id
                trace.append(entry)
            return trace
        except Exception as e:
            logger.exception("Agent 执行失败")
            return [{"type": "error", "content": f"查询执行失败: {e}"}]


_agent_instance: ShipHullAgent | None = None
_agent_config_hash: int = 0
_agent_lock = threading.Lock()


def create_agent(config: dict[str, Any] | None = None) -> ShipHullAgent:
    global _agent_instance, _agent_config_hash
    config_hash = hash(str(config)) if config is not None else 0
    with _agent_lock:
        if _agent_instance is None or config_hash != _agent_config_hash:
            _agent_instance = ShipHullAgent(config)
            _agent_config_hash = config_hash
    return _agent_instance
