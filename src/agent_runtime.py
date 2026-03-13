"""Shared agent prompt and runtime guard for the local-first search flow."""

from dataclasses import dataclass
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions, PermissionResultAllow, PermissionResultDeny

from src.tools import create_tools_server

_LOCAL_SEARCH_FIRST_MESSAGE = (
    "This turn's first search tool must be local_search. "
    "Call local_search first with a short English query and only the filters you know. "
    "Only if local_search returns total=0 or a local-search error may you call "
    "parse_search_query or pdl_search."
)

_ALLOWED_TOOLS = [
    "mcp__people-search__local_search",
    "mcp__people-search__parse_search_query",
    "mcp__people-search__pdl_search",
    "mcp__people-search__score_and_filter_results",
    "mcp__people-search__auto_relax_params",
    "mcp__people-search__suggest_narrowing",
    "mcp__people-search__pdl_enrich",
]

_BASE_SYSTEM_PROMPT = """\
你是一个专业的人脉搜索助手。用中文与用户交流。

## 核心原则
- 对每个新的搜索请求，第一个搜索工具必须是 local_search。
- local_search 返回 total > 0 后，不要再调用 parse_search_query 或 pdl_search，直接基于本地结果回答。
- 只有当 local_search 返回 total = 0，或本地搜索明确报错时，才允许走 PDL 兜底流程：parse_search_query -> pdl_search。
- 调用 local_search 时，只传你确定的字段；不要为了凑参数传空字符串、unknown、null。
- local_search 的 query 使用 2 到 6 个英文关键词；country、seniority、industry 用英文小写；page 默认 0，size 默认 10。
- pdl_search 会消耗 PDL 搜索额度。如果返回额度/账单错误，要明确告诉用户，不要伪装成“没有结果”。

## 支持场景
- recruiting：招聘、找候选人、技术人才
- marketing：营销获客、找目标受众
- kol：找博主、内容创作者、意见领袖
- sales：找决策人、BD 对象、采购负责人

## 工作流程

### 第一步：理解需求
- 如果信息明显不足且会导致结果过宽，最多追问 1 个最关键的问题。
- 如果用户说“直接搜”，或者需求已经足够明确，立即搜索。

### 第二步：本地搜索
- 第一个搜索工具必须是 local_search。
- source="local" 时，明确告诉用户结果来自本地数据库（免费）。
- 用户说“换一批”时，复用上一轮已确认的 query 和 filters，再次调用 local_search，把 page 加 1。

### 第三步：PDL 兜底
- 只有 local_search total=0 或报错时，才允许：
  1. 调用 parse_search_query 生成 PDL SQL
  2. 调用 pdl_search 执行全球搜索
- source="pdl" 时，明确告诉用户结果来自全球数据库（PDL）。
- 如果 pdl_search 返回 402/额度错误，直接说明是 PDL 配额问题，并建议用户缩小本地搜索条件或稍后重试。

### 第四步：结果整理
- 结果很多且噪音较大时，可以调用 score_and_filter_results 做排序和摘要。
- 结果过多（>1000）时，可以调用 suggest_narrowing 生成追问建议。
- 结果很少（0 或 <5）且你已经在走 PDL 流程时，可以调用 auto_relax_params 放宽条件重试。

## 工具顺序示例
示例 1：
用户：直接搜美国的 machine learning engineer
先调用：local_search(query="machine learning engineer", country="united states", page=0, size=10)
如果 total > 0：直接回答，不要调用 parse_search_query，不要调用 pdl_search

示例 2：
用户：找旧金山量子计算公司 CTO
先调用：local_search(query="quantum computing cto", country="united states", page=0, size=10)
如果 total = 0：再调用 parse_search_query，然后调用 pdl_search
"""

_JSON_RESULTS_APPENDIX = """

## 输出格式
- 有搜索结果时，必须在回复末尾输出一个 JSON 代码块：

```json
{"type":"results","scenario":"recruiting","sql":"","total":123,"people":[...],"summary":"...","has_more":true,"page":0,"source":"local"}
```

- scenario 只能是 recruiting、marketing、kol、sales 之一。
- local_search 命中时，sql 设为 ""。
- people 保留字段：name, title, company, location, company_industry, company_size, linkedin_url, has_email, has_phone
- 如果本轮只是追问用户，不要输出 JSON。
"""


def build_system_prompt(include_json_results: bool = False) -> str:
    """Build the shared agent system prompt."""
    if include_json_results:
        return _BASE_SYSTEM_PROMPT + _JSON_RESULTS_APPENDIX
    return _BASE_SYSTEM_PROMPT


@dataclass
class SearchFlowGuard:
    """Rejects PDL-first tool usage so the agent stays local-first."""

    local_search_attempted: bool = False

    def start_turn(self) -> None:
        self.local_search_attempted = False

    async def can_use_tool(self, tool_name: str, tool_input: dict[str, Any], context: Any) -> Any:
        del tool_input, context

        if tool_name == "local_search":
            self.local_search_attempted = True
            return PermissionResultAllow()

        if tool_name == "pdl_enrich":
            return PermissionResultAllow()

        if tool_name in {"parse_search_query", "pdl_search", "auto_relax_params", "suggest_narrowing"}:
            if not self.local_search_attempted:
                return PermissionResultDeny(
                    message=_LOCAL_SEARCH_FIRST_MESSAGE,
                    interrupt=False,
                )

        return PermissionResultAllow()


def create_agent_options(
    *,
    include_json_results: bool,
    max_turns: int,
    system_prompt: str | None = None,
    hooks: dict[str, list[Any]] | None = None,
) -> tuple[SearchFlowGuard, ClaudeAgentOptions]:
    """Create Claude SDK options with the shared prompt and local-first guard."""
    guard = SearchFlowGuard()
    options = ClaudeAgentOptions(
        allowed_tools=_ALLOWED_TOOLS,
        system_prompt=system_prompt or build_system_prompt(include_json_results),
        mcp_servers={"people-search": create_tools_server()},
        max_turns=max_turns,
        permission_mode="bypassPermissions",
        can_use_tool=guard.can_use_tool,
        hooks=hooks or {},
        model="claude-sonnet-4-20250514",
    )
    return guard, options
