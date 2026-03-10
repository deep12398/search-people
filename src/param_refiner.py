"""LLM-powered search parameter refinement for PDL SQL queries."""

import json
from src.config import PARSE_MODEL, get_anthropic_client, extract_json, system_prompt

RELAX_PROMPT = """\
你是一个 PDL SQL 搜索参数优化师。用户的人脉搜索返回了 0 个结果（或极少结果）。
你需要分析当前 SQL 查询，智能放宽条件。

放宽策略（按优先级）：
1. 扩大地点范围（城市→州/省→国家）
2. 用 LIKE 模糊匹配替代精确匹配
3. 放宽公司规模范围
4. 减少 AND 条件数量
5. 去掉层级限制

注意：不要一次放太多，每次只放宽 1-2 个维度。保留用户最核心的意图。

输入：
- original_query: 用户原始描述
- current_sql: 当前 SQL 查询
- result_count: 当前结果数

输出严格 JSON：
{
  "relaxed_sql": "SELECT * FROM person WHERE ...",
  "changes_made": ["扩大了地点范围", "改为模糊匹配"],
  "explanation": "一句话说明调整原因"
}
"""

NARROW_PROMPT = """\
你是一个搜索参数优化师。用户的人脉搜索返回了太多结果，需要缩小范围。

根据搜索场景生成 2-3 个追问建议：

**recruiting**：从技能栈、工作年限、学历、具体城市等维度缩小
**marketing**：从公司规模、具体行业细分、地域等维度缩小
**kol**：从内容领域细分、特定平台、粉丝量级等维度缩小
**sales**：从公司营收规模、具体行业、决策层级等维度缩小

输入：
- original_query: 用户原始描述
- current_sql: 当前 SQL 查询
- result_count: 当前结果数
- scenario: 搜索场景

输出严格 JSON：
{
  "suggestions": [
    "要限制具体城市吗？比如旧金山、纽约、西雅图？",
    "要指定公司规模吗？比如50人以下的创业公司？"
  ],
  "explanation": "结果较多，建议从XX维度缩小范围"
}
"""


def relax_params(original_query: str, current_sql: str, result_count: int, scenario: str = "recruiting") -> dict:
    """Relax PDL SQL query when results are too few."""
    client = get_anthropic_client()

    user_content = json.dumps({
        "original_query": original_query,
        "current_sql": current_sql,
        "result_count": result_count,
        "scenario": scenario,
    }, ensure_ascii=False, indent=2)

    response = client.messages.create(
        model=PARSE_MODEL,
        max_tokens=2048,
        system=system_prompt(RELAX_PROMPT),
        messages=[{"role": "user", "content": user_content}],
    )
    return extract_json(response.content[0].text)


def suggest_narrowing(original_query: str, current_sql: str, result_count: int, scenario: str = "recruiting") -> dict:
    """Suggest ways to narrow down when results are too many."""
    client = get_anthropic_client()

    user_content = json.dumps({
        "original_query": original_query,
        "current_sql": current_sql,
        "result_count": result_count,
        "scenario": scenario,
    }, ensure_ascii=False, indent=2)

    response = client.messages.create(
        model=PARSE_MODEL,
        max_tokens=1024,
        system=system_prompt(NARROW_PROMPT),
        messages=[{"role": "user", "content": user_content}],
    )
    return extract_json(response.content[0].text)
