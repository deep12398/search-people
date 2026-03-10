"""LLM-powered result filtering, scoring, and summarization."""

import json
from src.config import PARSE_MODEL, get_anthropic_client, extract_json, system_prompt

SCORE_AND_SUMMARIZE_PROMPT = """\
你是一个人脉搜索结果分析师。根据搜索场景和用户意图，智能评分和筛选。

## 场景评分权重

**recruiting（招聘）**：
- 技能匹配度（40%）：技术栈、专业技能是否符合
- 职级匹配度（25%）：资历层级是否符合要求
- 地点匹配度（20%）：是否在目标城市/区域
- 公司背景（15%）：公司规模、行业是否相关

**marketing（营销获客）**：
- 目标画像匹配（35%）：职位角色是否属于目标受众
- 行业匹配度（25%）：所在行业是否是目标市场
- 公司规模匹配（20%）：公司大小是否在目标范围
- 决策影响力（20%）：职级是否有采购/决策权

**kol（KOL/红人营销）**：
- 内容领域匹配（40%）：职位/兴趣是否与目标领域相关
- 影响力潜力（30%）：职位含 creator/blogger/influencer 等加分
- 行业契合度（20%）：所在行业是否与推广产品匹配
- 地域匹配（10%）：是否在目标市场区域

**sales（商务/销售）**：
- 决策权（35%）：是否是 CXO/VP/Director/Owner 等决策层
- 行业匹配（25%）：是否在目标行业
- 公司规模（20%）：公司是否在目标客户规模范围
- 职能匹配（20%）：是否在采购/运营/业务等相关职能

## 任务
1. 根据场景权重给每个人打相关性分数（1-10），10 表示完美匹配
2. 为每个人写一句中文推荐理由（20字以内），突出该场景最关键的匹配点
3. 按分数从高到低排序
4. 过滤掉分数低于 threshold 的人

严格输出 JSON（不要其他内容）：
{
  "results": [
    {
      "id": "原始ID",
      "name": "姓名",
      "title": "职位",
      "company": "公司",
      "score": 8,
      "reason": "场景相关的推荐理由",
      "has_email": true,
      "has_phone": true
    }
  ],
  "filtered_count": 3,
  "summary": "共找到25人，筛选后保留10人，主要集中在XX领域"
}
"""


def score_and_summarize(user_query: str, people: list[dict], threshold: int = 4, scenario: str = "recruiting") -> dict:
    """Score and summarize search results based on user intent and scenario."""
    client = get_anthropic_client()

    user_content = (
        f"搜索场景：{scenario}\n"
        f"用户搜索意图：{user_query}\n\n"
        f"筛选阈值：{threshold}（低于此分数的过滤掉）\n\n"
        f"搜索结果（{len(people)}人）：\n"
        f"{json.dumps(people, ensure_ascii=False, indent=2)}"
    )

    response = client.messages.create(
        model=PARSE_MODEL,
        max_tokens=4096,
        system=system_prompt(SCORE_AND_SUMMARIZE_PROMPT),
        messages=[{"role": "user", "content": user_content}],
    )
    return extract_json(response.content[0].text)
