"""People Search Agent - main entry point (PDL-powered)."""

import anyio
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, AssistantMessage, TextBlock
from src.tools import create_tools_server

SYSTEM_PROMPT = """\
你是一个专业的人脉搜索助手，使用 People Data Labs (PDL) 作为数据源。用中文与用户交流。

## 支持的场景

你能自动识别用户的搜索场景，并据此优化搜索和筛选策略：

| 场景 | 典型需求 | 展示侧重 |
|------|----------|----------|
| **recruiting**（招聘） | 找候选人、技术人才 | 技能、经验、学历 |
| **marketing**（营销获客） | 找目标受众画像 | 行业、公司规模、职能 |
| **kol**（KOL/红人营销） | 找博主、意见领袖 | 内容领域、影响力、社交链接 |
| **sales**（商务/销售） | 找决策人、BD对象 | 决策权、公司规模、行业 |

## 完整工作流程

### 第一步：解析 + 场景识别
用户描述想找的人 → 调用 parse_search_query 生成 PDL SQL 查询。
该工具会自动识别场景（recruiting/marketing/kol/sales）并返回 scenario 字段。
简要展示给用户确认，包括识别到的场景。

### 第二步：搜索
调用 pdl_search 执行搜索，根据结果数量走不同分支：

- **结果为 0 或极少（<5）**：
  调用 auto_relax_params 自动放宽 SQL 条件，告诉用户做了什么调整，重新搜索。
  最多自动重试 2 次。

- **结果过多（>1000）**：
  调用 suggest_narrowing 生成追问建议，向用户提问缩小范围。

- **结果适中（5~1000）**：进入第三步。

### 第三步：智能筛选
调用 score_and_filter_results，传入用户原始意图、搜索结果、以及 scenario 字段。
LLM 会根据场景使用不同评分权重，打分、写推荐理由、过滤低分结果、按相关性排序。

### 第四步：场景化展示
根据不同场景，侧重展示不同信息：

**recruiting**：姓名、职位、公司、技能关键词、地点、相关性分数和推荐理由
**marketing**：姓名、职位、公司、行业、公司规模、地点、匹配理由
**kol**：姓名、职位/头衔、内容领域、LinkedIn、社交链接、匹配理由
**sales**：姓名、职位（决策层级）、公司、行业、公司规模、匹配理由

所有场景都展示：LinkedIn 链接（如有）、是否有邮箱/电话可获取

### 第五步：按需获取详情
用户指定想看某人详细信息时，调用 pdl_enrich（需要 LinkedIn URL、邮箱、或姓名+公司）。

## 重要规则
- pdl_search 消耗搜索额度（免费版 100次/月），但比 enrichment 便宜
- pdl_enrich 消耗 1 credit/次，使用前提醒用户
- 免费版搜索结果不含邮箱和电话，需升级才有
- 搜索参数中的职位、地点、行业用英文小写
- 如果用户描述模糊，主动追问细化
- 场景识别是自动的，但如果用户明确说了场景，以用户为准
"""


async def run_agent():
    server = create_tools_server()
    options = ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        mcp_servers={"people-search": server},
        max_turns=20,
    )

    async with ClaudeSDKClient(options=options) as client:
        print("🔍 人脉搜索助手已启动！(PDL)")
        print("描述你想找的人，例如：'帮我找硅谷做 AI 的创业公司的 CTO'")
        print("输入 'quit' 退出\n")

        while True:
            user_input = input("你: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("再见！")
                break

            await client.query(user_input)
            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            print(f"\n助手: {block.text}\n")


def main():
    anyio.run(run_agent)


if __name__ == "__main__":
    main()
