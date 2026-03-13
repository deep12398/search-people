"""People Search Agent - CLI entry point."""

import anyio
from claude_agent_sdk import AssistantMessage, ClaudeSDKClient, TextBlock
from src.agent_runtime import build_system_prompt, create_agent_options

SYSTEM_PROMPT = build_system_prompt(include_json_results=False)


async def run_agent():
    guard, options = create_agent_options(
        include_json_results=False,
        max_turns=20,
        system_prompt=SYSTEM_PROMPT,
    )

    async with ClaudeSDKClient(options=options) as client:
        print("🔍 人脉搜索助手已启动！(本地优先，PDL 兜底)")
        print("描述你想找的人，例如：'帮我找硅谷做 AI 的创业公司的 CTO'")
        print("输入 'quit' 退出\n")

        while True:
            user_input = input("你: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("再见！")
                break

            guard.start_turn()
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
