# search-people — CLAUDE.md

## 启动
```bash
python3 -m uvicorn src.web:app --host 0.0.0.0 --port 8000 --reload
```

## 架构
- Web 入口: `src/web.py` — FastAPI，`/api/chat`（Agent SDK 多轮对话）+ `/api/enrich`
- CLI 入口: `src/main.py` — ClaudeSDKClient 交互式终端
- 工具层: `src/tools.py` — 6 个 MCP tools（parse/search/score/relax/narrow/enrich）
- 前端: `static/index.html` — 单页应用，Demo 模式内置 mock 数据（不需要 API 额度）

## 关键注意事项
- Anthropic proxy (`ANTHROPIC_BASE_URL`): system prompt 必须用 array format，见 `config.py:system_prompt()`
- Agent SDK: 需设 `permission_mode="bypassPermissions"`，否则 agent 等待工具授权挂起
- Agent SDK: `ClaudeSDKClient` 可跨 HTTP 请求复用，同一 asyncio event loop 即可
- PDL API: 404 = 无结果（正常），402 = 额度用完，两者均已在 `pdl_client.py` 中捕获
- PDL 免费版: `location_name` 返回 `True/False` 而非字符串，需 fallback 到 `job_company_location_name`
- PDL 免费版: 不支持 `"from"` 分页参数（已删除），用 `scroll_token` 代替

## 模型
`PARSE_MODEL = "claude-sonnet-4-20250514"`（proxy 仅支持此 model ID）

## 测试
```bash
python3 test_all.py   # 7 个集成测试

# 测试多轮对话第一轮（应返回追问，不直接搜索）
curl -X POST localhost:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"找Python工程师"}'
```
