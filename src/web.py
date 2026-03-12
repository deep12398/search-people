"""FastAPI web server for the people search frontend."""

import json
import re
import uuid
import time
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, AssistantMessage, TextBlock
from src.tools import create_tools_server
from src.pdl_client import enrich_person
from src.auth import get_current_user, require_auth
from src.config import SUPABASE_URL, SUPABASE_ANON_KEY

app = FastAPI(title="People Search")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

STATIC_DIR = Path(__file__).parent.parent / "static"

# ─── Agent system prompt (web version) ───────────────────────────────────────
# 与 main.py 的 CLI 版本相比，多了一个约定：有结果时输出 JSON 代码块
SYSTEM_PROMPT = """\
你是一个专业的人脉搜索助手，使用 People Data Labs (PDL) 作为数据源。用中文与用户交流。

## 支持的场景
| 场景 | 典型需求 |
|------|----------|
| **recruiting** | 找候选人、技术人才 |
| **marketing** | 找目标受众、营销对象 |
| **kol** | 找博主、意见领袖 |
| **sales** | 找决策人、BD对象 |

## 工作流程

### 第一步：深挖需求（重要！）
用户描述往往很模糊。**在搜索之前，先追问一个最关键的问题**，帮助明确：
- 使用场景（招聘/营销/KOL/销售）
- 行业、技术栈或职能方向
- 地域或公司规模
- 职级或经验要求

满足以下 2 条以上才发起搜索：场景明确、行业/技能明确、地域/规模明确、职级明确。
如果用户拒绝回答或已经明确说"直接搜"，则立即搜索。

### 第二步：解析 + 搜索
调用 parse_search_query 生成 PDL SQL，再调用 pdl_search 搜索。
根据结果数量：
- **0 结果**：调用 auto_relax_params 放宽条件重试
- **>1000 结果**：调用 suggest_narrowing 生成追问
- **5~1000 条**：调用 score_and_filter_results 评分筛选

### 第三步：输出结果（关键格式要求）
有搜索结果时，**必须**在回复末尾输出以下格式的 JSON 代码块：

```json
{"type":"results","scenario":"recruiting","sql":"SELECT ...","total":1234,"people":[...],"summary":"..."}
```

people 数组中每个对象保留：name, title, company, location, company_industry, company_size, linkedin_url, score, reason, has_email, has_phone

如果追问用户（没有结果），**不要**输出 JSON，只输出纯文本问题。

## 规则
- pdl_enrich 消耗 1 credit/次，使用前提醒用户
- 搜索参数中的职位、地点、行业用英文小写
- 场景识别是自动的，用户明确说了场景以用户为准
"""

# ─── Session store ────────────────────────────────────────────────────────────
# session_id → {client: ClaudeSDKClient, created_at: float}
_sessions: dict[str, dict] = {}
SESSION_TTL = 3600  # 1 hour


def _cleanup_sessions():
    """Remove expired sessions."""
    now = time.time()
    expired = [sid for sid, s in _sessions.items() if now - s["created_at"] > SESSION_TTL]
    for sid in expired:
        _sessions.pop(sid, None)


async def _get_or_create_session(session_id: str | None) -> tuple[str, ClaudeSDKClient]:
    """Get existing or create new Agent SDK session."""
    _cleanup_sessions()

    if session_id and session_id in _sessions:
        return session_id, _sessions[session_id]["client"]

    # Create new session
    sid = session_id or str(uuid.uuid4())
    server = create_tools_server()
    options = ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        mcp_servers={"people-search": server},
        max_turns=30,
        permission_mode="bypassPermissions",
    )
    client = ClaudeSDKClient(options=options)
    await client.connect()
    _sessions[sid] = {"client": client, "created_at": time.time()}
    return sid, client


def _parse_agent_response(text: str) -> dict:
    """
    Parse agent response text.
    If it contains a ```json ... ``` block with type=results, extract it.
    Otherwise treat as a clarifying question.
    """
    # Look for ```json ... ``` block
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            if data.get("type") == "results":
                return data
        except json.JSONDecodeError:
            pass

    # No results block — it's a question or status message
    # Strip any partial/malformed JSON from the text
    clean_text = re.sub(r"```json.*?```", "", text, flags=re.DOTALL).strip()
    return {"type": "question", "question": clean_text or text}


# ─── Request / Response models ────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: str = ""  # empty = new session


class EnrichRequest(BaseModel):
    linkedin_url: str = ""
    email: str = ""
    name: str = ""
    company: str = ""


class FavoriteRequest(BaseModel):
    person: dict
    linkedin_url: str = ""


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/config")
async def api_config():
    """Public config for frontend Supabase initialization."""
    return {
        "supabase_url": SUPABASE_URL,
        "supabase_anon_key": SUPABASE_ANON_KEY,
    }


@app.post("/api/chat")
async def api_chat(req: ChatRequest, request: Request):
    """Stateful chat endpoint using Agent SDK. Handles clarification + search."""
    try:
        sid, client = await _get_or_create_session(req.session_id or None)
    except Exception as e:
        raise HTTPException(500, f"Failed to create agent session: {e}")

    # Send message to agent
    await client.query(req.message)

    # Collect response
    response_text = ""
    async for message in client.receive_response():
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    response_text += block.text

    # Parse: question or results
    parsed = _parse_agent_response(response_text)
    parsed["session_id"] = sid

    # Auto-save search history if user is authenticated and we got results
    if SUPABASE_URL and parsed.get("type") == "results":
        auth_user = await get_current_user(request)
        if auth_user:
            try:
                from src.supabase_client import save_search_history, save_search_results
                history_row = await save_search_history(
                    user_id=auth_user.id,
                    query=req.message,
                    scenario=parsed.get("scenario", "auto"),
                    result_count=parsed.get("total", 0),
                    access_token=auth_user.token,
                )
                if history_row.get("id"):
                    await save_search_results(
                        search_id=history_row["id"],
                        people=parsed.get("people", []),
                        sql_used=parsed.get("sql", ""),
                        summary=parsed.get("summary", ""),
                        access_token=auth_user.token,
                    )
            except Exception:
                pass  # Don't fail the search if DB save fails

    return parsed


@app.delete("/api/chat/{session_id}")
async def delete_session(session_id: str):
    """Close and remove a session (call on Home/reset)."""
    if session_id in _sessions:
        try:
            await _sessions[session_id]["client"].disconnect()
        except Exception:
            pass
        _sessions.pop(session_id, None)
    return {"ok": True}


@app.post("/api/enrich")
async def api_enrich(req: EnrichRequest):
    params = {}
    if req.linkedin_url:
        params["linkedin_url"] = req.linkedin_url
    elif req.email:
        params["email"] = req.email
    elif req.name and req.company:
        params["name"] = req.name
        params["company"] = req.company
    else:
        raise HTTPException(400, "Need linkedin_url, email, or name+company")

    result = await enrich_person(params)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@app.get("/login", response_class=HTMLResponse)
async def login_page():
    login_file = STATIC_DIR / "login.html"
    if not login_file.exists():
        raise HTTPException(404, "Login page not found")
    return login_file.read_text(encoding="utf-8")


# ─── Supabase-backed endpoints (require auth) ────────────────────────────────

@app.get("/api/history")
async def api_get_history(request: Request):
    user = await require_auth(request)
    from src.supabase_client import get_search_history
    return await get_search_history(user.id, access_token=user.token)


@app.get("/api/history/{history_id}/results")
async def api_get_history_results(history_id: str, request: Request):
    user = await require_auth(request)
    from src.supabase_client import get_search_results
    result = await get_search_results(history_id, access_token=user.token)
    if not result:
        raise HTTPException(404, "Results not found")
    return result


@app.delete("/api/history/{history_id}")
async def api_delete_history(history_id: str, request: Request):
    user = await require_auth(request)
    from src.supabase_client import delete_search_history
    await delete_search_history(user.id, history_id, access_token=user.token)
    return {"ok": True}


@app.get("/api/favorites")
async def api_get_favorites(request: Request):
    user = await require_auth(request)
    from src.supabase_client import get_favorites
    return await get_favorites(user.id, access_token=user.token)


@app.post("/api/favorites")
async def api_add_favorite(req: FavoriteRequest, request: Request):
    user = await require_auth(request)
    from src.supabase_client import add_favorite
    return await add_favorite(user.id, req.person, access_token=user.token)


@app.delete("/api/favorites")
async def api_remove_favorite(request: Request, linkedin_url: str = ""):
    user = await require_auth(request)
    if not linkedin_url:
        raise HTTPException(400, "linkedin_url query param required")
    from src.supabase_client import remove_favorite
    await remove_favorite(user.id, linkedin_url, access_token=user.token)
    return {"ok": True}


@app.get("/api/me")
async def api_me(request: Request):
    """Get current user info (used by frontend to check auth status)."""
    user = await get_current_user(request)
    if not user:
        return {"authenticated": False}
    return {"authenticated": True, "user_id": user.id}


def main():
    import uvicorn
    uvicorn.run("src.web:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
