"""FastAPI web server for the people search frontend."""

import json
import re
import uuid
import time
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from claude_agent_sdk import AssistantMessage, ClaudeSDKClient, TextBlock
from src.agent_runtime import SearchFlowGuard, build_system_prompt, create_agent_options
from src.pdl_client import enrich_person
from src.auth import get_current_user, require_auth
from src.config import SUPABASE_URL, SUPABASE_ANON_KEY

app = FastAPI(title="People Search")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

STATIC_DIR = Path(__file__).parent.parent / "static"

SYSTEM_PROMPT = build_system_prompt(include_json_results=True)

# ─── Session store ────────────────────────────────────────────────────────────
# session_id → {client: ClaudeSDKClient, guard: SearchFlowGuard, created_at: float}
_sessions: dict[str, dict] = {}
SESSION_TTL = 3600  # 1 hour


def _cleanup_sessions():
    """Remove expired sessions."""
    now = time.time()
    expired = [sid for sid, s in _sessions.items() if now - s["created_at"] > SESSION_TTL]
    for sid in expired:
        _sessions.pop(sid, None)


async def _get_or_create_session(session_id: str | None) -> tuple[str, ClaudeSDKClient, SearchFlowGuard]:
    """Get existing or create new Agent SDK session."""
    _cleanup_sessions()

    if session_id and session_id in _sessions:
        session = _sessions[session_id]
        return session_id, session["client"], session["guard"]

    # Create new session
    sid = session_id or str(uuid.uuid4())
    guard, options = create_agent_options(
        include_json_results=True,
        max_turns=30,
        system_prompt=SYSTEM_PROMPT,
    )
    client = ClaudeSDKClient(options=options)
    await client.connect()
    _sessions[sid] = {"client": client, "guard": guard, "created_at": time.time()}
    return sid, client, guard


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


@app.get("/api/health")
async def api_health():
    """Health check: test DB connection."""
    try:
        from src.local_search import search_local
        result = await search_local("engineer", page=0, size=1)
        return {"db": "ok", "total": result["total"]}
    except Exception as e:
        return {"db": "error", "message": str(e)[:300]}


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
        sid, client, guard = await _get_or_create_session(req.session_id or None)
    except Exception as e:
        raise HTTPException(500, f"Failed to create agent session: {e}")

    # Send message to agent
    guard.start_turn()
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

    # Auto-save search history for every conversation turn (not just results)
    if SUPABASE_URL:
        auth_user = await get_current_user(request)
        if auth_user:
            try:
                from src.supabase_client import save_search_history, save_search_results
                result_count = parsed.get("total", 0) if parsed.get("type") == "results" else 0
                history_row = await save_search_history(
                    user_id=auth_user.id,
                    query=req.message,
                    scenario=parsed.get("scenario", "auto"),
                    result_count=result_count,
                    access_token=auth_user.token,
                )
                if history_row.get("id") and parsed.get("type") == "results":
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


@app.post("/api/chat/stream")
async def api_chat_stream(req: ChatRequest, request: Request):
    """Streaming chat endpoint using SSE. Sends text chunks as they arrive."""
    try:
        sid, client, guard = await _get_or_create_session(req.session_id or None)
    except Exception as e:
        raise HTTPException(500, f"Failed to create agent session: {e}")

    auth_user = await get_current_user(request) if SUPABASE_URL else None

    async def event_stream():
        # Send session_id immediately
        yield f"data: {json.dumps({'type': 'session', 'session_id': sid})}\n\n"

        try:
            guard.start_turn()
            await client.query(req.message)

            response_text = ""
            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            response_text += block.text
                            yield f"data: {json.dumps({'type': 'chunk', 'text': block.text})}\n\n"

            # Send final parsed result
            parsed = _parse_agent_response(response_text)
            parsed["session_id"] = sid
            # Ensure 'type' is always 'done' for the final SSE event
            # (parsed may contain type='question' or type='results')
            parsed["result_type"] = parsed.pop("type", "question")
            parsed["type"] = "done"
            yield f"data: {json.dumps(parsed)}\n\n"

            # Save history
            if auth_user:
                try:
                    from src.supabase_client import save_search_history, save_search_results
                    result_count = parsed.get("total", 0) if parsed.get("type") == "results" else 0
                    history_row = await save_search_history(
                        user_id=auth_user.id,
                        query=req.message,
                        scenario=parsed.get("scenario", "auto"),
                        result_count=result_count,
                        access_token=auth_user.token,
                    )
                    if history_row.get("id") and parsed.get("type") == "results":
                        await save_search_results(
                            search_id=history_row["id"],
                            people=parsed.get("people", []),
                            sql_used=parsed.get("sql", ""),
                            summary=parsed.get("summary", ""),
                            access_token=auth_user.token,
                        )
                except Exception:
                    pass
        except Exception as e:
            error_data = {'type': 'done', 'question': f'Agent error: {e}', 'session_id': sid}
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


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
