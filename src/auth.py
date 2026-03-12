"""FastAPI authentication middleware using Supabase JWT tokens."""

from dataclasses import dataclass
from fastapi import Request, HTTPException
from supabase import create_client
from src.config import SUPABASE_URL, SUPABASE_ANON_KEY


@dataclass
class AuthUser:
    id: str
    token: str


async def get_current_user(request: Request) -> AuthUser | None:
    """Extract and verify Supabase JWT from Authorization header.

    Returns AuthUser(id, token) if authenticated, None if no token.
    Raises 401 if token is invalid/expired.
    """
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return None

    token = auth_header[7:]
    if not token:
        return None

    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        return None

    try:
        sb = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        user_response = sb.auth.get_user(token)
        if user_response and user_response.user:
            return AuthUser(id=user_response.user.id, token=token)
        raise HTTPException(status_code=401, detail="Invalid token")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Auth failed: {e}")


async def require_auth(request: Request) -> AuthUser:
    """Same as get_current_user but raises 401 if not authenticated."""
    user = await get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user
