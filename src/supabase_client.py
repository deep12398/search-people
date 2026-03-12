"""Supabase database operations for search history, results, and favorites.

Uses user JWT tokens with RLS (no service_role key needed).
Each function accepts an access_token to authenticate as the user.
"""

from supabase import create_client, Client
from src.config import SUPABASE_URL, SUPABASE_ANON_KEY, SUPABASE_SERVICE_KEY


def get_supabase(access_token: str = "") -> Client:
    """Get Supabase client.

    If service_role key is set, use it (bypasses RLS).
    Otherwise, use anon key + user's access_token (respects RLS).
    """
    if not SUPABASE_URL:
        raise RuntimeError("SUPABASE_URL must be set")

    if SUPABASE_SERVICE_KEY:
        return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

    sb = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    if access_token:
        sb.postgrest.auth(access_token)
    return sb


async def save_search_history(user_id: str, query: str, scenario: str = "auto",
                              result_count: int = 0, access_token: str = "") -> dict:
    """Insert a search history record. Returns the inserted row."""
    sb = get_supabase(access_token)
    result = sb.table("search_history").insert({
        "user_id": user_id,
        "query": query,
        "scenario": scenario,
        "result_count": result_count,
    }).execute()
    return result.data[0] if result.data else {}


async def save_search_results(search_id: str, people: list, sql_used: str = "",
                              summary: str = "", access_token: str = "") -> dict:
    """Insert search results snapshot linked to a search_history record."""
    sb = get_supabase(access_token)
    result = sb.table("search_results").insert({
        "search_id": search_id,
        "people": people,
        "sql_used": sql_used,
        "summary": summary,
    }).execute()
    return result.data[0] if result.data else {}


async def get_search_history(user_id: str, limit: int = 20, access_token: str = "") -> list:
    """Get recent search history for a user."""
    sb = get_supabase(access_token)
    result = sb.table("search_history") \
        .select("*") \
        .eq("user_id", user_id) \
        .order("created_at", desc=True) \
        .limit(limit) \
        .execute()
    return result.data or []


async def get_search_results(search_id: str, access_token: str = "") -> dict | None:
    """Get search results for a specific search history entry."""
    sb = get_supabase(access_token)
    result = sb.table("search_results") \
        .select("*") \
        .eq("search_id", search_id) \
        .limit(1) \
        .execute()
    return result.data[0] if result.data else None


async def delete_search_history(user_id: str, history_id: str, access_token: str = "") -> bool:
    """Delete a single search history entry (cascade deletes results)."""
    sb = get_supabase(access_token)
    result = sb.table("search_history") \
        .delete() \
        .eq("id", history_id) \
        .eq("user_id", user_id) \
        .execute()
    return bool(result.data)


async def add_favorite(user_id: str, person: dict, access_token: str = "") -> dict:
    """Add a person to favorites. Uses linkedin_url for dedup."""
    sb = get_supabase(access_token)
    row = {
        "user_id": user_id,
        "person": person,
        "linkedin_url": person.get("linkedin_url", ""),
    }
    result = sb.table("favorites").upsert(row, on_conflict="user_id,linkedin_url").execute()
    return result.data[0] if result.data else {}


async def remove_favorite(user_id: str, linkedin_url: str, access_token: str = "") -> bool:
    """Remove a person from favorites by linkedin_url."""
    sb = get_supabase(access_token)
    result = sb.table("favorites") \
        .delete() \
        .eq("user_id", user_id) \
        .eq("linkedin_url", linkedin_url) \
        .execute()
    return bool(result.data)


async def get_favorites(user_id: str, access_token: str = "") -> list:
    """Get all favorites for a user."""
    sb = get_supabase(access_token)
    result = sb.table("favorites") \
        .select("*") \
        .eq("user_id", user_id) \
        .order("created_at", desc=True) \
        .execute()
    return result.data or []
