"""Apollo API client for people search and enrichment."""

import httpx
from src.config import APOLLO_API_KEY, APOLLO_BASE_URL


def _headers() -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Cache-Control": "no-cache",
        "x-api-key": APOLLO_API_KEY,
    }


async def search_people(params: dict) -> dict:
    """Search people via Apollo API. Does not consume credits.

    Requires Apollo Basic plan or above (free tier returns 403).
    """
    apollo_params = {
        "person_titles": params.get("person_titles", []),
        "include_similar_titles": params.get("include_similar_titles", True),
        "person_seniorities": params.get("person_seniorities", []),
        "person_locations": params.get("person_locations", []),
        "q_organization_domains_list": params.get("organization_domains", []),
        "organization_locations": params.get("organization_locations", []),
        "organization_num_employees_ranges": params.get("employee_ranges", []),
        "q_organization_keyword_tags": params.get("keywords", []),
        "per_page": params.get("per_page", 25),
        "page": params.get("page", 1),
    }
    apollo_params = {k: v for k, v in apollo_params.items() if v}

    async with httpx.AsyncClient(timeout=30) as http:
        resp = await http.post(
            f"{APOLLO_BASE_URL}/mixed_people/api_search",
            headers=_headers(),
            json=apollo_params,
        )
        resp.raise_for_status()
        return resp.json()


async def enrich_people(person_ids: list[str]) -> list[dict]:
    """Enrich people by IDs via bulk_match. Consumes credits (max 10 per call).

    Requires Apollo Basic plan or above (free tier returns 403).
    """
    if not person_ids:
        return []

    async with httpx.AsyncClient(timeout=30) as http:
        resp = await http.post(
            f"{APOLLO_BASE_URL}/people/bulk_match",
            headers=_headers(),
            json={
                "details": [{"id": pid} for pid in person_ids[:10]],
                "reveal_personal_emails": True,
                "reveal_phone_number": True,
            },
        )
        resp.raise_for_status()
        return resp.json().get("matches", [])
