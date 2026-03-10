"""People Data Labs API client for people search and enrichment."""

import httpx
from src.config import PDL_API_KEY, PDL_BASE_URL


def _headers() -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "X-Api-Key": PDL_API_KEY,
    }


async def search_people(params: dict) -> dict:
    """Search people via PDL Person Search API using SQL query.

    Free tier: 100 lookups/month (no contact data).
    Paid tier: includes emails, phones, social profiles.

    params should contain:
        - sql_query: str (SQL WHERE clause)
        - size: int (results per page, max 100)
        - from_: int (offset for pagination)
    """
    body = {
        "sql": params["sql_query"],
        "size": params.get("size", 10),
    }
    if params.get("scroll_token"):
        body["scroll_token"] = params["scroll_token"]

    async with httpx.AsyncClient(timeout=30) as http:
        resp = await http.post(
            f"{PDL_BASE_URL}/person/search",
            headers=_headers(),
            json=body,
        )
        data = resp.json()

        # PDL returns 404 when no results found — not a real error
        if resp.status_code == 404:
            return {"total_entries": 0, "people": [], "scroll_token": None}

        # Handle billing/quota errors gracefully
        if resp.status_code == 402:
            return {"total_entries": 0, "people": [], "scroll_token": None,
                    "error": "PDL API quota exceeded (402). Please check your plan or billing."}

        if resp.status_code == 401:
            return {"total_entries": 0, "people": [], "scroll_token": None,
                    "error": "PDL API key invalid (401). Please check your API key."}

        resp.raise_for_status()

        # Normalize response to a common format
        people = []
        for p in data.get("data", []):
            people.append({
                "id": p.get("id", ""),
                "name": p.get("full_name", ""),
                "first_name": p.get("first_name", ""),
                "last_name": p.get("last_name", ""),
                "title": p.get("job_title", ""),
                "company": p.get("job_company_name", ""),
                "company_domain": p.get("job_company_website", ""),
                "company_size": p.get("job_company_size", ""),
                "company_industry": p.get("job_company_industry", ""),
                "location": (
                    p.get("location_name") if isinstance(p.get("location_name"), str)
                    else p.get("job_company_location_name", "")
                ),
                "country": (
                    p.get("location_country") if isinstance(p.get("location_country"), str)
                    else p.get("job_company_location_country", "")
                ),
                "linkedin_url": p.get("linkedin_url", ""),
                "email": p.get("work_email", p.get("recommended_personal_email", "")),
                "phone": p.get("mobile_phone", ""),
                "has_email": bool(p.get("work_email") or p.get("recommended_personal_email")),
                "has_phone": bool(p.get("mobile_phone")),
            })

        return {
            "total_entries": data.get("total", 0),
            "people": people,
            "scroll_token": data.get("scroll_token"),
        }


async def enrich_person(params: dict) -> dict:
    """Enrich a single person by name+company or LinkedIn URL.

    params can contain:
        - linkedin_url: str
        - name: str + company: str
        - email: str
    """
    query_params = {}
    if params.get("linkedin_url"):
        query_params["profile"] = params["linkedin_url"]
    elif params.get("email"):
        query_params["email"] = params["email"]
    elif params.get("name") and params.get("company"):
        query_params["name"] = params["name"]
        query_params["company"] = params["company"]
    else:
        return {"error": "Need linkedin_url, email, or name+company"}

    async with httpx.AsyncClient(timeout=30) as http:
        resp = await http.get(
            f"{PDL_BASE_URL}/person/enrich",
            headers=_headers(),
            params=query_params,
        )
        resp.raise_for_status()
        data = resp.json()

        p = data.get("data", data)
        return {
            "name": p.get("full_name", ""),
            "title": p.get("job_title", ""),
            "company": p.get("job_company_name", ""),
            "email": p.get("work_email", ""),
            "personal_emails": p.get("personal_emails", []),
            "phone": p.get("mobile_phone", ""),
            "linkedin_url": p.get("linkedin_url", ""),
            "github_url": p.get("github_url", ""),
            "twitter_url": p.get("twitter_url", ""),
            "location": p.get("location_name", ""),
            "experience": [
                {
                    "title": exp.get("title", {}).get("name", ""),
                    "company": exp.get("company", {}).get("name", ""),
                    "start": exp.get("start_date"),
                    "end": exp.get("end_date"),
                    "is_primary": exp.get("is_primary", False),
                }
                for exp in (p.get("experience", []) or [])[:5]
            ],
            "education": [
                {
                    "school": edu.get("school", {}).get("name", ""),
                    "degree": edu.get("degrees", []),
                    "majors": edu.get("majors", []),
                }
                for edu in (p.get("education", []) or [])[:3]
            ],
            "skills": (p.get("skills", []) or [])[:10],
        }
