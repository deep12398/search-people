"""FastAPI web server for the people search frontend."""

import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.query_parser import parse_query
from src.pdl_client import search_people, enrich_person
from src.result_processor import score_and_summarize
from src.param_refiner import relax_params, suggest_narrowing

app = FastAPI(title="People Search")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

STATIC_DIR = Path(__file__).parent.parent / "static"


class SearchRequest(BaseModel):
    query: str
    size: int = 10


class EnrichRequest(BaseModel):
    linkedin_url: str = ""
    email: str = ""
    name: str = ""
    company: str = ""


@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.post("/api/search")
async def api_search(req: SearchRequest):
    # Step 1: Parse query → scenario + SQL
    parsed = parse_query(req.query)
    scenario = parsed.get("scenario", "recruiting")
    sql = parsed.get("sql_query", "")
    description = parsed.get("description", "")

    # Step 2: Search PDL
    results = await search_people({"sql_query": sql, "size": req.size})

    # Handle API errors (quota, auth, etc.)
    if results.get("error"):
        return {
            "scenario": scenario,
            "sql": sql,
            "description": description,
            "total": 0,
            "people": [],
            "error": results["error"],
        }

    total = results.get("total_entries", 0)
    people = results.get("people", [])

    # Step 3: Handle edge cases
    if total == 0:
        relaxed = relax_params(req.query, sql, 0, scenario)
        relaxed_sql = relaxed.get("relaxed_sql", sql)
        results = await search_people({"sql_query": relaxed_sql, "size": req.size})
        total = results.get("total_entries", 0)
        people = results.get("people", [])
        return {
            "scenario": scenario,
            "sql": relaxed_sql,
            "description": description,
            "total": total,
            "people": people,
            "relaxed": True,
            "relax_info": relaxed,
        }

    if total > 1000:
        narrowing = suggest_narrowing(req.query, sql, total, scenario)
        # Still return results, but with suggestions
        scored = score_and_summarize(req.query, people, threshold=4, scenario=scenario)
        return {
            "scenario": scenario,
            "sql": sql,
            "description": description,
            "total": total,
            "people": scored.get("results", people),
            "summary": scored.get("summary", ""),
            "narrowing": narrowing,
        }

    # Step 4: Score and filter
    scored = score_and_summarize(req.query, people, threshold=4, scenario=scenario)
    return {
        "scenario": scenario,
        "sql": sql,
        "description": description,
        "total": total,
        "people": scored.get("results", people),
        "summary": scored.get("summary", ""),
    }


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


def main():
    import uvicorn
    uvicorn.run("src.web:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
