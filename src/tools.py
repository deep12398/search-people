"""Custom MCP tools for the people search agent (PDL-powered)."""

import json
from claude_agent_sdk import tool, create_sdk_mcp_server
from src.query_parser import parse_query
from src.pdl_client import search_people, enrich_person
from src.result_processor import score_and_summarize
from src.param_refiner import relax_params, suggest_narrowing


def _text(content: str) -> dict:
    return {"content": [{"type": "text", "text": content}]}


@tool(
    "parse_search_query",
    "Parse a natural language people search query into a PDL SQL query. "
    "Input: the user's search description in any language. "
    "Output: PDL SQL query and parameters.",
    {"query": str},
)
async def parse_search_query_tool(args: dict) -> dict:
    query = args["query"]
    try:
        params = parse_query(query)
        return _text(json.dumps(params, ensure_ascii=False, indent=2))
    except (json.JSONDecodeError, IndexError) as e:
        return _text(f"Error parsing query: {e}")


@tool(
    "pdl_search",
    "Search for people using People Data Labs API with a SQL query. "
    "Free tier: 100 lookups/month. Returns name, title, company, location, LinkedIn. "
    "Input: sql_query (PDL SQL string), size (results count, default 10).",
    {"sql_query": str, "size": int},
)
async def pdl_search_tool(args: dict) -> dict:
    params = {
        "sql_query": args["sql_query"],
        "size": args.get("size", 10),
    }
    try:
        results = await search_people(params)
        total = results.get("total_entries", 0)
        people = results.get("people", [])

        output = {
            "total_entries": total,
            "returned_count": len(people),
            "people": people,
        }
        return _text(json.dumps(output, ensure_ascii=False, indent=2))
    except Exception as e:
        return _text(f"PDL search error: {e}")


@tool(
    "score_and_filter_results",
    "Use LLM to score, filter, and summarize search results based on user intent and scenario. "
    "Scores each person 1-10, generates a Chinese recommendation, "
    "filters out low-relevance results, and sorts by score. "
    "Input: user_query (original search intent), people_json (JSON array), "
    "threshold (minimum score, default 4), "
    "scenario (one of: recruiting, marketing, kol, sales; default recruiting).",
    {"user_query": str, "people_json": str, "threshold": int, "scenario": str},
)
async def score_and_filter_tool(args: dict) -> dict:
    try:
        people = json.loads(args["people_json"])
    except json.JSONDecodeError as e:
        return _text(f"Invalid people JSON: {e}")

    threshold = args.get("threshold", 4)
    scenario = args.get("scenario", "recruiting")
    try:
        result = score_and_summarize(args["user_query"], people, threshold, scenario)
        return _text(json.dumps(result, ensure_ascii=False, indent=2))
    except (json.JSONDecodeError, IndexError) as e:
        return _text(f"Scoring error: {e}")


@tool(
    "auto_relax_params",
    "When a search returns 0 or very few results, automatically relax the PDL SQL query. "
    "Input: original_query (user's description), sql_query (current SQL), "
    "result_count (how many results returned), "
    "scenario (one of: recruiting, marketing, kol, sales; default recruiting).",
    {"original_query": str, "sql_query": str, "result_count": int, "scenario": str},
)
async def auto_relax_params_tool(args: dict) -> dict:
    try:
        scenario = args.get("scenario", "recruiting")
        result = relax_params(args["original_query"], args["sql_query"], args["result_count"], scenario)
        return _text(json.dumps(result, ensure_ascii=False, indent=2))
    except (json.JSONDecodeError, IndexError) as e:
        return _text(f"Relax error: {e}")


@tool(
    "suggest_narrowing",
    "When a search returns too many results (>1000), generate follow-up questions "
    "to help the user narrow down. "
    "Input: original_query (user's description), sql_query (current SQL), "
    "result_count (how many results returned), "
    "scenario (one of: recruiting, marketing, kol, sales; default recruiting).",
    {"original_query": str, "sql_query": str, "result_count": int, "scenario": str},
)
async def suggest_narrowing_tool(args: dict) -> dict:
    try:
        scenario = args.get("scenario", "recruiting")
        result = suggest_narrowing(args["original_query"], args["sql_query"], args["result_count"], scenario)
        return _text(json.dumps(result, ensure_ascii=False, indent=2))
    except (json.JSONDecodeError, IndexError) as e:
        return _text(f"Narrowing error: {e}")


@tool(
    "pdl_enrich",
    "Get full profile for a specific person via PDL enrichment. "
    "Consumes 1 credit per successful lookup. "
    "Input: provide ONE of: linkedin_url, email, or both name AND company.",
    {"linkedin_url": str, "email": str, "name": str, "company": str},
)
async def pdl_enrich_tool(args: dict) -> dict:
    params = {k: v for k, v in args.items() if v}
    if not params:
        return _text("Need at least linkedin_url, email, or name+company.")

    try:
        result = await enrich_person(params)
        if "error" in result:
            return _text(f"Enrich error: {result['error']}")
        return _text(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as e:
        return _text(f"PDL enrich error: {e}")


def create_tools_server():
    """Create the MCP server with all people search tools."""
    return create_sdk_mcp_server(
        "people-search-tools",
        tools=[
            parse_search_query_tool,
            pdl_search_tool,
            score_and_filter_tool,
            auto_relax_params_tool,
            suggest_narrowing_tool,
            pdl_enrich_tool,
        ],
    )
