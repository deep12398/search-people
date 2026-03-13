import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
APOLLO_API_KEY = os.environ.get("APOLLO_API_KEY", "")
APOLLO_BASE_URL = "https://api.apollo.io/api/v1"

PDL_API_KEY = os.environ.get("PDL_API_KEY", "")
PDL_BASE_URL = "https://api.peopledatalabs.com/v5"

# Supabase
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
# Direct DB connection params (for people table)
DB_HOST = os.environ.get("DB_HOST", "")
DB_PORT = int(os.environ.get("DB_PORT", "5432"))
DB_NAME = os.environ.get("DB_NAME", "postgres")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")

# LLM model for query parsing (cheaper model for structured extraction)
PARSE_MODEL = "claude-sonnet-4-20250514"


def get_anthropic_client():
    """Create an Anthropic client with custom base URL support."""
    import anthropic
    return anthropic.Anthropic(
        api_key=ANTHROPIC_API_KEY,
        base_url=ANTHROPIC_BASE_URL,
    )


def system_prompt(text: str) -> list[dict]:
    """Format system prompt as array for proxy compatibility."""
    return [{"type": "text", "text": text}]


def extract_json(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks."""
    import json
    import re
    # Strip ```json ... ``` wrapper if present
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
    if match:
        text = match.group(1)
    return json.loads(text.strip())
