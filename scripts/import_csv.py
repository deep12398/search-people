"""One-time CSV import into Supabase people table.

Usage:
    python scripts/import_csv.py [path_to_csv]
    Default CSV: 领英数据9,978条.csv in project root.
"""

import sys
from pathlib import Path

# Add project root to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from src.config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

# ─── Schema ──────────────────────────────────────────────────────────────────

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS people (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  first_name TEXT,
  last_name TEXT,
  title TEXT,
  company TEXT,
  email TEXT,
  email_status TEXT,
  seniority TEXT,
  departments TEXT,
  phone TEXT,
  linkedin_url TEXT UNIQUE,
  city TEXT,
  state TEXT,
  country TEXT,
  industry TEXT,
  keywords TEXT,
  company_size TEXT,
  company_linkedin_url TEXT,
  website TEXT,
  annual_revenue BIGINT,
  -- enrich fields
  work_email TEXT,
  personal_email TEXT,
  mobile_phone TEXT,
  education TEXT,
  experience TEXT,
  skills TEXT,
  summary TEXT,
  profile_pic_url TEXT,
  -- metadata
  source TEXT DEFAULT 'csv',
  enriched_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_people_fts ON people
  USING gin(to_tsvector('english',
    COALESCE(title,'') || ' ' || COALESCE(company,'') || ' ' ||
    COALESCE(keywords,'') || ' ' || COALESCE(industry,'') || ' ' ||
    COALESCE(first_name,'') || ' ' || COALESCE(last_name,'')
  ));

CREATE INDEX IF NOT EXISTS idx_people_country ON people(country);
CREATE INDEX IF NOT EXISTS idx_people_seniority ON people(seniority);
CREATE INDEX IF NOT EXISTS idx_people_industry ON people(industry);
"""


# ─── Helpers ─────────────────────────────────────────────────────────────────

def employee_count_to_range(val) -> str | None:
    """Convert numeric employee count to range string."""
    if pd.isna(val):
        return None
    try:
        n = int(float(val))
    except (ValueError, TypeError):
        return str(val) if val else None

    if n <= 10:
        return "1-10"
    elif n <= 50:
        return "11-50"
    elif n <= 200:
        return "51-200"
    elif n <= 500:
        return "201-500"
    elif n <= 1000:
        return "501-1000"
    elif n <= 5000:
        return "1001-5000"
    elif n <= 10000:
        return "5001-10000"
    else:
        return "10001+"


def pick_phone(row) -> str | None:
    """Pick first non-empty phone from Work Direct > Mobile > Corporate."""
    for col in ["Work Direct Phone", "Mobile Phone", "Corporate Phone"]:
        val = row.get(col)
        if pd.notna(val) and str(val).strip():
            return str(val).strip()
    return None


def safe_int(val) -> int | None:
    """Convert revenue string to int, handling currency formats."""
    if pd.isna(val):
        return None
    try:
        # Remove $, commas, etc.
        cleaned = str(val).replace("$", "").replace(",", "").strip()
        return int(float(cleaned)) if cleaned else None
    except (ValueError, TypeError):
        return None


def safe_str(val) -> str | None:
    """Convert to string, returning None for NaN/empty."""
    if pd.isna(val):
        return None
    s = str(val).strip()
    return s if s else None


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else str(
        Path(__file__).parent.parent / "领英数据9,978条.csv"
    )

    print(f"Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path, dtype=str)
    print(f"  Rows: {len(df)}")

    # Connect to Supabase PostgreSQL
    print(f"Connecting to {DB_HOST}:{DB_PORT}/{DB_NAME}...")
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
        user=DB_USER, password=DB_PASSWORD,
        sslmode="require",
    )
    cur = conn.cursor()

    # Create table + indexes
    print("Creating people table...")
    cur.execute(CREATE_TABLE_SQL)
    conn.commit()

    # Prepare rows
    columns = [
        "first_name", "last_name", "title", "company", "email", "email_status",
        "seniority", "departments", "phone", "linkedin_url", "city", "state",
        "country", "industry", "keywords", "company_size", "company_linkedin_url",
        "website", "annual_revenue", "source",
    ]

    rows = []
    for _, r in df.iterrows():
        row = (
            safe_str(r.get("First Name")),
            safe_str(r.get("Last Name")),
            safe_str(r.get("Title")),
            safe_str(r.get("Company Name")),
            safe_str(r.get("Email")),
            safe_str(r.get("Email Status")),
            safe_str(r.get("Seniority")),
            safe_str(r.get("Departments")),
            pick_phone(r),
            safe_str(r.get("Person Linkedin Url")),
            safe_str(r.get("City")),
            safe_str(r.get("State")),
            safe_str(r.get("Country")),
            safe_str(r.get("Industry")),
            safe_str(r.get("Keywords")),
            employee_count_to_range(r.get("# Employees")),
            safe_str(r.get("Company Linkedin Url")),
            safe_str(r.get("Website")),
            safe_int(r.get("Annual Revenue")),
            "csv",
        )
        rows.append(row)

    # Batch insert
    print(f"Inserting {len(rows)} rows...")
    insert_sql = f"""
        INSERT INTO people ({', '.join(columns)})
        VALUES %s
        ON CONFLICT (linkedin_url) DO NOTHING
    """
    execute_values(cur, insert_sql, rows, page_size=500)
    conn.commit()

    # Verify
    cur.execute("SELECT COUNT(*) FROM people WHERE source = 'csv'")
    count = cur.fetchone()[0]
    print(f"Done! {count} CSV rows in people table.")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
