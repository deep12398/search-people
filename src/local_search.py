"""Local people search using PostgreSQL full-text search on Supabase people table."""

import json
import socket
import sys
import psycopg2
from psycopg2.extras import RealDictCursor
from src.config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)

# Cache resolved IPv4 address
_ipv4_cache: dict[str, str] = {}


def _resolve_ipv4(host: str) -> str:
    """Resolve hostname to IPv4, using DNS-over-HTTPS as fallback."""
    if host in _ipv4_cache:
        return _ipv4_cache[host]

    # Method 1: local socket (works if OS DNS returns A records)
    try:
        results = socket.getaddrinfo(host, None, socket.AF_INET)
        if results:
            ip = results[0][4][0]
            _log(f"[ipv4] socket OK: {host} -> {ip}")
            _ipv4_cache[host] = ip
            return ip
    except socket.gaierror as e:
        _log(f"[ipv4] socket failed: {e}")

    # Method 2: DNS-over-HTTPS via httpx (handles SSL properly in slim containers)
    try:
        import httpx
        for dns_url in [
            f"https://cloudflare-dns.com/dns-query?name={host}&type=A",
            f"https://dns.google/resolve?name={host}&type=A",
        ]:
            try:
                resp = httpx.get(dns_url, headers={"Accept": "application/dns-json"}, timeout=5)
                data = resp.json()
                for answer in data.get("Answer", []):
                    if answer.get("type") == 1:  # A record
                        ip = answer["data"]
                        _log(f"[ipv4] DoH OK: {host} -> {ip}")
                        _ipv4_cache[host] = ip
                        return ip
            except Exception as e:
                _log(f"[ipv4] DoH {dns_url.split('/')[2]} failed: {e}")
    except ImportError:
        _log("[ipv4] httpx not available")

    # Method 3: raw UDP DNS query to 8.8.8.8 (no HTTPS/TLS needed)
    try:
        import struct
        qname = b""
        for part in host.split("."):
            qname += struct.pack("B", len(part)) + part.encode()
        qname += b"\x00"
        # DNS header: ID=0x1234, flags=0x0100 (standard query), 1 question
        header = struct.pack(">HHHHHH", 0x1234, 0x0100, 1, 0, 0, 0)
        question = qname + struct.pack(">HH", 1, 1)  # type A, class IN
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(3)
        sock.sendto(header + question, ("8.8.8.8", 53))
        resp_data = sock.recv(512)
        sock.close()
        # Parse answer: skip header (12 bytes) + question section
        pos = 12
        # Skip question name
        while resp_data[pos] != 0:
            pos += resp_data[pos] + 1
        pos += 5  # null byte + type(2) + class(2)
        # Parse answers
        answer_count = struct.unpack(">H", resp_data[6:8])[0]
        for _ in range(answer_count):
            # Skip name (pointer or labels)
            if resp_data[pos] & 0xC0 == 0xC0:
                pos += 2
            else:
                while resp_data[pos] != 0:
                    pos += resp_data[pos] + 1
                pos += 1
            rtype, rclass, ttl, rdlength = struct.unpack(">HHIH", resp_data[pos:pos+10])
            pos += 10
            if rtype == 1 and rdlength == 4:  # A record
                ip = socket.inet_ntoa(resp_data[pos:pos+4])
                _log(f"[ipv4] UDP DNS OK: {host} -> {ip}")
                _ipv4_cache[host] = ip
                return ip
            pos += rdlength
    except Exception as e:
        _log(f"[ipv4] UDP DNS failed: {e}")

    _log(f"[ipv4] All methods failed for {host}")
    return host


def _get_conn():
    """Create a new database connection, forcing IPv4."""
    ipv4 = _resolve_ipv4(DB_HOST) if DB_HOST else None
    # Use hostaddr to bypass psycopg2's own DNS resolution
    # Keep host for SSL SNI verification
    if ipv4 and ipv4 != DB_HOST:
        return psycopg2.connect(
            host=DB_HOST, hostaddr=ipv4, port=DB_PORT, dbname=DB_NAME,
            user=DB_USER, password=DB_PASSWORD,
            sslmode="require",
        )
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
        user=DB_USER, password=DB_PASSWORD,
        sslmode="require",
    )


async def search_local(
    keywords: str,
    filters: dict | None = None,
    page: int = 0,
    size: int = 10,
) -> dict:
    """
    PostgreSQL full-text search on people table.

    Args:
        keywords: Search terms (English, extracted by LLM from user query)
        filters: Optional {seniority, country, industry, has_email}
        page: Page number (0-based), OFFSET = page * size
        size: Results per page

    Returns:
        {total, people, has_more, page}
    """
    filters = filters or {}

    # Build WHERE clauses
    conditions = []
    params = []

    if keywords.strip():
        # Convert keywords to tsquery: "python engineer" → "python & engineer"
        terms = keywords.strip().split()
        tsquery = " & ".join(terms)
        conditions.append(
            "to_tsvector('english', "
            "COALESCE(title,'') || ' ' || COALESCE(company,'') || ' ' || "
            "COALESCE(keywords,'') || ' ' || COALESCE(industry,'') || ' ' || "
            "COALESCE(first_name,'') || ' ' || COALESCE(last_name,'')) "
            "@@ to_tsquery('english', %s)"
        )
        params.append(tsquery)

    if filters.get("seniority"):
        conditions.append("LOWER(seniority) = LOWER(%s)")
        params.append(filters["seniority"])

    if filters.get("country"):
        conditions.append("LOWER(country) = LOWER(%s)")
        params.append(filters["country"])

    if filters.get("industry"):
        conditions.append("LOWER(industry) LIKE LOWER(%s)")
        params.append(f"%{filters['industry']}%")

    if filters.get("has_email"):
        conditions.append("email IS NOT NULL AND email != ''")

    where_clause = " AND ".join(conditions) if conditions else "TRUE"

    conn = _get_conn()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Count total matches
        cur.execute(f"SELECT COUNT(*) as cnt FROM people WHERE {where_clause}", params)
        total = cur.fetchone()["cnt"]

        # Fetch page
        offset = page * size
        query = f"""
            SELECT first_name, last_name, title, company, email, email_status,
                   seniority, departments, phone, linkedin_url, city, state,
                   country, industry, keywords, company_size, source, enriched_at
            FROM people
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s
        """
        cur.execute(query, params + [size, offset])
        rows = cur.fetchall()

        # Format people for frontend
        people = []
        for r in rows:
            name = f"{r['first_name'] or ''} {r['last_name'] or ''}".strip()
            location_parts = [p for p in [r["city"], r["state"], r["country"]] if p]
            people.append({
                "name": name,
                "title": r["title"],
                "company": r["company"],
                "location": ", ".join(location_parts),
                "company_industry": r["industry"],
                "company_size": r["company_size"],
                "linkedin_url": r["linkedin_url"],
                "has_email": bool(r["email"]),
                "has_phone": bool(r["phone"]),
                "source": r["source"],
                "enriched": r["enriched_at"] is not None,
            })

        cur.close()
        return {
            "total": total,
            "people": people,
            "has_more": (offset + size) < total,
            "page": page,
        }
    finally:
        conn.close()


async def save_pdl_to_people(people_list: list, source: str = "pdl_search") -> int:
    """
    Save PDL search/enrich results into the people table.

    Uses ON CONFLICT(linkedin_url) to upsert:
    - pdl_search: insert new rows, skip existing
    - pdl_enrich: update existing rows with detailed fields

    Returns: number of rows inserted/updated.
    """
    if not people_list:
        return 0

    conn = _get_conn()
    try:
        cur = conn.cursor()
        count = 0

        for p in people_list:
            linkedin_url = p.get("linkedin_url") or p.get("profiles", [{}])[0].get("url") if isinstance(p.get("profiles"), list) else p.get("linkedin_url")
            if not linkedin_url:
                continue

            if source == "pdl_enrich":
                # Enrich: update existing row with detailed fields
                emails = p.get("emails", [])
                phones = p.get("phone_numbers", [])
                work_email = next((e.get("address") for e in emails if e.get("type") == "professional"), None) if emails else None
                personal_email = next((e.get("address") for e in emails if e.get("type") == "personal"), None) if emails else None
                mobile_phone = phones[0].get("number") if phones else None

                cur.execute("""
                    UPDATE people SET
                        work_email = COALESCE(%s, work_email),
                        personal_email = COALESCE(%s, personal_email),
                        mobile_phone = COALESCE(%s, mobile_phone),
                        education = COALESCE(%s, education),
                        experience = COALESCE(%s, experience),
                        skills = COALESCE(%s, skills),
                        summary = COALESCE(%s, summary),
                        profile_pic_url = COALESCE(%s, profile_pic_url),
                        source = 'pdl_enrich',
                        enriched_at = now(),
                        updated_at = now()
                    WHERE linkedin_url = %s
                """, (
                    work_email,
                    personal_email,
                    mobile_phone,
                    json.dumps(p.get("education", []), ensure_ascii=False) if p.get("education") else None,
                    json.dumps(p.get("experience", []), ensure_ascii=False) if p.get("experience") else None,
                    ", ".join(p.get("skills", [])) if p.get("skills") else None,
                    p.get("summary"),
                    p.get("profile_pic_url"),
                    linkedin_url,
                ))
                if cur.rowcount > 0:
                    count += cur.rowcount
                else:
                    # Row doesn't exist yet, insert it
                    _insert_pdl_person(cur, p, linkedin_url, "pdl_enrich")
                    count += 1
            else:
                # pdl_search: insert new, skip existing
                _insert_pdl_person(cur, p, linkedin_url, source)
                count += 1 if cur.rowcount > 0 else 0

        conn.commit()
        cur.close()
        return count
    finally:
        conn.close()


def _insert_pdl_person(cur, p: dict, linkedin_url: str, source: str):
    """Insert a single PDL person record."""
    # Extract location
    location_name = p.get("location_name")
    if isinstance(location_name, bool):
        location_name = None

    city = p.get("location_locality") or ""
    state = p.get("location_region") or ""
    country = p.get("location_country") or ""

    # Extract company info
    company = p.get("job_company_name") or ""
    title = p.get("job_title") or ""
    industry = p.get("industry") or p.get("job_company_industry") or ""
    company_size = p.get("job_company_size") or ""

    cur.execute("""
        INSERT INTO people (
            first_name, last_name, title, company, linkedin_url,
            city, state, country, industry, company_size, source
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (linkedin_url) DO NOTHING
    """, (
        p.get("first_name"),
        p.get("last_name"),
        title,
        company,
        linkedin_url,
        city,
        state,
        country,
        industry,
        company_size,
        source,
    ))
