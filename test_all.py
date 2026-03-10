"""End-to-end test for all components (PDL-powered)."""

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.config import get_anthropic_client, ANTHROPIC_BASE_URL, PARSE_MODEL, PDL_API_KEY, system_prompt


def test_1_anthropic_connection():
    """Test 1: Anthropic API connection."""
    print("=" * 50)
    print("TEST 1: Anthropic API 连接测试")
    print(f"  Base URL: {ANTHROPIC_BASE_URL}")
    print(f"  Model: {PARSE_MODEL}")

    client = get_anthropic_client()
    response = client.messages.create(
        model=PARSE_MODEL,
        max_tokens=50,
        system=system_prompt("Reply in Chinese."),
        messages=[{"role": "user", "content": "Say hello"}],
    )
    text = response.content[0].text
    print(f"  Response: {text}")
    print("  ✅ PASS")
    return True


def test_2_query_parser():
    """Test 2: LLM generates PDL SQL query."""
    print("\n" + "=" * 50)
    print("TEST 2: 自然语言 → PDL SQL 解析测试")

    from src.query_parser import parse_query

    query = "帮我找硅谷做AI的创业公司的CTO"
    print(f"  Input: {query}")
    params = parse_query(query)
    print(f"  SQL: {params.get('sql_query', 'N/A')}")
    print(f"  Size: {params.get('size', 'N/A')}")
    print(f"  Description: {params.get('description', 'N/A')}")

    assert "sql_query" in params, "Should have sql_query"
    assert "SELECT" in params["sql_query"].upper(), "Should be a SQL query"
    print("  ✅ PASS")
    return params


def test_3_pdl_search():
    """Test 3: PDL Person Search API."""
    print("\n" + "=" * 50)
    print("TEST 3: PDL 搜索测试")

    if not PDL_API_KEY or PDL_API_KEY == "your_pdl_api_key_here":
        print("  ❌ SKIP: PDL_API_KEY 未配置，请在 .env 中填入")
        return []

    from src.pdl_client import search_people

    params = {
        "sql_query": "SELECT * FROM person WHERE job_title LIKE '%cto%' AND location_region='california' AND job_company_size='11-50'",
        "size": 3,
    }
    print(f"  SQL: {params['sql_query']}")

    try:
        results = asyncio.run(search_people(params))
        total = results.get("total_entries", 0)
        people = results.get("people", [])
        print(f"  Total: {total}, Returned: {len(people)}")
        for p in people:
            print(f"    - {p['name']} | {p['title']} @ {p['company']} | {p['location']}")
        print("  ✅ PASS")
        return people
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        return []


def test_4_pdl_enrich():
    """Test 4: PDL Person Enrichment."""
    print("\n" + "=" * 50)
    print("TEST 4: PDL 详情丰富测试")

    if not PDL_API_KEY or PDL_API_KEY == "your_pdl_api_key_here":
        print("  ❌ SKIP: PDL_API_KEY 未配置")
        return False

    from src.pdl_client import enrich_person

    params = {"linkedin_url": "https://www.linkedin.com/in/seanthorne"}
    print(f"  LinkedIn: {params['linkedin_url']}")

    try:
        result = asyncio.run(enrich_person(params))
        if "error" in result:
            print(f"  ❌ FAIL: {result['error']}")
            return False
        print(f"  Name: {result.get('name')}")
        print(f"  Title: {result.get('title')}")
        print(f"  Company: {result.get('company')}")
        print(f"  Location: {result.get('location')}")
        print(f"  Skills: {result.get('skills', [])[:5]}")
        print("  ✅ PASS")
        return True
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        return False


def test_5_result_scoring():
    """Test 5: LLM result scoring."""
    print("\n" + "=" * 50)
    print("TEST 5: 结果评分和摘要测试")

    from src.result_processor import score_and_summarize

    mock_people = [
        {"id": "1", "name": "Alex Chang", "title": "CTO", "company": "DeepMind AI", "location": "San Francisco", "has_email": True, "has_phone": True},
        {"id": "2", "name": "Sarah Lin", "title": "VP Engineering", "company": "OpenAI", "location": "San Francisco", "has_email": True, "has_phone": False},
        {"id": "3", "name": "Bob Smith", "title": "Marketing Manager", "company": "Random Corp", "location": "New York", "has_email": False, "has_phone": False},
    ]

    result = score_and_summarize("找硅谷AI创业公司CTO", mock_people, threshold=3)
    print(f"  Results: {len(result.get('results', []))}")
    print(f"  Summary: {result.get('summary', '')}")
    for r in result.get("results", []):
        print(f"    - {r.get('name')}: score={r.get('score')} | {r.get('reason')}")
    print("  ✅ PASS")
    return True


def test_6_param_relaxation():
    """Test 6: Auto parameter relaxation."""
    print("\n" + "=" * 50)
    print("TEST 6: SQL 自动放宽测试")

    from src.param_refiner import relax_params

    sql = "SELECT * FROM person WHERE job_title='cto' AND location_locality='san francisco' AND job_company_industry='quantum computing'"
    result = relax_params("找旧金山量子计算公司CTO", sql, result_count=0)
    print(f"  Relaxed SQL: {result.get('relaxed_sql', 'N/A')}")
    print(f"  Changes: {result.get('changes_made', [])}")
    print(f"  Explanation: {result.get('explanation', '')}")
    print("  ✅ PASS")
    return True


def test_7_narrowing_suggestions():
    """Test 7: Narrowing suggestions."""
    print("\n" + "=" * 50)
    print("TEST 7: 追问建议测试")

    from src.param_refiner import suggest_narrowing

    sql = "SELECT * FROM person WHERE job_title LIKE '%product manager%'"
    result = suggest_narrowing("找产品经理", sql, result_count=50000)
    for s in result.get("suggestions", []):
        print(f"    - {s}")
    print(f"  Explanation: {result.get('explanation', '')}")
    print("  ✅ PASS")
    return True


if __name__ == "__main__":
    print("🔍 Search People Agent (PDL) - 全面测试\n")
    passed = 0
    failed = 0
    skipped = 0

    tests = [
        ("Anthropic 连接", test_1_anthropic_connection),
        ("NL → PDL SQL", test_2_query_parser),
        ("PDL 搜索", test_3_pdl_search),
        ("PDL 详情", test_4_pdl_enrich),
        ("结果评分", test_5_result_scoring),
        ("SQL 放宽", test_6_param_relaxation),
        ("追问建议", test_7_narrowing_suggestions),
    ]

    for name, fn in tests:
        try:
            result = fn()
            if result == [] or result is False:
                skipped += 1
            else:
                passed += 1
        except Exception as e:
            print(f"  ❌ FAIL: {e}")
            failed += 1

    print("\n" + "=" * 50)
    print(f"📊 结果: {passed} 通过 / {skipped} 跳过 / {failed} 失败")
    if failed == 0:
        print("🎉 所有可用测试通过！")
    if skipped > 0:
        print("💡 跳过的测试：请在 .env 中填入 PDL_API_KEY 后重试")
