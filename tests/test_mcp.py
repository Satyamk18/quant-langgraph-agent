import os
import sys
import json
import asyncio

sys.path.insert(0, os.path.abspath("."))

from src.mcp_server import server

async def test_mcp_tool_listing():
    print("[TEST 1/4] Testing MCP tool discovery and schemas...")
    tools = await server.list_tools()
    tool_names = [t.name for t in tools]
    assert "get_financial_metrics" in tool_names, "Missing get_financial_metrics tool"
    assert "search_sec_filings" in tool_names, "Missing search_sec_filings tool"
    assert "run_backtest_sandbox" in tool_names, "Missing run_backtest_sandbox tool"
    print(f"  -> Passed! Found {len(tools)} registered MCP tools: {tool_names}")

async def test_mcp_financial_metrics():
    print("[TEST 2/4] Testing get_financial_metrics tool via MCP...")
    res = await server.call_tool("get_financial_metrics", {"ticker": "BA"})
    assert res is not None
    # Extract text from content
    text = res.content[0].text
    data = json.loads(text)
    assert data["symbol"] == "BA"
    assert "altman_z_score" in data
    assert "solvency_and_liquidity" in data
    print(f"  -> Passed! Ticker: {data['symbol']}, Altman Z-Score: {data['altman_z_score']['score']}")

async def test_mcp_sec_filings_rag():
    print("[TEST 3/4] Testing search_sec_filings RAG tool via MCP...")
    res = await server.call_tool("search_sec_filings", {"query": "debt obligations", "ticker": "BA"})
    assert res is not None
    text = res.content[0].text
    data = json.loads(text)
    assert "sources" in data
    assert len(data["sources"]) > 0
    first_chunk = data["sources"][0]
    assert "content" in first_chunk
    assert "boeing" in first_chunk["source"].lower()
    print(f"  -> Passed! Retrieved {len(data['sources'])} filing chunks via MCP. First source: {first_chunk['source']}")

async def test_mcp_backtest_sandbox():
    print("[TEST 4/4] Testing run_backtest_sandbox tool via MCP...")
    dummy_code = """
import json
metrics = {
    "ticker": "MCP_TEST",
    "total_strategy_return": "18.5%",
    "sharpe_ratio": 1.55,
    "max_drawdown": "-6.1%"
}
print("===METRICS_JSON_START===")
print(json.dumps(metrics))
print("===METRICS_JSON_END===")
"""
    res = await server.call_tool("run_backtest_sandbox", {"code": dummy_code})
    assert res is not None
    text = res.content[0].text
    data = json.loads(text)
    assert data["success"] is True
    assert data["metrics"]["total_strategy_return"] == "18.5%"
    print(f"  -> Passed! Sandbox executed via MCP. Strategy Return: {data['metrics']['total_strategy_return']}")

async def main():
    print("==========================================")
    print("RUNNING ALPHA-AGENT MCP PROTOCOL TEST SUITE")
    print("==========================================\n")
    await test_mcp_tool_listing()
    await test_mcp_financial_metrics()
    await test_mcp_sec_filings_rag()
    await test_mcp_backtest_sandbox()
    print("\n==========================================")
    print("ALL 4 MCP PROTOCOL TESTS PASSED (4/4)!")
    print("==========================================")

if __name__ == "__main__":
    asyncio.run(main())
