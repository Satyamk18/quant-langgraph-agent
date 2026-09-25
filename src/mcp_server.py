"""
AlphaAgent MCP Server.
Exposes quantitative financial tools, SEC 10-K RAG retrieval, and sandboxed
backtesting over the Model Context Protocol (MCP) using the official `mcp` SDK.
"""

import os
import sys
import json
from typing import Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mcp.server.mcpserver import MCPServer
from src.tools.financial_data import fetch_financial_metrics
from src.tools.executor import execute_backtest_code
from src.rag.vectorstore import query_filings

# Initialize MCP Server
server = MCPServer(
    name="AlphaAgent-Financial-MCP-Server",
    version="1.0.0",
    instructions="Financial Intelligence MCP Server providing quantitative ratios, SEC 10-K RAG, and backtesting."
)

@server.tool(
    name="get_financial_metrics",
    description="Fetches live corporate balance sheet data, valuation metrics, and calculates the Altman Z-Score for bankruptcy risk analysis using yfinance."
)
def get_financial_metrics(ticker: str) -> str:
    """
    Args:
        ticker: The stock ticker symbol in uppercase (e.g. 'BA', 'AAPL', 'NVDA', 'MSFT').
    Returns:
        JSON string containing valuation, margins, solvency/liquidity, and Altman Z-score.
    """
    try:
        data = fetch_financial_metrics(ticker)
        return json.dumps(data, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch financial metrics for {ticker}: {str(e)}"})

@server.tool(
    name="search_sec_filings",
    description="Performs semantic vector search across official SEC Form 10-K annual reports in ChromaDB to retrieve qualitative risk factors, supply chain dependencies, and management footnotes with citations."
)
def search_sec_filings(query: str, ticker: Optional[str] = None) -> str:
    """
    Args:
        query: The qualitative research question or topic (e.g., 'debt covenant risks and liquidity constraints').
        ticker: Optional stock ticker filter (e.g., 'BA', 'AAPL').
    Returns:
        JSON string containing the top relevant excerpts, source filenames, and similarity scores.
    """
    try:
        results = query_filings(query=query, ticker=ticker, top_k=4)
        if not results:
            return json.dumps({"message": f"No filing excerpts found for query: '{query}' with ticker: {ticker}"})
        return json.dumps({"query": query, "ticker": ticker, "sources": results}, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to search SEC filings: {str(e)}"})

@server.tool(
    name="run_backtest_sandbox",
    description="Executes a quantitative trading strategy script in an isolated subprocess sandbox. Captures stdout/stderr, extracts performance metrics (Sharpe ratio, max drawdown, total return), and saves the equity curve plot."
)
def run_backtest_sandbox(code: str) -> str:
    """
    Args:
        code: Standalone Python backtest script using pandas, numpy, and yfinance.
    Returns:
        JSON string with execution status, parsed metrics, error logs (if any), and chart path.
    """
    try:
        res = execute_backtest_code(code)
        return json.dumps(res, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Sandbox execution runner error: {str(e)}"})

def main():
    """Runs the MCP server via standard I/O transport."""
    # When run directly, start stdio async server
    import asyncio
    asyncio.run(server.run_stdio_async())

if __name__ == "__main__":
    main()
