import os
import sys

sys.path.insert(0, os.path.abspath("."))

from src.rag.vectorstore import ingest_filings, query_filings
from src.state import AgentState
from src.graph import route_intent_edge, build_alpha_graph

def test_rag_ingestion():
    print("[TEST 1/4] Testing filing ingestion into ChromaDB...")
    num_chunks = ingest_filings()
    assert num_chunks > 0, "No chunks were ingested."
    print(f"  -> Passed! Ingested {num_chunks} semantic chunks.")

def test_rag_retrieval_ticker_filter():
    print("[TEST 2/4] Testing vector search with ticker filtering (BA vs AAPL)...")
    ba_results = query_filings("debt obligations and liquidity", ticker="BA", top_k=2)
    assert len(ba_results) > 0, "No results for Boeing."
    for r in ba_results:
        assert r["ticker"] == "BA"
        assert "boeing" in r["source"].lower()
    print("  -> Passed! Boeing search correctly filtered to Boeing documents.")

    aapl_results = query_filings("supply chain and TSMC chip manufacturing", ticker="AAPL", top_k=2)
    assert len(aapl_results) > 0, "No results for Apple."
    for r in aapl_results:
        assert r["ticker"] == "AAPL"
        assert "apple" in r["source"].lower()
    print("  -> Passed! Apple search correctly filtered to Apple documents.")

def test_rag_citations_structure():
    print("[TEST 3/4] Testing RAG citation metadata formatting...")
    results = query_filings("FAA oversight", ticker="BA", top_k=1)
    chunk = results[0]
    assert "source" in chunk
    assert "ticker" in chunk
    assert "content" in chunk
    assert "score" in chunk
    print(f"  -> Passed! Citation metadata present: Source={chunk['source']}, Score={chunk['score']}")

def test_rag_graph_routing():
    print("[TEST 4/4] Testing LangGraph 3-way conditional routing...")
    rag_state = AgentState(intent="filing_qa", query="What are Boeing 10-K risks?", ticker="BA", code="", execution_output=None, error_log=None, retry_count=0, max_retries=3, metrics=None, chart_path=None, sources=None, final_report="", messages=[])
    assert route_intent_edge(rag_state) == "filing_rag"

    health_state = AgentState(intent="financial_health", query="health of Boeing", ticker="BA", code="", execution_output=None, error_log=None, retry_count=0, max_retries=3, metrics=None, chart_path=None, sources=None, final_report="", messages=[])
    assert route_intent_edge(health_state) == "financial_health"

    backtest_state = AgentState(intent="backtest", query="backtest SMA", ticker="NVDA", code="", execution_output=None, error_log=None, retry_count=0, max_retries=3, metrics=None, chart_path=None, sources=None, final_report="", messages=[])
    assert route_intent_edge(backtest_state) == "generate_code"

    graph = build_alpha_graph()
    assert graph is not None
    print("  -> Passed! LangGraph 3-way routing edge validated.")

if __name__ == "__main__":
    print("==========================================")
    print("RUNNING ALPHA-AGENT RAG VERIFICATION SUITE")
    print("==========================================\n")
    test_rag_ingestion()
    test_rag_retrieval_ticker_filter()
    test_rag_citations_structure()
    test_rag_graph_routing()
    print("\n==========================================")
    print("ALL 4 RAG TESTS PASSED (4/4)!")
    print("==========================================")
