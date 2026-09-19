import os
import sys
import json

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath("."))

from src.state import AgentState
from src.tools.financial_data import fetch_financial_metrics
from src.tools.executor import execute_backtest_code
from src.graph import should_retry_edge, route_intent_edge, build_alpha_graph

def test_financial_audit():
    print("[TEST 1/5] Testing deterministic financial audit tool on AAPL...")
    data = fetch_financial_metrics("AAPL")
    assert data["symbol"] == "AAPL"
    assert "valuation" in data
    assert "altman_z_score" in data
    assert data["altman_z_score"]["score"] is not None
    print(f"  -> Passed! Company: {data['company_name']}, Z-Score: {data['altman_z_score']['score']} ({data['altman_z_score']['status']})")

def test_code_executor_success():
    print("[TEST 2/5] Testing code execution sandbox with simulated backtest...")
    sample_code = """
import json
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.plot([1, 2, 3, 4], [100, 108, 105, 120], label='Strategy')
plt.title("Sample Test Equity Curve")
plt.legend()
plt.savefig("outputs/equity_curve.png")
plt.close()

metrics = {
    "ticker": "TEST",
    "total_strategy_return": "20.0%",
    "sharpe_ratio": 1.45,
    "max_drawdown": "-4.2%"
}
print("===METRICS_JSON_START===")
print(json.dumps(metrics))
print("===METRICS_JSON_END===")
"""
    res = execute_backtest_code(sample_code)
    assert res["success"] is True
    assert res["metrics"]["total_strategy_return"] == "20.0%"
    assert res["chart_path"] is not None
    assert os.path.exists(res["chart_path"])
    print(f"  -> Passed! Execution returned metrics and generated chart at: {res['chart_path']}")

def test_code_executor_error_handling():
    print("[TEST 3/5] Testing error handling and traceback capture...")
    buggy_code = """
import pandas as pd
df = pd.DataFrame({'a': [1, 2, 3]})
# Intentional bug to test error catching
print(df['non_existent_column'])
"""
    res = execute_backtest_code(buggy_code)
    assert res["success"] is False
    assert "KeyError" in res["error"]
    print("  -> Passed! Correctly caught KeyError in sandbox without crashing.")

def test_graph_routing_and_edges():
    print("[TEST 4/5] Testing LangGraph conditional edges and self-healing loop routing...")
    # 1. Intent routing
    health_state = AgentState(intent="financial_health", query="health of Boeing", ticker="BA", code="", execution_output=None, error_log=None, retry_count=0, max_retries=3, metrics=None, chart_path=None, final_report="", messages=[])
    assert route_intent_edge(health_state) == "financial_health"
    
    backtest_state = AgentState(intent="backtest", query="backtest SMA", ticker="NVDA", code="", execution_output=None, error_log=None, retry_count=0, max_retries=3, metrics=None, chart_path=None, final_report="", messages=[])
    assert route_intent_edge(backtest_state) == "generate_code"
    
    # 2. Self-healing routing: error present, retries remaining -> routes to fix_code
    error_state = AgentState(intent="backtest", query="", ticker="NVDA", code="", execution_output=None, error_log="KeyError: 'Close'", retry_count=1, max_retries=3, metrics=None, chart_path=None, final_report="", messages=[])
    assert should_retry_edge(error_state) == "fix_code"
    
    # 3. Max retries reached -> routes to synthesize_report
    maxed_state = AgentState(intent="backtest", query="", ticker="NVDA", code="", execution_output=None, error_log="KeyError: 'Close'", retry_count=3, max_retries=3, metrics=None, chart_path=None, final_report="", messages=[])
    assert should_retry_edge(maxed_state) == "synthesize_report"
    
    # 4. Success -> routes to synthesize_report
    success_state = AgentState(intent="backtest", query="", ticker="NVDA", code="", execution_output={"success": True}, error_log=None, retry_count=1, max_retries=3, metrics={"return": "10%"}, chart_path=None, final_report="", messages=[])
    assert should_retry_edge(success_state) == "synthesize_report"
    
    print("  -> Passed! All routing decisions and self-healing edges verified.")

def test_graph_compilation():
    print("[TEST 5/5] Testing LangGraph state graph compilation...")
    graph = build_alpha_graph()
    assert graph is not None
    print("  -> Passed! LangGraph workflow graph compiled successfully.")

if __name__ == "__main__":
    print("==========================================")
    print("RUNNING ALPHA-AGENT VERIFICATION SUITE")
    print("==========================================\n")
    test_financial_audit()
    test_code_executor_success()
    test_code_executor_error_handling()
    test_graph_routing_and_edges()
    test_graph_compilation()
    print("\n==========================================")
    print("ALL 5 VERIFICATION TESTS PASSED (5/5)!")
    print("==========================================")
