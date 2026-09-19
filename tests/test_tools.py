import sys, os
sys.path.insert(0, os.path.abspath("."))
import json
from src.tools.financial_data import fetch_financial_metrics
from src.tools.executor import execute_backtest_code

def test_financial():
    res = fetch_financial_metrics("AAPL")
    assert res["symbol"] == "AAPL"
    assert "valuation" in res
    assert "altman_z_score" in res
    print("[PASS] Financial data test passed!")

def test_executor():
    dummy_code = """
import json
metrics = {
    "total_return": "15.2%",
    "sharpe_ratio": 1.45,
    "max_drawdown": "-8.4%"
}
print("===METRICS_JSON_START===")
print(json.dumps(metrics))
print("===METRICS_JSON_END===")
"""
    res = execute_backtest_code(dummy_code)
    assert res["success"] is True
    assert res["metrics"]["total_return"] == "15.2%"
    assert res["metrics"]["sharpe_ratio"] == 1.45
    print("[PASS] Executor test passed!")

if __name__ == "__main__":
    test_financial()
    test_executor()
