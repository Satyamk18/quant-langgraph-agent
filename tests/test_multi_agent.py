"""
AlphaAgent Hierarchical Multi-Agent & Adversarial Risk Officer Verification Test Suite.
Validates:
1. Maximum Drawdown Policy Veto (Drawdown > 15%)
2. Sharpe Ratio Hurdle Veto (Sharpe < 1.0)
3. Compliant Strategy Approval (Drawdown <= 15%, Sharpe >= 1.0)
4. Re-Hedging Budget Exhaustion & Capital Preservation Routing
5. Full Multi-Agent Graph Compilation & Node Registry
"""

import os
import sys

# Ensure root in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.state import AgentState
from src.graph import (
    adversarial_risk_officer_node,
    should_rehedge_edge,
    should_retry_edge,
    build_alpha_graph
)

def test_drawdown_policy_veto():
    print("[TEST 1/5] Testing Adversarial Risk Officer Drawdown Policy Veto...")
    state: AgentState = {
        "query": "High beta momentum strategy",
        "ticker": "NVDA",
        "intent": "backtest",
        "code": "# simulated code",
        "execution_output": {"success": True},
        "error_log": None,
        "retry_count": 0,
        "max_retries": 3,
        "metrics": {
            "ticker": "NVDA",
            "total_strategy_return": "42.5%",
            "sharpe_ratio": 1.45,
            "max_drawdown": "-24.8%",  # Policy violation: > 15%
            "total_trades": 18
        },
        "chart_path": "outputs/equity_curve.png",
        "sources": None,
        "risk_review": None,
        "risk_review_count": 0,
        "max_risk_reviews": 2,
        "final_report": "",
        "messages": []
    }
    
    update = adversarial_risk_officer_node(state)
    review = update["risk_review"]
    
    assert review["status"] == "VETOED_RETRY", f"Expected VETOED_RETRY, got {review['status']}"
    assert any("Drawdown" in v for v in review["violations"]), "Expected Drawdown violation in audit"
    assert review["mandate_directive"] is not None
    assert "stop-loss" in review["mandate_directive"].lower() or "drawdown" in review["mandate_directive"].lower()
    
    state.update(update)
    assert should_rehedge_edge(state) == "rehedge_code"
    print(f"  -> Passed! Vetoed with directive: {review['mandate_directive']}")

def test_sharpe_hurdle_veto():
    print("[TEST 2/5] Testing Adversarial Risk Officer Sharpe Ratio Hurdle Veto...")
    state: AgentState = {
        "query": "Noisy mean-reversion strategy",
        "ticker": "TSLA",
        "intent": "backtest",
        "code": "# simulated code",
        "execution_output": {"success": True},
        "error_log": None,
        "retry_count": 0,
        "max_retries": 3,
        "metrics": {
            "ticker": "TSLA",
            "total_strategy_return": "8.2%",
            "sharpe_ratio": 0.42,      # Policy violation: < 1.0
            "max_drawdown": "-9.5%",   # Compliant
            "total_trades": 35
        },
        "chart_path": "outputs/equity_curve.png",
        "sources": None,
        "risk_review": None,
        "risk_review_count": 0,
        "max_risk_reviews": 2,
        "final_report": "",
        "messages": []
    }
    
    update = adversarial_risk_officer_node(state)
    review = update["risk_review"]
    
    assert review["status"] == "VETOED_RETRY", f"Expected VETOED_RETRY, got {review['status']}"
    assert any("Sharpe" in v for v in review["violations"]), "Expected Sharpe hurdle violation in audit"
    print("  -> Passed! Correctly flagged Sharpe hurdle failure (< 1.00).")

def test_compliant_strategy_approval():
    print("[TEST 3/5] Testing Compliant Strategy Approval by Risk Officer...")
    state: AgentState = {
        "query": "Prudent trend-following strategy",
        "ticker": "MSFT",
        "intent": "backtest",
        "code": "# simulated code",
        "execution_output": {"success": True},
        "error_log": None,
        "retry_count": 0,
        "max_retries": 3,
        "metrics": {
            "ticker": "MSFT",
            "total_strategy_return": "28.4%",
            "sharpe_ratio": 1.72,      # Compliant: >= 1.0
            "max_drawdown": "-11.8%",  # Compliant: <= 15%
            "total_trades": 14
        },
        "chart_path": "outputs/equity_curve.png",
        "sources": None,
        "risk_review": None,
        "risk_review_count": 0,
        "max_risk_reviews": 2,
        "final_report": "",
        "messages": []
    }
    
    update = adversarial_risk_officer_node(state)
    review = update["risk_review"]
    
    assert review["status"] == "APPROVED", f"Expected APPROVED, got {review['status']}"
    assert len(review["violations"]) == 0
    assert review["risk_score"] <= 3.0
    
    state.update(update)
    assert should_rehedge_edge(state) == "synthesize_report"
    print(f"  -> Passed! Approved with low risk score: {review['risk_score']}/10.0.")

def test_max_rehedge_budget_exhaustion():
    print("[TEST 4/5] Testing Capital Preservation on Re-Hedge Budget Exhaustion...")
    state: AgentState = {
        "query": "Persistently toxic strategy",
        "ticker": "MEME",
        "intent": "backtest",
        "code": "# simulated code",
        "execution_output": {"success": True},
        "error_log": None,
        "retry_count": 0,
        "max_retries": 3,
        "metrics": {
            "ticker": "MEME",
            "total_strategy_return": "-15.0%",
            "sharpe_ratio": -0.2,
            "max_drawdown": "-45.0%",
            "total_trades": 80
        },
        "chart_path": None,
        "sources": None,
        "risk_review": None,
        "risk_review_count": 2,        # Reached max allowable re-hedges
        "max_risk_reviews": 2,
        "final_report": "",
        "messages": []
    }
    
    update = adversarial_risk_officer_node(state)
    review = update["risk_review"]
    
    assert review["status"] == "VETOED_REJECTED", f"Expected VETOED_REJECTED, got {review['status']}"
    assert "exhausted" in review["mandate_directive"].lower() or "preservation" in review["mandate_directive"].lower()
    
    state.update(update)
    assert should_rehedge_edge(state) == "synthesize_report"
    print("  -> Passed! Capital preservation mandated upon budget exhaustion.")

def test_multi_agent_graph_compilation():
    print("[TEST 5/5] Testing Multi-Agent StateGraph compilation & node topology...")
    graph = build_alpha_graph()
    assert graph is not None
    
    # Verify all 9 committee nodes exist
    nodes = list(graph.nodes.keys())
    expected_nodes = [
        "classify_intent",
        "filing_rag",
        "financial_health",
        "generate_code",
        "execute_code",
        "fix_code",
        "adversarial_risk_officer",
        "rehedge_code",
        "synthesize_report"
    ]
    for en in expected_nodes:
        assert en in nodes, f"Missing expected committee node: {en}"
        
    print(f"  -> Passed! All {len(expected_nodes)} committee nodes registered and compiled.")

if __name__ == "__main__":
    print("==================================================================")
    print("RUNNING ALPHA-AGENT MULTI-AGENT & ADVERSARIAL RISK TEST SUITE")
    print("==================================================================\n")
    test_drawdown_policy_veto()
    test_sharpe_hurdle_veto()
    test_compliant_strategy_approval()
    test_max_rehedge_budget_exhaustion()
    test_multi_agent_graph_compilation()
    print("\n==================================================================")
    print("ALL 5 MULTI-AGENT & RISK GUARDRAIL TESTS PASSED (5/5)!")
    print("==================================================================")
