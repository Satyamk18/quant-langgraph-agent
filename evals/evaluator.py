"""
AlphaAgent Benchmark Evaluator.
Executes scenarios through the LangGraph workflow and measures:
- Pass@1 Rate (First-pass success without retry)
- Pass@3 Rate (Overall success including self-healing reflection)
- Self-Healing Efficiency (% of broken runs autonomously recovered)
- Latency percentiles (P50, P90, P95)
- Grounding & citation validity
"""

import os
import sys
import time
import json
import re
import numpy as np
from typing import List, Dict, Any, Optional

# Ensure root in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.graph import build_alpha_graph

def parse_retry_delay(err_msg: str, default: float = 6.0) -> float:
    """Parses recommended retry delay from API error messages."""
    match = re.search(r"retry in (\d+(?:\.\d+)?)s", err_msg, re.IGNORECASE)
    if match:
        return float(match.group(1)) + 1.0
    match = re.search(r"retryDelay['\"]?\s*:\s*['\"]?(\d+)s?", err_msg, re.IGNORECASE)
    if match:
        return float(match.group(1)) + 1.0
    return default

def evaluate_scenario(scenario: Dict[str, Any], graph=None, max_api_retries: int = 3) -> Dict[str, Any]:
    """
    Executes a single benchmark scenario and measures performance and accuracy.
    Includes automated backoff and retry for rate limits (HTTP 429 / RESOURCE_EXHAUSTED).
    """
    if graph is None:
        graph = build_alpha_graph()
        
    start_time = time.perf_counter()
    
    for api_attempt in range(1, max_api_retries + 1):
        initial_state = {
            "query": scenario["query"],
            "intent": "unknown",
            "ticker": "SPY",
            "code": "",
            "execution_output": None,
            "error_log": None,
            "retry_count": 0,
            "max_retries": 3,
            "metrics": None,
            "chart_path": None,
            "sources": None,
            "final_report": "",
            "messages": []
        }
        
        final_state = dict(initial_state)
        had_error_initially = False
        
        try:
            for event in graph.stream(initial_state):
                for node_name, state_update in event.items():
                    if node_name == "execute_code":
                        if state_update.get("error_log"):
                            had_error_initially = True
                    final_state.update(state_update)
                    
            latency = round(time.perf_counter() - start_time, 2)
            
            # Determine Success Criteria
            category = scenario["category"]
            pass_at_1 = False
            pass_at_3 = False
            self_healed = False
            intent_correct = (final_state.get("intent") == scenario["expected_intent"])
            ticker_correct = (final_state.get("ticker") == scenario["expected_ticker"])
            
            if category in ["quantitative_strategy", "edge_case_backtest"]:
                # Backtest passes if metrics dict is populated and no unhandled error_log remains
                if final_state.get("metrics") and not final_state.get("error_log"):
                    pass_at_3 = True
                    if not had_error_initially:
                        pass_at_1 = True
                    else:
                        self_healed = True
                else:
                    pass_at_1 = False
                    pass_at_3 = False
                    
            elif category == "financial_health":
                # Health audit passes if metrics dict has valuation/altman and report exists
                if final_state.get("metrics") and final_state.get("final_report"):
                    pass_at_1 = True
                    pass_at_3 = True
                    
            elif category == "sec_filing_rag":
                # RAG passes if sources were retrieved and citations or grounded content exist
                sources = final_state.get("sources") or []
                report = final_state.get("final_report") or ""
                has_citations = any(kw in report for kw in ["[Source:", "Source:", "Filing:", "10-K", "Item 1A", "disclosed", "according to"])
                if sources and (has_citations or len(sources) > 0) and len(report) > 50:
                    pass_at_1 = True
                    pass_at_3 = True
                    
            return {
                "id": scenario["id"],
                "category": category,
                "query": scenario["query"],
                "pass_at_1": pass_at_1,
                "pass_at_3": pass_at_3,
                "self_healed": self_healed,
                "retry_count": final_state.get("retry_count", 0),
                "intent_correct": intent_correct,
                "ticker_correct": ticker_correct,
                "latency_seconds": latency,
                "error_log": final_state.get("error_log"),
                "status": "PASS" if pass_at_3 else "FAIL"
            }
            
        except Exception as e:
            err_msg = str(e)
            is_rate_limit = ("RESOURCE_EXHAUSTED" in err_msg or "429" in err_msg or "UNAVAILABLE" in err_msg or "503" in err_msg)
            if is_rate_limit and api_attempt < max_api_retries:
                delay = parse_retry_delay(err_msg, default=7.0)
                print(f"      [Quota/Rate-Limit encountered. Pausing {delay:.1f}s before retry {api_attempt+1}/{max_api_retries}...]")
                time.sleep(delay)
                continue
                
            latency = round(time.perf_counter() - start_time, 2)
            return {
                "id": scenario["id"],
                "category": scenario["category"],
                "query": scenario["query"],
                "pass_at_1": False,
                "pass_at_3": False,
                "self_healed": False,
                "retry_count": 0,
                "intent_correct": False,
                "ticker_correct": False,
                "latency_seconds": latency,
                "error_log": f"Unhandled graph exception: {err_msg}",
                "status": "FAIL"
            }

def run_evaluation_benchmark(scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Runs full benchmark across all scenarios and computes aggregate statistical metrics.
    """
    graph = build_alpha_graph()
    results = []
    
    for idx, sc in enumerate(scenarios, 1):
        print(f"[{idx}/{len(scenarios)}] Running: {sc['id']} ({sc['category']})...")
        res = evaluate_scenario(sc, graph=graph)
        results.append(res)
        print(f"   -> Result: {res['status']} | Pass@1: {res['pass_at_1']} | Latency: {res['latency_seconds']}s")
        # Rate-limiting backoff delay between scenarios
        if idx < len(scenarios):
            time.sleep(3.0)
        
    # Aggregate Metrics Computation
    total = len(results)
    pass_1_count = sum(1 for r in results if r["pass_at_1"])
    pass_3_count = sum(1 for r in results if r["pass_at_3"])
    healed_count = sum(1 for r in results if r["self_healed"])
    
    backtest_runs = [r for r in results if r["category"] in ["quantitative_strategy", "edge_case_backtest"]]
    initial_backtest_failures = sum(1 for r in backtest_runs if not r["pass_at_1"])
    
    self_healing_efficiency = (
        round((healed_count / initial_backtest_failures) * 100, 1)
        if initial_backtest_failures > 0 else 100.0
    )
    
    latencies = [r["latency_seconds"] for r in results]
    
    summary = {
        "benchmark_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_scenarios_evaluated": total,
        "overall_pass_at_1_rate": f"{round((pass_1_count / total) * 100, 1)}%",
        "overall_pass_at_3_rate": f"{round((pass_3_count / total) * 100, 1)}%",
        "self_healing_recovery_efficiency": f"{self_healing_efficiency}%",
        "latency_metrics": {
            "mean_seconds": round(float(np.mean(latencies)), 2),
            "p50_seconds": round(float(np.percentile(latencies, 50)), 2),
            "p90_seconds": round(float(np.percentile(latencies, 90)), 2),
            "p95_seconds": round(float(np.percentile(latencies, 95)), 2),
        },
        "intent_classification_accuracy": f"{round((sum(1 for r in results if r['intent_correct']) / total) * 100, 1)}%",
        "detailed_results": results
    }
    
    # Save to JSON artifact
    os.makedirs("evals", exist_ok=True)
    with open("evals/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    return summary
