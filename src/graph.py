import re
import json
from typing import Dict, Any, Literal, Optional, List
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import SystemMessage, HumanMessage

from src.state import AgentState, RiskReview, SpecialistFindings
from src.llm import get_llm
from src.prompts import (
    INTENT_SYSTEM_PROMPT,
    CODE_GEN_SYSTEM_PROMPT,
    CODE_FIX_SYSTEM_PROMPT,
    STRATEGY_SYNTHESIS_PROMPT,
    HEALTH_SYNTHESIS_PROMPT,
    FILING_RAG_PROMPT,
    ADVERSARIAL_RISK_PROMPT,
    QUANT_REHEDGE_PROMPT,
    COMMITTEE_SYNTHESIS_PROMPT
)
from src.tools.financial_data import fetch_financial_metrics
from src.tools.executor import execute_backtest_code
from src.rag.vectorstore import query_filings

def extract_text_content(content: Any) -> str:
    """Extracts raw text string from LLM response (handles both string and list format)."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join([part.get("text", "") if isinstance(part, dict) else str(part) for part in content]).strip()
    return str(content).strip()

def clean_python_code(raw_code: str) -> str:
    """Strips markdown code fences from LLM response."""
    cleaned = raw_code.strip()
    match = re.search(r"```(?:python)?\s*(.*?)\s*```", cleaned, re.DOTALL)
    if match:
        return match.group(1).strip()
    return cleaned

def parse_percentage_or_float(val: Any) -> Optional[float]:
    """Safely extracts numeric float from strings like '-18.4%' or numbers."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        cleaned = val.replace("%", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None

# ==================== NODE DEFINITIONS ====================

def classify_intent_node(state: AgentState) -> Dict[str, Any]:
    """
    Chief Investment Officer (Supervisor) triage:
    Classifies user query intent into backtest, financial_health, filing_qa, or committee.
    """
    llm = get_llm(temperature=0.0)
    messages = [
        SystemMessage(content=INTENT_SYSTEM_PROMPT),
        HumanMessage(content=f"User Query: {state['query']}")
    ]
    response = llm.invoke(messages)
    content = extract_text_content(response.content)
    
    intent = "backtest"
    ticker = "SPY"
    try:
        json_match = re.search(r"\{.*?\}", content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            intent = data.get("intent", "backtest")
            ticker = data.get("ticker", "SPY").upper()
    except Exception:
        pass
        
    # Heuristic fallback safety
    q = state["query"].lower()
    if any(w in q for w in ["committee", "comprehensive", "full review", "mandate", "cio"]):
        intent = "committee"
    elif any(w in q for w in ["10-k", "filing", "risk factor", "disclose", "annual report", "supply chain", "footnote", "faa", "why is", "qualitative"]):
        intent = "filing_qa"
    elif any(w in q for w in ["balance sheet", "health", "debt to equity", "solvency", "pe ratio", "ratios", "altman"]):
        intent = "financial_health"
        
    return {
        "intent": intent,
        "ticker": ticker,
        "retry_count": state.get("retry_count", 0),
        "max_retries": state.get("max_retries", 3),
        "risk_review_count": state.get("risk_review_count", 0),
        "max_risk_reviews": state.get("max_risk_reviews", 2)
    }

def route_intent_edge(state: AgentState) -> Literal["financial_health", "generate_code", "filing_rag"]:
    """Routes query based on classified intent."""
    if state["intent"] == "financial_health":
        return "financial_health"
    elif state["intent"] == "filing_qa":
        return "filing_rag"
    return "generate_code"

# ----------------- SPECIALIST: SEC 10-K FORENSIC AUDITOR -----------------

def filing_rag_node(state: AgentState) -> Dict[str, Any]:
    """
    SEC 10-K Forensic Auditor:
    Retrieves qualitative 10-K disclosures from ChromaDB vector store
    and synthesizes an answer with exact source citations.
    """
    ticker = state.get("ticker", "SPY")
    sources = query_filings(query=state["query"], ticker=ticker, top_k=4)
    
    if not sources:
        report = f"No relevant filing disclosures found in vector database for {ticker}."
        return {"sources": [], "final_report": report}
        
    context_chunks = []
    for idx, s in enumerate(sources, 1):
        context_chunks.append(
            f"--- Excerpt {idx} [Document: {s['source']}, Relevance Score: {s['score']}] ---\n"
            f"{s['content']}\n"
        )
    formatted_context = "\n".join(context_chunks)
    
    system_prompt = FILING_RAG_PROMPT.format(context=formatted_context)
    llm = get_llm(temperature=0.2)
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"User Question: {state['query']}")
    ])
    report_text = extract_text_content(response.content)
    
    return {
        "sources": sources,
        "final_report": report_text
    }

# ----------------- SPECIALIST: FUNDAMENTAL SOLVENCY AUDITOR -----------------

def financial_health_node(state: AgentState) -> Dict[str, Any]:
    """
    Fundamental Solvency Auditor:
    Fetches balance sheet & ratio metrics deterministically via yfinance,
    then synthesizes an analyst diagnosis using the LLM.
    """
    ticker = state.get("ticker", "SPY")
    metrics_data = fetch_financial_metrics(ticker)
    
    llm = get_llm(temperature=0.2)
    prompt = (
        f"Company: {metrics_data.get('company_name')} ({ticker})\n"
        f"Current Price: ${metrics_data.get('current_price')}\n"
        f"Valuation: {json.dumps(metrics_data.get('valuation', {}))}\n"
        f"Profitability: {json.dumps(metrics_data.get('profitability', {}))}\n"
        f"Solvency/Liquidity: {json.dumps(metrics_data.get('solvency_and_liquidity', {}))}\n"
        f"Altman Z-Score: {json.dumps(metrics_data.get('altman_z_score', {}))}\n"
    )
    
    messages = [
        SystemMessage(content=HEALTH_SYNTHESIS_PROMPT),
        HumanMessage(content=prompt)
    ]
    response = llm.invoke(messages)
    report_text = extract_text_content(response.content)
    
    return {
        "metrics": metrics_data,
        "final_report": report_text
    }

# ----------------- SPECIALIST: QUANT RESEARCHER & SANDBOX -----------------

def generate_code_node(state: AgentState) -> Dict[str, Any]:
    """Quant Researcher: Generates Python backtesting script using the LLM."""
    llm = get_llm(temperature=0.1)
    prompt = (
        f"Write a backtest script for the following request.\n"
        f"Ticker: {state['ticker']}\n"
        f"Strategy Requirement: {state['query']}\n"
    )
    messages = [
        SystemMessage(content=CODE_GEN_SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]
    response = llm.invoke(messages)
    code_text = extract_text_content(response.content)
    cleaned_code = clean_python_code(code_text)
    
    return {
        "code": cleaned_code
    }

def execute_code_node(state: AgentState) -> Dict[str, Any]:
    """Executes the generated Python backtest code in the Zero-Trust Sandbox."""
    code = state["code"]
    res = execute_backtest_code(code)
    
    if res["success"]:
        return {
            "execution_output": res,
            "error_log": None,
            "metrics": res["metrics"],
            "chart_path": res["chart_path"]
        }
    else:
        return {
            "execution_output": res,
            "error_log": res["error"],
            "metrics": None,
            "chart_path": None
        }

def should_retry_edge(state: AgentState) -> Literal["fix_code", "adversarial_risk_officer", "synthesize_report"]:
    """
    Inner Self-Healing Loop Edge:
    - If execution failed with error: routes to fix_code (up to max_retries).
    - If max retries exhausted with error: routes to synthesize_report.
    - If execution succeeded: routes to adversarial_risk_officer for institutional audit!
    """
    if state.get("error_log"):
        if state.get("retry_count", 0) < state.get("max_retries", 3):
            return "fix_code"
        return "synthesize_report"
    return "adversarial_risk_officer"

def fix_code_node(state: AgentState) -> Dict[str, Any]:
    """
    Inner Self-Healing Node: Passes traceback & failed code to LLM to fix syntax/runtime bugs.
    """
    llm = get_llm(temperature=0.1)
    prompt = (
        f"Original User Request: {state['query']}\n"
        f"Failed Code:\n```python\n{state['code']}\n```\n\n"
        f"Error Traceback:\n{state['error_log']}\n"
    )
    messages = [
        SystemMessage(content=CODE_FIX_SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]
    response = llm.invoke(messages)
    code_text = extract_text_content(response.content)
    cleaned_code = clean_python_code(code_text)
    
    new_retry_count = state.get("retry_count", 0) + 1
    return {
        "code": cleaned_code,
        "retry_count": new_retry_count
    }

# ----------------- ADVERSARIAL RISK & COMPLIANCE OFFICER -----------------

def adversarial_risk_officer_node(state: AgentState) -> Dict[str, Any]:
    """
    Adversarial Risk & Compliance Officer:
    Independently audits the proposed quantitative strategy against institutional risk budgets:
    1. Maximum Drawdown <= 15.0%
    2. Annualized Sharpe Ratio >= 1.0
    3. Solvency & Altman Z-Score alignment
    
    Has hard veto authority: renders APPROVED, VETOED_RETRY, or VETOED_REJECTED verdict.
    """
    metrics = state.get("metrics") or {}
    ticker = state.get("ticker", "SPY")
    
    # 1. Deterministic Policy Rule Checking
    violations: List[str] = []
    
    # Policy 1: Max Drawdown Check (<= 15.0%)
    drawdown_raw = metrics.get("max_drawdown")
    drawdown_val = parse_percentage_or_float(drawdown_raw)
    if drawdown_val is not None:
        # Max drawdown is typically negative, e.g. -22.4% -> abs is 22.4%
        if abs(drawdown_val) > 15.0:
            violations.append(
                f"Policy 1 Violation: Maximum Drawdown ({drawdown_val:.1f}%) exceeds institutional limit (15.0%)."
            )
            
    # Policy 2: Sharpe Ratio Hurdle (>= 1.0)
    sharpe_raw = metrics.get("sharpe_ratio")
    sharpe_val = parse_percentage_or_float(sharpe_raw)
    if sharpe_val is not None:
        if sharpe_val < 1.0:
            violations.append(
                f"Policy 2 Violation: Annualized Sharpe Ratio ({sharpe_val:.2f}) falls below hurdle rate (1.00)."
            )
            
    current_risk_count = state.get("risk_review_count", 0)
    max_risk_reviews = state.get("max_risk_reviews", 2)
    
    # Determine Status
    if not violations:
        status: Literal["APPROVED", "VETOED_RETRY", "VETOED_REJECTED"] = "APPROVED"
        risk_score = 2.0
        mandate_directive = None
        summary = "Strategy fully complies with institutional risk budgets (Drawdown <= 15.0%, Sharpe >= 1.0)."
    else:
        risk_score = min(9.5, 4.0 + len(violations) * 2.5)
        if current_risk_count < max_risk_reviews:
            status = "VETOED_RETRY"
            directive_parts = []
            if any("Drawdown" in v for v in violations):
                directive_parts.append("implement a 5% trailing stop-loss or tighten volatility exit bands")
            if any("Sharpe" in v for v in violations):
                directive_parts.append("filter out low-conviction signals using an additional trend filter (e.g. 200 SMA)")
            mandate_directive = f"MANDATORY RE-HEDGE DIRECTIVE: Quant must {' and '.join(directive_parts)} to compress risk."
            summary = f"Strategy VETOED by Risk Officer due to {len(violations)} institutional policy violation(s)."
        else:
            status = "VETOED_REJECTED"
            mandate_directive = "Re-hedging budget exhausted. Recommend capital preservation in risk-free cash."
            summary = "Strategy permanently REJECTED after repeated compliance failures. Capital preservation mandated."

    risk_review: RiskReview = {
        "status": status,
        "violations": violations,
        "risk_score": risk_score,
        "mandate_directive": mandate_directive,
        "summary": summary
    }

    # Increment risk review count if veto occurred
    new_risk_count = current_risk_count + (1 if status != "APPROVED" else 0)

    return {
        "risk_review": risk_review,
        "risk_review_count": new_risk_count
    }

def should_rehedge_edge(state: AgentState) -> Literal["rehedge_code", "synthesize_report"]:
    """
    Outer Risk Re-Hedging Loop Edge:
    - If strategy was VETOED_RETRY and re-hedge attempts remain: routes to rehedge_code.
    - If strategy was APPROVED or retries exhausted: routes to synthesize_report.
    """
    review = state.get("risk_review")
    if review and review.get("status") == "VETOED_RETRY":
        if state.get("risk_review_count", 0) <= state.get("max_risk_reviews", 2):
            return "rehedge_code"
    return "synthesize_report"

def rehedge_code_node(state: AgentState) -> Dict[str, Any]:
    """
    Quant Re-Hedging Node:
    Instructs the Quant to rewrite the trading strategy adhering strictly to the
    Risk & Compliance Officer's mandatory mitigation instructions.
    """
    llm = get_llm(temperature=0.1)
    risk_review = state.get("risk_review", {})
    directive = risk_review.get("mandate_directive", "Implement a 5% trailing stop-loss to reduce drawdown.")
    violations = "\n".join(f"- {v}" for v in risk_review.get("violations", []))
    
    prompt = (
        f"Ticker: {state['ticker']}\n"
        f"Previous Strategy Code:\n```python\n{state['code']}\n```\n\n"
        f"Risk Officer Violations:\n{violations}\n\n"
        f"MANDATORY RISK MITIGATION DIRECTIVE:\n{directive}\n\n"
        f"Modify the script to incorporate the required risk safeguards and reduce maximum drawdown below 15%."
    )
    messages = [
        SystemMessage(content=QUANT_REHEDGE_PROMPT),
        HumanMessage(content=prompt)
    ]
    response = llm.invoke(messages)
    code_text = extract_text_content(response.content)
    cleaned_code = clean_python_code(code_text)
    
    return {
        "code": cleaned_code
    }

# ----------------- COMMITTEE SYNTHESIS: MEMORANDUM -----------------

def synthesize_report_node(state: AgentState) -> Dict[str, Any]:
    """
    Chief Investment Officer Synthesis:
    Synthesizes specialist findings and Risk Officer verdict into an
    Institutional Investment & Risk Memorandum.
    """
    # 1. Error state handling
    if state.get("error_log"):
        report = (
            f"❌ **Backtest Execution Incomplete**\n\n"
            f"The agent attempted to execute and self-heal the strategy script {state.get('retry_count', 0)} times, "
            f"but encountered persistent runtime errors:\n\n"
            f"```\n{state.get('error_log')}\n```"
        )
        return {"final_report": report}
        
    metrics = state.get("metrics", {})
    risk_review = state.get("risk_review")
    
    # 2. If Risk Review was conducted, incorporate it into the Institutional Memo
    if risk_review:
        status = risk_review.get("status", "APPROVED")
        status_badge = "🟢 APPROVED" if status == "APPROVED" else ("🔴 VETOED (REJECTED)" if status == "VETOED_REJECTED" else "🟡 VETOED (RE-HEDGING)")
        
        prompt = (
            f"Asset Ticker: {state['ticker']}\n"
            f"User Strategy Query: {state['query']}\n"
            f"Quantitative Backtest Metrics: {json.dumps(metrics, indent=2)}\n"
            f"Chief Risk Officer Audit:\n"
            f"  Status: {status_badge}\n"
            f"  Risk Score: {risk_review.get('risk_score')}/10.0\n"
            f"  Policy Violations: {json.dumps(risk_review.get('violations', []))}\n"
            f"  Mandate Directive: {risk_review.get('mandate_directive')}\n"
            f"  Risk Summary: {risk_review.get('summary')}\n"
        )
        llm = get_llm(temperature=0.2)
        messages = [
            SystemMessage(content=COMMITTEE_SYNTHESIS_PROMPT),
            HumanMessage(content=prompt)
        ]
        response = llm.invoke(messages)
        report_text = extract_text_content(response.content)
        
        # Prepend explicit visual compliance scorecard
        scorecard = (
            f"### 🏛️ Institutional Investment Committee Memorandum\n\n"
            f"| Governance Dimension | Assessment | Status |\n"
            f"| :--- | :--- | :--- |\n"
            f"| **Target Asset** | `{state['ticker']}` | Active Mandate |\n"
            f"| **Chief Risk Officer Verdict** | `{status}` | {status_badge} |\n"
            f"| **Risk Score** | `{risk_review.get('risk_score')}/10.0` | {'🟢 Low Hazard' if risk_review.get('risk_score', 5) < 5 else '🔴 Elevated Risk'} |\n"
            f"| **Max Drawdown** | `{metrics.get('max_drawdown', 'N/A')}` | {'🟢 Compliant' if not any('Drawdown' in v for v in risk_review.get('violations', [])) else '🔴 Policy Breach (>15%)'} |\n"
            f"| **Sharpe Ratio** | `{metrics.get('sharpe_ratio', 'N/A')}` | {'🟢 Compliant' if not any('Sharpe' in v for v in risk_review.get('violations', [])) else '🔴 Hurdle Failure (<1.0)'} |\n\n"
        )
        return {"final_report": scorecard + report_text}

    # 3. Standard strategy synthesis fallback
    llm = get_llm(temperature=0.2)
    prompt = (
        f"Strategy Backtest Results for {state['ticker']}:\n"
        f"Metrics: {json.dumps(metrics, indent=2)}\n"
    )
    messages = [
        SystemMessage(content=STRATEGY_SYNTHESIS_PROMPT),
        HumanMessage(content=prompt)
    ]
    response = llm.invoke(messages)
    report_text = extract_text_content(response.content)
    
    return {
        "final_report": report_text
    }

# ==================== GRAPH BUILDER ====================

def build_alpha_graph():
    """
    Compiles and returns the LangGraph Hierarchical Multi-Agent workflow:
    - 3-Way Intent Routing (Backtest, Solvency, 10-K RAG)
    - Inner Self-Healing Loop (Execute <-> Fix Code)
    - Outer Adversarial Risk Loop (Risk Officer <-> Re-Hedge <-> Execute)
    - Executive Committee Memorandum Synthesis
    """
    workflow = StateGraph(AgentState)
    
    # Register Nodes
    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("filing_rag", filing_rag_node)
    workflow.add_node("financial_health", financial_health_node)
    workflow.add_node("generate_code", generate_code_node)
    workflow.add_node("execute_code", execute_code_node)
    workflow.add_node("fix_code", fix_code_node)
    workflow.add_node("adversarial_risk_officer", adversarial_risk_officer_node)
    workflow.add_node("rehedge_code", rehedge_code_node)
    workflow.add_node("synthesize_report", synthesize_report_node)
    
    # Register Entry Edge
    workflow.add_edge(START, "classify_intent")
    
    # Conditional edge from intent classification (3-Way Branching)
    workflow.add_conditional_edges(
        "classify_intent",
        route_intent_edge,
        {
            "filing_rag": "filing_rag",
            "financial_health": "financial_health",
            "generate_code": "generate_code"
        }
    )
    
    # RAG branch completes after synthesis
    workflow.add_edge("filing_rag", END)
    
    # Financial Health branch completes after synthesis
    workflow.add_edge("financial_health", END)
    
    # Quant Generation -> Sandbox Execution
    workflow.add_edge("generate_code", "execute_code")
    
    # Inner Loop: Code execution error checking & self-healing
    workflow.add_conditional_edges(
        "execute_code",
        should_retry_edge,
        {
            "fix_code": "fix_code",
            "adversarial_risk_officer": "adversarial_risk_officer",
            "synthesize_report": "synthesize_report"
        }
    )
    
    # Fix code loops back to execute code (Inner Loop cycle)
    workflow.add_edge("fix_code", "execute_code")
    
    # Outer Loop: Risk Officer evaluation & re-hedging
    workflow.add_conditional_edges(
        "adversarial_risk_officer",
        should_rehedge_edge,
        {
            "rehedge_code": "rehedge_code",
            "synthesize_report": "synthesize_report"
        }
    )
    
    # Re-hedge code loops back to execute code (Outer Loop cycle)
    workflow.add_edge("rehedge_code", "execute_code")
    
    # Report synthesis terminates the workflow
    workflow.add_edge("synthesize_report", END)
    
    return workflow.compile()
