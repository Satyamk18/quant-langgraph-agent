import re
import json
from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import SystemMessage, HumanMessage

from src.state import AgentState
from src.llm import get_llm
from src.prompts import (
    INTENT_SYSTEM_PROMPT,
    CODE_GEN_SYSTEM_PROMPT,
    CODE_FIX_SYSTEM_PROMPT,
    STRATEGY_SYNTHESIS_PROMPT,
    HEALTH_SYNTHESIS_PROMPT,
    FILING_RAG_PROMPT
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

# ==================== NODE DEFINITIONS ====================

def classify_intent_node(state: AgentState) -> Dict[str, Any]:
    """Classifies user query intent into backtest, financial_health, or filing_qa."""
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
    if any(w in q for w in ["10-k", "filing", "risk factor", "disclose", "annual report", "supply chain", "footnote", "faa", "why is", "qualitative"]):
        intent = "filing_qa"
    elif any(w in q for w in ["balance sheet", "health", "debt to equity", "solvency", "pe ratio", "ratios", "altman"]):
        intent = "financial_health"
        
    return {
        "intent": intent,
        "ticker": ticker
    }

def route_intent_edge(state: AgentState) -> Literal["financial_health", "generate_code", "filing_rag"]:
    """Routes query based on classified intent."""
    if state["intent"] == "financial_health":
        return "financial_health"
    elif state["intent"] == "filing_qa":
        return "filing_rag"
    return "generate_code"

def filing_rag_node(state: AgentState) -> Dict[str, Any]:
    """
    Retrieves qualitative 10-K disclosures from ChromaDB vector store
    and synthesizes an answer with exact source citations.
    """
    ticker = state.get("ticker", "SPY")
    sources = query_filings(query=state["query"], ticker=ticker, top_k=4)
    
    if not sources:
        report = f"No relevant filing disclosures found in vector database for {ticker}."
        return {"sources": [], "final_report": report}
        
    # Format retrieved excerpts into context
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

def financial_health_node(state: AgentState) -> Dict[str, Any]:
    """
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

def generate_code_node(state: AgentState) -> Dict[str, Any]:
    """Generates Python backtesting script using the LLM."""
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
    """Executes the generated Python backtest code locally."""
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

def should_retry_edge(state: AgentState) -> Literal["fix_code", "synthesize_report"]:
    """
    Evaluates execution result. If errors occurred and retries remain,
    routes to the self-healing node.
    """
    if state.get("error_log"):
        if state.get("retry_count", 0) < state.get("max_retries", 3):
            return "fix_code"
    return "synthesize_report"

def fix_code_node(state: AgentState) -> Dict[str, Any]:
    """
    Self-healing node: Passes traceback & failed code to LLM to fix bugs.
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

def synthesize_report_node(state: AgentState) -> Dict[str, Any]:
    """Synthesizes the backtest performance into an executive summary."""
    if state.get("error_log"):
        report = (
            f"❌ **Backtest Execution Incomplete**\n\n"
            f"The agent attempted to execute and self-heal the strategy script {state.get('retry_count', 0)} times, "
            f"but encountered persistent runtime errors:\n\n"
            f"```\n{state.get('error_log')}\n```"
        )
        return {"final_report": report}
        
    metrics = state.get("metrics", {})
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
    """Compiles and returns the LangGraph workflow."""
    workflow = StateGraph(AgentState)
    
    # Register Nodes
    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("filing_rag", filing_rag_node)
    workflow.add_node("financial_health", financial_health_node)
    workflow.add_node("generate_code", generate_code_node)
    workflow.add_node("execute_code", execute_code_node)
    workflow.add_node("fix_code", fix_code_node)
    workflow.add_node("synthesize_report", synthesize_report_node)
    
    # Register Edges
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
    
    # Backtest branch
    workflow.add_edge("generate_code", "execute_code")
    
    # Conditional edge for self-healing loop
    workflow.add_conditional_edges(
        "execute_code",
        should_retry_edge,
        {
            "fix_code": "fix_code",
            "synthesize_report": "synthesize_report"
        }
    )
    
    # Fix code loops back to execute code! (LangGraph cycle)
    workflow.add_edge("fix_code", "execute_code")
    
    # Report synthesis terminates the graph
    workflow.add_edge("synthesize_report", END)
    
    return workflow.compile()
