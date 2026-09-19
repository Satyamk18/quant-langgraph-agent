from typing import TypedDict, Optional, Dict, Any, List
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """
    Core state for AlphaAgent graph.
    Tracks user intent, generated code, execution logs, self-healing retries,
    and financial metrics.
    """
    query: str
    intent: str                   # "backtest", "financial_health", or "unknown"
    ticker: str                   # e.g. "AAPL", "MSFT"
    code: str                     # Generated backtest Python code
    execution_output: Optional[Dict[str, Any]] # Metrics returned by local execution
    error_log: Optional[str]      # Captured stderr/traceback if code fails
    retry_count: int              # Number of self-healing attempts
    max_retries: int              # Maximum allowed retries (default: 3)
    metrics: Optional[Dict[str, Any]] # Formatted financial or performance metrics
    chart_path: Optional[str]     # Path to generated equity curve / chart
    final_report: str             # Final LLM synthesis report
    messages: List[BaseMessage]   # Message history for agent tracking
