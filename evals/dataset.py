"""
AlphaAgent Evaluation Benchmark Dataset.
Contains 20 curated scenarios across 4 categories to statistically measure:
1. Pass@1 Success Rate (first-try generation)
2. Pass@3 Success Rate (post self-healing)
3. Self-Healing Recovery Efficiency
4. Latency Distribution (P50, P90, P95)
5. RAG Grounding & Citation Accuracy
"""

from typing import List, Dict, Any

BENCHMARK_SCENARIOS: List[Dict[str, Any]] = [
    # ==================== CATEGORY 1: QUANTITATIVE STRATEGIES (Standard) ====================
    {
        "id": "quant_sma_crossover",
        "category": "quantitative_strategy",
        "query": "Test a 20 and 50 SMA crossover strategy on NVDA for the last 1 year",
        "expected_intent": "backtest",
        "expected_ticker": "NVDA",
        "description": "Standard two-moving-average trend-following crossover."
    },
    {
        "id": "quant_rsi_mean_reversion",
        "category": "quantitative_strategy",
        "query": "Backtest an RSI oversold (<30) buy and overbought (>70) sell strategy on AAPL for the last 2 years",
        "expected_intent": "backtest",
        "expected_ticker": "AAPL",
        "description": "Mean-reversion momentum strategy using 14-day RSI."
    },
    {
        "id": "quant_macd_momentum",
        "category": "quantitative_strategy",
        "query": "Simulate a MACD histogram crossover strategy on MSFT using 2 years of daily data",
        "expected_intent": "backtest",
        "expected_ticker": "MSFT",
        "description": "Exponential moving average convergence/divergence momentum."
    },
    {
        "id": "quant_bollinger_breakout",
        "category": "quantitative_strategy",
        "query": "Test a Bollinger Bands breakout strategy on TSLA: buy when price breaks above upper band, exit below middle band",
        "expected_intent": "backtest",
        "expected_ticker": "TSLA",
        "description": "Volatility band expansion strategy."
    },
    {
        "id": "quant_dual_ema",
        "category": "quantitative_strategy",
        "query": "Backtest a 9 and 21 Exponential Moving Average (EMA) golden cross on SPY over the last 18 months",
        "expected_intent": "backtest",
        "expected_ticker": "SPY",
        "description": "Fast-moving exponential trend strategy on benchmark index."
    },

    # ==================== CATEGORY 2: EDGE-CASE / ADVERSARIAL BACKTESTS (Tests Self-Healing) ====================
    {
        "id": "edge_multi_index_columns",
        "category": "edge_case_backtest",
        "query": "Run a backtest for GOOGL comparing 50-day volume-weighted average price (VWAP) against close with multi-period metrics",
        "expected_intent": "backtest",
        "expected_ticker": "GOOGL",
        "description": "Tests multi-index column flattening from yfinance downloads."
    },
    {
        "id": "edge_zero_trades_protection",
        "category": "edge_case_backtest",
        "query": "Test an extreme RSI strategy on AMZN buying only when RSI is below 10 and selling when RSI is above 95",
        "expected_intent": "backtest",
        "expected_ticker": "AMZN",
        "description": "High probability of zero trades; tests zero-division safeguards in Sharpe ratio calculation."
    },
    {
        "id": "edge_stop_loss_logic",
        "category": "edge_case_backtest",
        "query": "Backtest a 50 SMA trend strategy on META with a strict 5% trailing stop-loss",
        "expected_intent": "backtest",
        "expected_ticker": "META",
        "description": "Stateful path-dependent stop-loss simulation."
    },
    {
        "id": "edge_short_history",
        "category": "edge_case_backtest",
        "query": "Test a 5-day and 10-day moving average crossover on ARM with daily returns",
        "expected_intent": "backtest",
        "expected_ticker": "ARM",
        "description": "Recently listed IPO with shorter historical price horizon."
    },
    {
        "id": "edge_complex_filter",
        "category": "edge_case_backtest",
        "query": "Backtest buying AMD when price is above 200 SMA AND 14-day RSI is between 40 and 60, exit when price drops below 50 SMA",
        "expected_intent": "backtest",
        "expected_ticker": "AMD",
        "description": "Multi-conditional compound boolean filtering in pandas."
    },

    # ==================== CATEGORY 3: DETERMINISTIC FINANCIAL HEALTH AUDITS ====================
    {
        "id": "health_boeing_distress",
        "category": "financial_health",
        "query": "Evaluate the financial health, solvency ratios, and bankruptcy risk of Boeing (BA)",
        "expected_intent": "financial_health",
        "expected_ticker": "BA",
        "description": "High debt, low margin industrials test case (Altman Z in Distress Zone)."
    },
    {
        "id": "health_microsoft_cash",
        "category": "financial_health",
        "query": "Audit the balance sheet strength, cash reserves, and Altman Z-score of Microsoft (MSFT)",
        "expected_intent": "financial_health",
        "expected_ticker": "MSFT",
        "description": "High cash, strong operating margins software benchmark."
    },
    {
        "id": "health_apple_solvency",
        "category": "financial_health",
        "query": "Check Apple (AAPL) debt-to-equity, current ratio, and capital return health",
        "expected_intent": "financial_health",
        "expected_ticker": "AAPL",
        "description": "Massive share repurchase program and net-cash-neutral policy."
    },
    {
        "id": "health_tesla_margins",
        "category": "financial_health",
        "query": "Analyze Tesla (TSLA) valuation ratios, operating margins, and Altman Z-score",
        "expected_intent": "financial_health",
        "expected_ticker": "TSLA",
        "description": "High valuation multiples and volatile automotive margins."
    },
    {
        "id": "health_nvidia_growth",
        "category": "financial_health",
        "query": "Assess financial health and balance sheet liquidity of Nvidia (NVDA)",
        "expected_intent": "financial_health",
        "expected_ticker": "NVDA",
        "description": "Hyper-growth semiconductor balance sheet with high gross margins."
    },

    # ==================== CATEGORY 4: QUALITATIVE SEC 10-K RAG RETRIEVAL ====================
    {
        "id": "rag_boeing_debt_risks",
        "category": "sec_filing_rag",
        "query": "What specific debt obligations and credit rating downgrade risks did Boeing disclose in their 10-K filing?",
        "expected_intent": "filing_qa",
        "expected_ticker": "BA",
        "description": "Retrieves Boeing 10-K debt covenant constraints and S&P/Moody's ratings risks."
    },
    {
        "id": "rag_boeing_faa_oversight",
        "category": "sec_filing_rag",
        "query": "What operational challenges and FAA quality directives on the 737 MAX were detailed in Boeing's 10-K?",
        "expected_intent": "filing_qa",
        "expected_ticker": "BA",
        "description": "Retrieves FAA production cap disclosures following the door-plug incident."
    },
    {
        "id": "rag_apple_tsmc_dependency",
        "category": "sec_filing_rag",
        "query": "What are Apple's single-source semiconductor fabrication risks with TSMC according to their 10-K?",
        "expected_intent": "filing_qa",
        "expected_ticker": "AAPL",
        "description": "Retrieves Apple 10-K disclosures regarding sole-source TSMC chip fabrication."
    },
    {
        "id": "rag_apple_supply_chain_geography",
        "category": "sec_filing_rag",
        "query": "What geographic concentration risks in Asia and lean inventory vulnerabilities did Apple disclose in their 10-K?",
        "expected_intent": "filing_qa",
        "expected_ticker": "AAPL",
        "description": "Retrieves Asian outsourced manufacturing and lean inventory stockout risks."
    },
    {
        "id": "rag_apple_ai_competition",
        "category": "sec_filing_rag",
        "query": "What disclosures did Apple make regarding generative artificial intelligence development risks in their 10-K?",
        "expected_intent": "filing_qa",
        "expected_ticker": "AAPL",
        "description": "Retrieves Apple Intelligence investments and rapid technological transition disclosures."
    }
]
