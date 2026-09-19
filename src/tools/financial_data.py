import yfinance as yf
from typing import Dict, Any

def fetch_financial_metrics(ticker_symbol: str) -> Dict[str, Any]:
    """
    Deterministically fetches balance sheet, income statement, and valuation metrics
    for a company using yfinance. Calculates Altman Z-Score, Solvency, and Liquidity.
    Cost: 0 LLM tokens.
    """
    symbol = ticker_symbol.upper().strip()
    ticker = yf.Ticker(symbol)
    
    info = ticker.info or {}
    balance_sheet = ticker.balance_sheet
    financials = ticker.financials
    cash_flow = ticker.cashflow
    
    company_name = info.get("shortName") or info.get("longName") or symbol
    sector = info.get("sector", "N/A")
    industry = info.get("industry", "N/A")
    market_cap = info.get("marketCap", 0)
    current_price = info.get("currentPrice") or info.get("regularMarketPrice") or 0.0
    
    # 1. Valuation
    pe_ratio = info.get("trailingPE")
    forward_pe = info.get("forwardPE")
    pb_ratio = info.get("priceToBook")
    ev_to_ebitda = info.get("enterpriseToEbitda")
    
    # 2. Profitability
    gross_margin = info.get("grossMargins")
    operating_margin = info.get("operatingMargins")
    net_margin = info.get("profitMargins")
    roe = info.get("returnOnEquity")
    roa = info.get("returnOnAssets")
    
    # 3. Solvency & Liquidity
    current_ratio = info.get("currentRatio")
    quick_ratio = info.get("quickRatio")
    debt_to_equity = info.get("debtToEquity")
    total_debt = info.get("totalDebt")
    total_cash = info.get("totalCash")
    fcf = info.get("freeCashflow")
    
    # 4. Altman Z-Score Estimation
    z_score = None
    z_status = "Insufficient Data"
    
    try:
        if not balance_sheet.empty and not financials.empty:
            latest_bs = balance_sheet.iloc[:, 0]
            latest_fin = financials.iloc[:, 0]
            
            total_assets = latest_bs.get("Total Assets", 0)
            current_assets = latest_bs.get("Current Assets", 0)
            current_liab = latest_bs.get("Current Liabilities", 0)
            retained_earnings = latest_bs.get("Retained Earnings", 0)
            total_liab = latest_bs.get("Total Liabilities Net Minority Interest", 0) or latest_bs.get("Total Debt", 0)
            ebit = latest_fin.get("EBIT", 0) or latest_fin.get("Operating Income", 0)
            revenue = latest_fin.get("Total Revenue", 0)
            
            if total_assets and total_assets > 0:
                working_capital = current_assets - current_liab
                x1 = working_capital / total_assets
                x2 = (retained_earnings or 0) / total_assets
                x3 = (ebit or 0) / total_assets
                x4 = (market_cap / total_liab) if total_liab and total_liab > 0 else 1.0
                x5 = (revenue or 0) / total_assets
                
                # Altman Z-Score for public manufacturing / general corporate
                z = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 0.999 * x5
                z_score = round(float(z), 2)
                
                if z_score >= 2.99:
                    z_status = "Safe Zone (Low Bankruptcy Risk)"
                elif z_score >= 1.81:
                    z_status = "Grey Zone (Moderate Financial Risk)"
                else:
                    z_status = "Distress Zone (High Solvency Risk)"
    except Exception:
        z_score = None
        z_status = "Calculation Unavailable"
        
    return {
        "symbol": symbol,
        "company_name": company_name,
        "sector": sector,
        "industry": industry,
        "current_price": current_price,
        "market_cap": market_cap,
        "valuation": {
            "trailing_pe": pe_ratio,
            "forward_pe": forward_pe,
            "price_to_book": pb_ratio,
            "ev_to_ebitda": ev_to_ebitda,
        },
        "profitability": {
            "gross_margin": f"{round(gross_margin * 100, 2)}%" if gross_margin else "N/A",
            "operating_margin": f"{round(operating_margin * 100, 2)}%" if operating_margin else "N/A",
            "net_profit_margin": f"{round(net_margin * 100, 2)}%" if net_margin else "N/A",
            "roe": f"{round(roe * 100, 2)}%" if roe else "N/A",
            "roa": f"{round(roa * 100, 2)}%" if roa else "N/A",
        },
        "solvency_and_liquidity": {
            "current_ratio": round(current_ratio, 2) if current_ratio else "N/A",
            "quick_ratio": round(quick_ratio, 2) if quick_ratio else "N/A",
            "debt_to_equity": round(debt_to_equity, 2) if debt_to_equity else "N/A",
            "total_debt": f"${total_debt:,.0f}" if total_debt else "N/A",
            "total_cash": f"${total_cash:,.0f}" if total_cash else "N/A",
            "free_cash_flow": f"${fcf:,.0f}" if fcf else "N/A",
        },
        "altman_z_score": {
            "score": z_score,
            "status": z_status
        }
    }
