import os
import sys
import json
import argparse
from dotenv import load_dotenv

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

# Load environment
load_dotenv()

console = Console()

def check_or_prompt_api_key():
    """Checks for API keys, or interactively prompts user if missing."""
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if (gemini_key and gemini_key != "your_gemini_api_key_here") or (openai_key and openai_key != "your_openai_api_key_here"):
        return True
        
    console.print(Panel.fit(
        "[bold yellow]AlphaAgent Configuration[/bold yellow]\n\n"
        "No LLM API key detected in `.env`.\n"
        "To enable strategy code generation and analysis, you need an API key.\n"
        "• [cyan]Google Gemini API key[/cyan] is free at: [u]https://aistudio.google.com/app/apikey[/u]\n"
        "• Or an [cyan]OpenAI API key[/cyan] from: [u]https://platform.openai.com/api-keys[/u]",
        title="API Key Required"
    ))
    
    key_input = console.input("\n[bold green]Enter your Gemini API key (or OpenAI key): [/bold green]").strip()
    if not key_input:
        console.print("[red]No key provided. Exiting.[/red]")
        return False
        
    os.environ["GEMINI_API_KEY"] = key_input
    os.environ["LLM_PROVIDER"] = "gemini"
    os.environ["MODEL_NAME"] = "gemini-3.6-flash"
    with open(".env", "a", encoding="utf-8") as f:
        f.write(f"\nGEMINI_API_KEY={key_input}\nLLM_PROVIDER=gemini\nMODEL_NAME=gemini-3.6-flash\n")
            
    console.print("[green]✓ Key saved to .env successfully![/green]\n")
    return True

def display_metrics_table(metrics: dict, title: str = "Results Scorecard"):
    """Displays key-value metrics inside a rich formatted table."""
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", width=30)
    table.add_column("Value", style="bold green")
    
    for k, v in metrics.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                table.add_row(f"{k} -> {sub_k}".replace("_", " ").title(), str(sub_v))
        else:
            table.add_row(str(k).replace("_", " ").title(), str(v))
            
    console.print(table)

def display_sources_table(sources: list):
    """Displays retrieved RAG sources and relevance scores."""
    table = Table(title="Retrieved SEC Filing Disclosures (RAG Context)", show_header=True, header_style="bold cyan")
    table.add_column("#", style="dim", width=4)
    table.add_column("Document Source", style="yellow", width=30)
    table.add_column("Ticker", style="bold green", width=8)
    table.add_column("Relevance Score", style="cyan", width=16)
    table.add_column("Excerpt Preview", style="white")
    
    for idx, s in enumerate(sources, 1):
        preview = s.get("content", "").replace("\n", " ")[:90] + "..."
        table.add_row(
            str(idx),
            s.get("source", "10-K Filing"),
            s.get("ticker", "N/A"),
            str(s.get("score", "N/A")),
            preview
        )
    console.print(table)

def run_agent(query: str):
    """Executes the query through the compiled LangGraph workflow."""
    from src.graph import build_alpha_graph
    
    graph = build_alpha_graph()
    
    initial_state = {
        "query": query,
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
    
    console.print(f"\n[bold blue]Running AlphaAgent on query:[/bold blue] '{query}'\n")
    
    with console.status("[bold green]Executing LangGraph workflow...[/bold green]", spinner="dots") as status:
        final_state = initial_state
        for event in graph.stream(initial_state):
            for node_name, state_update in event.items():
                if node_name == "classify_intent":
                    status.update(f"[cyan]Intent classified:[/cyan] {state_update.get('intent')} for ticker {state_update.get('ticker')}")
                elif node_name == "filing_rag":
                    status.update("[cyan]Searching SEC 10-K vector store & synthesizing citation report...[/cyan]")
                elif node_name == "generate_code":
                    status.update(f"[cyan]Generated backtest script...[/cyan] Executing locally")
                elif node_name == "execute_code":
                    if state_update.get("error_log"):
                        status.update("[yellow]Execution encountered an error. Triggering self-healing...[/yellow]")
                    else:
                        status.update("[green]Backtest executed successfully![/green]")
                elif node_name == "fix_code":
                    status.update(f"[yellow]Self-healing retry #{state_update.get('retry_count')}...[/yellow]")
                elif node_name == "financial_health":
                    status.update("[cyan]Fetched fundamental data & synthesizing health audit...[/cyan]")
                elif node_name == "synthesize_report":
                    status.update("[green]Generating final executive summary...[/green]")
                
                final_state.update(state_update)

    # 1. Display Metrics Scorecard (if numerical health or backtest)
    if final_state.get("metrics"):
        console.print("\n")
        display_metrics_table(final_state["metrics"], title=f"Scorecard for {final_state.get('ticker', 'Asset')}")
        
    # 2. Display RAG Sources (if qualitative filing search)
    if final_state.get("sources"):
        console.print("\n")
        display_sources_table(final_state["sources"])

    # 3. Display Equity Curve location if available
    if final_state.get("chart_path") and os.path.exists(final_state["chart_path"]):
        console.print(f"\n📈 [bold green]Equity Curve Plot Saved:[/bold green] [underline]{os.path.abspath(final_state['chart_path'])}[/underline]\n")
        
    # 4. Display Final Synthesis Report
    if final_state.get("final_report"):
        console.print(Panel(
            Markdown(final_state["final_report"]),
            title=f"AlphaAgent Executive Report - {final_state.get('ticker', '')}",
            border_style="green"
        ))

def main():
    parser = argparse.ArgumentParser(description="AlphaAgent CLI")
    parser.add_argument("-q", "--query", type=str, help="Single query to run directly without interactive prompt")
    args = parser.parse_args()

    if not check_or_prompt_api_key():
        return

    if args.query:
        run_agent(args.query)
        return

    console.print(Panel.fit(
        "[bold cyan]AlphaAgent: Quant Strategy, Financial Health & SEC 10-K RAG AI[/bold cyan]\n"
        "[dim]Powered by LangChain & LangGraph | Vector RAG & Self-Healing Architecture[/dim]\n\n"
        "Sample commands you can try:\n"
        "  1. [yellow]What specific debt and FAA regulatory risks did Boeing disclose in their 10-K?[/yellow]\n"
        "  2. [yellow]What are Apple's supply chain and TSMC chip manufacturing risks?[/yellow]\n"
        "  3. [yellow]Test a 20 and 50 SMA crossover strategy on NVDA for the last 1 year[/yellow]\n"
        "  4. [yellow]Evaluate financial health and bankruptcy risk of Boeing (BA)[/yellow]\n"
        "Type [bold red]'exit'[/bold red] or [bold red]'quit'[/bold red] to close.",
        title="Welcome to AlphaAgent"
    ))
    
    while True:
        try:
            query = console.input("\n[bold cyan]AlphaAgent > [/bold cyan]").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                console.print("[dim]Goodbye![/dim]")
                break
            run_agent(query)
        except KeyboardInterrupt:
            console.print("\n[dim]Session interrupted. Goodbye![/dim]")
            break
        except Exception as e:
            console.print(f"\n[bold red]Error running agent:[/bold red] {e}\n")

if __name__ == "__main__":
    main()
