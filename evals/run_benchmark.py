"""
CLI Runner for AlphaAgent Evaluation Benchmark Suite.
Usage:
  python evals/run_benchmark.py
  python evals/run_benchmark.py --limit 5
  python evals/run_benchmark.py --category quantitative_strategy
"""

import os
import sys
import argparse

# Ensure UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from evals.dataset import BENCHMARK_SCENARIOS
from evals.evaluator import run_evaluation_benchmark

console = Console()

def display_benchmark_table(summary: dict):
    """Renders formatted scorecard of empirical benchmark results."""
    # Summary Metrics Panel
    metrics_text = (
        f"[bold cyan]Total Scenarios:[/bold cyan] {summary['total_scenarios_evaluated']}\n"
        f"[bold green]Pass@1 Rate (First-Pass Success):[/bold green] {summary['overall_pass_at_1_rate']}\n"
        f"[bold green]Pass@3 Rate (Post Self-Healing):[/bold green] {summary['overall_pass_at_3_rate']}\n"
        f"[bold yellow]Self-Healing Recovery Efficiency:[/bold yellow] {summary['self_healing_recovery_efficiency']}\n"
        f"[bold magenta]Intent Classification Accuracy:[/bold magenta] {summary['intent_classification_accuracy']}\n\n"
        f"[bold blue]Latency Distribution:[/bold blue]\n"
        f"  • Mean: {summary['latency_metrics']['mean_seconds']}s\n"
        f"  • P50 (Median): {summary['latency_metrics']['p50_seconds']}s\n"
        f"  • P90: {summary['latency_metrics']['p90_seconds']}s\n"
        f"  • P95: {summary['latency_metrics']['p95_seconds']}s"
    )
    console.print(Panel(metrics_text, title="[bold white]AlphaAgent Benchmark Scorecard[/bold white]", border_style="green"))
    
    # Detailed Table
    table = Table(title="Scenario-by-Scenario Evaluation Results", header_style="bold cyan")
    table.add_column("Scenario ID", style="yellow", width=28)
    table.add_column("Category", style="cyan", width=22)
    table.add_column("Pass@1", style="bold", width=8)
    table.add_column("Pass@3", style="bold", width=8)
    table.add_column("Healed?", style="magenta", width=8)
    table.add_column("Latency", style="dim", width=9)
    table.add_column("Status", style="bold", width=8)
    
    for r in summary["detailed_results"]:
        p1 = "[green]✓[/green]" if r["pass_at_1"] else "[red]✗[/red]"
        p3 = "[green]✓[/green]" if r["pass_at_3"] else "[red]✗[/red]"
        healed = "[magenta]YES[/magenta]" if r["self_healed"] else "[dim]-[/dim]"
        status_color = "[bold green]PASS[/bold green]" if r["status"] == "PASS" else "[bold red]FAIL[/bold red]"
        
        table.add_row(
            r["id"],
            r["category"],
            p1,
            p3,
            healed,
            f"{r['latency_seconds']}s",
            status_color
        )
    console.print(table)
    console.print("\n[dim]Results saved to evals/benchmark_results.json[/dim]\n")

def main():
    parser = argparse.ArgumentParser(description="AlphaAgent Benchmark Suite")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of scenarios to evaluate")
    parser.add_argument("--category", type=str, default=None, help="Filter by scenario category")
    parser.add_argument("--sample-per-category", type=int, default=None, help="Sample N scenarios from each of the 4 categories")
    args = parser.parse_args()
    
    scenarios = BENCHMARK_SCENARIOS
    if args.category:
        scenarios = [s for s in scenarios if s["category"] == args.category]
        if not scenarios:
            console.print(f"[red]No scenarios found matching category: {args.category}[/red]")
            return
            
    if args.sample_per_category:
        categories = ["quantitative_strategy", "edge_case_backtest", "financial_health", "sec_filing_rag"]
        sampled = []
        for cat in categories:
            cat_scenarios = [s for s in BENCHMARK_SCENARIOS if s["category"] == cat]
            sampled.extend(cat_scenarios[:args.sample_per_category])
        scenarios = sampled

    if args.limit:
        scenarios = scenarios[:args.limit]
        
    console.print(Panel.fit(
        f"[bold cyan]Starting AlphaAgent Evaluation Benchmark[/bold cyan]\n"
        f"Evaluating [yellow]{len(scenarios)}[/yellow] scenarios across quantitative and qualitative categories...",
        title="AI Evaluation Harness"
    ))
    
    summary = run_evaluation_benchmark(scenarios)
    display_benchmark_table(summary)

if __name__ == "__main__":
    main()
