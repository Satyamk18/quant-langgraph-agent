import os
import sys
import subprocess
import json
import re
from typing import Dict, Any

def execute_backtest_code(code: str, output_dir: str = "outputs", timeout_seconds: int = 45) -> Dict[str, Any]:
    """
    Executes a generated backtest Python script locally in a controlled subprocess.
    Captures stdout, stderr, execution status, and parsed metrics.
    Cost: 0 LLM tokens.
    """
    os.makedirs(output_dir, exist_ok=True)
    script_path = os.path.join(output_dir, "temp_backtest.py")
    
    # Ensure code saves chart to outputs directory
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(code)
        
    try:
        process = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=os.getcwd()
        )
        
        stdout = process.stdout
        stderr = process.stderr
        
        if process.returncode != 0:
            return {
                "success": False,
                "error": f"Execution failed with return code {process.returncode}:\n{stderr}",
                "stdout": stdout,
                "stderr": stderr,
                "metrics": None,
                "chart_path": None
            }
            
        # Parse metrics printed by the script
        # The script is instructed to print JSON block: ===METRICS_JSON_START=== {...} ===METRICS_JSON_END===
        metrics = {}
        json_match = re.search(r"===METRICS_JSON_START===(.*?)===METRICS_JSON_END===", stdout, re.DOTALL)
        if json_match:
            try:
                metrics = json.loads(json_match.group(1).strip())
            except json.JSONDecodeError:
                metrics = {"raw_output": stdout}
        else:
            # Fallback: simple key-value lines
            metrics = {"raw_output": stdout}
            
        chart_path = os.path.join(output_dir, "equity_curve.png")
        if not os.path.exists(chart_path):
            chart_path = None
            
        return {
            "success": True,
            "error": None,
            "stdout": stdout,
            "stderr": stderr,
            "metrics": metrics,
            "chart_path": chart_path
        }
        
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": f"Execution timed out after {timeout_seconds} seconds.",
            "stdout": "",
            "stderr": "TimeoutExpired",
            "metrics": None,
            "chart_path": None
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Subprocess runner error: {str(e)}",
            "stdout": "",
            "stderr": str(e),
            "metrics": None,
            "chart_path": None
        }
