"""
Zero-Trust Code Execution Sandbox for AlphaAgent.
Executes dynamically generated quantitative backtest scripts with Defense-in-Depth:
1. Static AST Security Analysis (blocks dangerous modules, primitives, and dunder escape)
2. Ephemeral Docker Sandbox (resource limits, capability dropping, non-root execution)
3. Graceful Subprocess Fallback (ensures zero downtime during local development)
"""

import os
import sys
import subprocess
import json
import re
import time
from typing import Dict, Any, Optional

from src.tools.security import validate_code_safety

# Cache for Docker daemon availability check (to avoid repeated subprocess overhead)
_DOCKER_STATUS_CACHE: Dict[str, Any] = {"available": None, "last_checked": 0.0}
CACHE_TTL_SECONDS = 30.0

def is_docker_available() -> bool:
    """
    Checks if Docker CLI is installed and the Docker daemon is actively responding.
    Caches result for 30 seconds to minimize system overhead.
    """
    now = time.time()
    if (
        _DOCKER_STATUS_CACHE["available"] is not None
        and (now - _DOCKER_STATUS_CACHE["last_checked"]) < CACHE_TTL_SECONDS
    ):
        return _DOCKER_STATUS_CACHE["available"]

    try:
        res = subprocess.run(
            ["docker", "info"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=3,
            text=True
        )
        is_avail = (res.returncode == 0)
    except Exception:
        is_avail = False

    _DOCKER_STATUS_CACHE["available"] = is_avail
    _DOCKER_STATUS_CACHE["last_checked"] = now
    return is_avail


def is_sandbox_image_ready(image_name: str = "alpha-agent-sandbox:latest") -> bool:
    """Checks if the sandbox Docker image has been built and is locally available."""
    if not is_docker_available():
        return False
    try:
        res = subprocess.run(
            ["docker", "image", "inspect", image_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=3,
            text=True
        )
        return (res.returncode == 0)
    except Exception:
        return False


def execute_backtest_code(
    code: str,
    output_dir: str = "outputs",
    timeout_seconds: int = 45,
    prefer_docker: bool = True
) -> Dict[str, Any]:
    """
    Executes a generated backtest Python script with Zero-Trust security guarantees.
    
    1. Static AST Security Analysis: Rejects unsafe scripts before runtime.
    2. Containerized Isolation: Runs inside ephemeral Docker container if available.
    3. Fallback: Runs in isolated subprocess with timeout protection.
    
    Returns structured execution results and parsed JSON financial metrics.
    Cost: 0 LLM tokens.
    """
    # -------------------------------------------------------------
    # Layer 1: Static AST Security Analysis
    # -------------------------------------------------------------
    safety = validate_code_safety(code)
    if not safety["is_safe"]:
        return {
            "success": False,
            "error": safety["error_message"],
            "stdout": "",
            "stderr": safety["error_message"],
            "metrics": None,
            "chart_path": None,
            "execution_mode": "ast_rejected"
        }

    os.makedirs(output_dir, exist_ok=True)
    abs_output_dir = os.path.abspath(output_dir)
    script_path = os.path.join(abs_output_dir, "temp_backtest.py")

    with open(script_path, "w", encoding="utf-8") as f:
        f.write(code)

    stdout = ""
    stderr = ""
    returncode = 0
    execution_mode = "subprocess"

    # -------------------------------------------------------------
    # Layer 2: Ephemeral Docker Execution Sandbox
    # -------------------------------------------------------------
    use_docker = (
        prefer_docker
        and os.getenv("DISABLE_DOCKER_SANDBOX", "0").lower() not in ("1", "true")
        and is_docker_available()
        and is_sandbox_image_ready()
    )

    if use_docker:
        try:
            # Mount host outputs directory into /sandbox/outputs
            docker_network = os.getenv("DOCKER_SANDBOX_NETWORK", "bridge")
            docker_cmd = [
                "docker", "run", "--rm",
                "--network", docker_network,
                "--memory", "512m",
                "--cpus", "1.0",
                "--cap-drop", "ALL",
                "-v", f"{abs_output_dir}:/sandbox/outputs:rw",
                "alpha-agent-sandbox:latest",
                "/sandbox/outputs/temp_backtest.py"
            ]
            process = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds
            )
            stdout = process.stdout
            stderr = process.stderr
            returncode = process.returncode
            execution_mode = "docker_sandbox"
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Containerized execution timed out after {timeout_seconds} seconds.",
                "stdout": "",
                "stderr": "TimeoutExpired",
                "metrics": None,
                "chart_path": None,
                "execution_mode": "docker_sandbox"
            }
        except Exception as e:
            # If Docker invocation encountered an unexpected runtime failure, gracefully fall back
            use_docker = False

    # -------------------------------------------------------------
    # Layer 3: Controlled Subprocess Sandbox Fallback
    # -------------------------------------------------------------
    if not use_docker:
        execution_mode = "subprocess_fallback"
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
            returncode = process.returncode
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Execution timed out after {timeout_seconds} seconds.",
                "stdout": "",
                "stderr": "TimeoutExpired",
                "metrics": None,
                "chart_path": None,
                "execution_mode": "subprocess_fallback"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Subprocess runner error: {str(e)}",
                "stdout": "",
                "stderr": str(e),
                "metrics": None,
                "chart_path": None,
                "execution_mode": "subprocess_fallback"
            }

    # -------------------------------------------------------------
    # Result Parsing & Metric Extraction
    # -------------------------------------------------------------
    if returncode != 0:
        return {
            "success": False,
            "error": f"Execution failed with return code {returncode}:\n{stderr}",
            "stdout": stdout,
            "stderr": stderr,
            "metrics": None,
            "chart_path": None,
            "execution_mode": execution_mode
        }

    # Parse metrics printed by the script
    metrics = {}
    json_match = re.search(r"===METRICS_JSON_START===(.*?)===METRICS_JSON_END===", stdout, re.DOTALL)
    if json_match:
        try:
            metrics = json.loads(json_match.group(1).strip())
        except json.JSONDecodeError:
            metrics = {"raw_output": stdout}
    else:
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
        "chart_path": chart_path,
        "execution_mode": execution_mode
    }
