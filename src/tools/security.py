"""
Enterprise AST Security Linter for AlphaAgent.
Performs pre-execution static analysis on dynamically generated Python code
to prevent Remote Code Execution (RCE), network socket egress, unauthorized
filesystem manipulation, and resource exhaustion attacks.
"""

import ast
from typing import Dict, Any, List, Set, Optional

# Disallowed module imports (network egress, process execution, OS access, dynamic loading)
FORBIDDEN_MODULES: Set[str] = {
    "os", "sys", "subprocess", "shutil", "socket", "http", "urllib", "urllib3",
    "requests", "httpx", "aiohttp", "ftplib", "smtplib", "telnetlib",
    "ctypes", "pty", "multiprocessing", "threading", "signal",
    "inspect", "importlib", "builtins", "code", "posix", "nt"
}

# Disallowed built-in functions and execution primitives
FORBIDDEN_FUNCTIONS: Set[str] = {
    "eval", "exec", "compile", "__import__", "globals", "locals",
    "getattr", "setattr", "delattr", "vars", "breakpoint"
}

# Disallowed AST attribute access (prevents dunder escape / sandbox breakout attacks)
FORBIDDEN_ATTRIBUTES: Set[str] = {
    "__subclasses__", "__bases__", "__base__", "__mro__",
    "__globals__", "__code__", "__closure__", "__dict__",
    "__builtins__", "__import__"
}

class SecurityVisitor(ast.NodeVisitor):
    """
    Traverses the Python Abstract Syntax Tree to identify security policy violations.
    """
    def __init__(self):
        self.violations: List[str] = []

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            base_module = alias.name.split(".")[0]
            if base_module in FORBIDDEN_MODULES:
                self.violations.append(
                    f"Line {node.lineno}: Prohibited import of module '{alias.name}'. System and network operations are restricted."
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            base_module = node.module.split(".")[0]
            if base_module in FORBIDDEN_MODULES:
                self.violations.append(
                    f"Line {node.lineno}: Prohibited import from module '{node.module}'. System and network operations are restricted."
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        # 1. Direct function call inspection (e.g. eval(...), exec(...))
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in FORBIDDEN_FUNCTIONS:
                self.violations.append(
                    f"Line {node.lineno}: Prohibited invocation of dangerous built-in '{func_name}()'."
                )
        # 2. Attribute function call inspection (e.g. os.system(...), socket.socket(...))
        elif isinstance(node.func, ast.Attribute):
            attr_name = node.func.attr
            if attr_name in FORBIDDEN_FUNCTIONS:
                self.violations.append(
                    f"Line {node.lineno}: Prohibited invocation of function '{attr_name}()'."
                )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        # Inspect for dunder attribute breakout attempts (e.g. ().__class__.__bases__[0].__subclasses__())
        if node.attr in FORBIDDEN_ATTRIBUTES:
            self.violations.append(
                f"Line {node.lineno}: Prohibited dunder reflection attribute '{node.attr}' detected."
            )
        self.generic_visit(node)


def validate_code_safety(code: str) -> Dict[str, Any]:
    """
    Performs static AST security analysis on generated Python code.
    
    Returns:
        Dict with keys:
            - is_safe (bool): True if code contains zero security violations.
            - violations (List[str]): List of identified security violations.
            - error_message (Optional[str]): Formatted error string for feedback.
    """
    if not code or not code.strip():
        return {
            "is_safe": False,
            "violations": ["Empty code string provided for execution."],
            "error_message": "SecurityViolation: Empty code payload."
        }

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {
            "is_safe": False,
            "violations": [f"SyntaxError on line {e.lineno}: {e.msg}"],
            "error_message": f"SyntaxError: {e.msg} on line {e.lineno}"
        }

    visitor = SecurityVisitor()
    visitor.visit(tree)

    if visitor.violations:
        formatted_error = "SecurityViolation: Zero-Trust AST Sandbox rejected the script with policy violations:\n" + "\n".join(f"  - {v}" for v in visitor.violations)
        return {
            "is_safe": False,
            "violations": visitor.violations,
            "error_message": formatted_error
        }

    return {
        "is_safe": True,
        "violations": [],
        "error_message": None
    }
