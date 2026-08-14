import os
import subprocess
from pathlib import Path
from langchain_core.tools import tool

# Root directory the agent is allowed to operate in.
# Set via set_root_dir() before the agent runs.
_ROOT_DIR = Path(".").resolve()

def set_root_dir(path: str) -> None:
    global _ROOT_DIR
    _ROOT_DIR = Path(path).resolve()

def _resolve_safe(path: str) -> Path:
    """Resolve a path and ensure it stays within _ROOT_DIR."""
    candidate = (_ROOT_DIR / path).resolve()
    if _ROOT_DIR not in candidate.parents and candidate != _ROOT_DIR:
        raise ValueError(f"Access denied: '{path}' is outside root dir")
    return candidate

@tool
def read_file(path: str) -> str:
    """Read and return the contents of a text file, given a path relative to the project root."""
    try:
        target = _resolve_safe(path)
        if not target.exists():
            return f"Error: file not found: {path}"
        if not target.is_file():
            return f"Error: not a file: {path}"
        return target.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error reading '{path}': {e}"

@tool
def find(pattern: str) -> str:
    """Find files matching a glob pattern (e.g. '*.py', '**/*.md') relative to the project root."""
    try:
        matches = sorted(str(p.relative_to(_ROOT_DIR)) for p in _ROOT_DIR.glob(pattern) if p.is_file())
        if not matches:
            return f"No files matched pattern: {pattern}"
        return "\n".join(matches)
    except Exception as e:
        return f"Error in find('{pattern}'): {e}"

@tool
def grep(query: str, path: str = ".") -> str:
    """Search for a text pattern inside files under the given path (relative to project root)."""
    try:
        target = _resolve_safe(path)
        result = subprocess.run(
            ["grep", "-rn", "--include=*.py", "--include=*.md", "--include=*.txt", query, str(target)],
            capture_output=True, text=True, timeout=10,
        )
        output = result.stdout.strip()
        if not output:
            return f"No matches for '{query}' in {path}"
        # Trim absolute prefix so output stays root-relative
        return output.replace(str(_ROOT_DIR) + os.sep, "")
    except Exception as e:
        return f"Error in grep('{query}', '{path}'): {e}"

ALL_TOOLS = [read_file, find, grep]