import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools import set_root_dir, read_file, find, grep

def test_read_file_within_root(tmp_path):
    (tmp_path / "hello.txt").write_text("hello world")
    set_root_dir(str(tmp_path))
    result = read_file.invoke({"path": "hello.txt"})
    assert result == "hello world"

def test_read_file_missing_returns_error_not_exception(tmp_path):
    set_root_dir(str(tmp_path))
    result = read_file.invoke({"path": "nope.txt"})
    assert "Error" in result or "not found" in result

def test_read_file_blocks_path_traversal(tmp_path):
    set_root_dir(str(tmp_path))
    result = read_file.invoke({"path": "../../etc/passwd"})
    assert "denied" in result.lower() or "error" in result.lower()

def test_find_matches_glob(tmp_path):
    (tmp_path / "a.py").write_text("x")
    (tmp_path / "b.txt").write_text("y")
    set_root_dir(str(tmp_path))
    result = find.invoke({"pattern": "*.py"})
    assert "a.py" in result
    assert "b.txt" not in result

def test_find_no_matches(tmp_path):
    set_root_dir(str(tmp_path))
    result = find.invoke({"pattern": "*.nonexistent"})
    assert "No files matched" in result