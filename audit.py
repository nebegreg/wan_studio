#!/usr/bin/env python3
from __future__ import annotations
import py_compile, re, ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent

SKIP_DIRS = {
    ".venv", "venv", "__pycache__", ".git", ".mypy_cache", ".pytest_cache",
    "outputs", "tools", "build", "dist"
}

def _should_skip(path: Path) -> bool:
    # Skip if any parent directory is in SKIP_DIRS
    for part in path.parts:
        if part in SKIP_DIRS:
            return True
    s = str(path)
    if "site-packages" in s:
        return True
    return False

def compile_all():
    errs = []
    for p in ROOT.rglob("*.py"):
        if _should_skip(p):
            continue
        try:
            py_compile.compile(str(p), doraise=True)
        except Exception as e:
            errs.append((str(p.relative_to(ROOT)), str(e)))
    return errs

def connect_sanity(gui_py: Path):
    txt = gui_py.read_text(encoding="utf-8")
    tree = ast.parse(txt)
    mw = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MainWindow"), None)
    if not mw:
        return ["MainWindow class not found"]

    methods = {n.name for n in mw.body if isinstance(n, ast.FunctionDef)}
    lines = txt.splitlines()
    block = "\n".join(lines[mw.lineno - 1: mw.end_lineno])
    refs = re.findall(r"\.connect\(\s*self\.([A-Za-z_]\w*)\s*\)", block)

    ignore = {"close", "accept", "reject"}
    missing = sorted({r for r in refs if r not in methods and r not in ignore})
    return missing

def main():
    errs = compile_all()
    if errs:
        print("❌ Compile errors:")
        for e in errs[:30]:
            print("  ", e)
        raise SystemExit(1)

    gui = ROOT / "wan_studio" / "gui.py"
    missing = connect_sanity(gui)
    if missing:
        print("❌ Missing MainWindow slot(s) referenced by connect():")
        for m in missing:
            print("  ", m)
        raise SystemExit(2)

    print("✅ Audit OK (compile + MainWindow Qt connects).")

if __name__ == "__main__":
    main()
