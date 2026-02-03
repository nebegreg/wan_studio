#!/usr/bin/env python3
from __future__ import annotations
import py_compile, re, ast, shutil, os
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

def _connect_sanity(path: Path):
    txt = path.read_text(encoding="utf-8")
    tree = ast.parse(txt)
    missing = []
    ignore = {"close", "accept", "reject"}

    for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
        methods = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
        lines = txt.splitlines()
        block = "\n".join(lines[cls.lineno - 1: cls.end_lineno])
        refs = re.findall(r"\.connect\(\s*self\.([A-Za-z_]\w*)\s*\)", block)
        for r in sorted({r for r in refs if r not in methods and r not in ignore}):
            missing.append(f"{cls.name}.{r}")
    return missing


def check_dirs():
    needed = [
        ROOT / "outputs",
        ROOT / "outputs" / "assets",
        ROOT / "outputs" / "cache",
    ]
    missing = [p for p in needed if not p.exists()]
    return missing


def check_ffmpeg():
    return shutil.which("ffmpeg")

def main():
    errs = compile_all()
    if errs:
        print("❌ Compile errors:")
        for e in errs[:30]:
            print("  ", e)
        raise SystemExit(1)

    missing_slots = []
    for rel in ("wan_studio/gui.py", "wan_studio/studio_premiere.py", "wan_studio/settings_dialogs.py"):
        path = ROOT / rel
        if not path.exists():
            continue
        missing_slots.extend(_connect_sanity(path))

    if missing_slots:
        print("❌ Missing Qt slot(s) referenced by connect():")
        for m in missing_slots:
            print("  ", m)
        raise SystemExit(2)

    missing_dirs = check_dirs()
    if missing_dirs:
        print("⚠️ Missing folders (creating):")
        for p in missing_dirs:
            print("  ", p)
            try:
                os.makedirs(p, exist_ok=True)
            except Exception:
                pass

    ffmpeg = check_ffmpeg()
    if ffmpeg:
        print(f"✅ ffmpeg: {ffmpeg}")
    else:
        print("⚠️ ffmpeg: MISSING (fallbacks will be used)")

    print("✅ Audit OK (compile + Qt connects).")

if __name__ == "__main__":
    main()
