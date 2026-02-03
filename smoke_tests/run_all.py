from __future__ import annotations

import importlib
import os
import sys


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main() -> int:
    root = _project_root()
    if root not in sys.path:
        sys.path.insert(0, root)

    tests = [
        "smoke_tests.test_config_roundtrip",
        "smoke_tests.test_storyboard_apply_in_place",
        "smoke_tests.test_init_cache_key_stable",
        "smoke_tests.test_init_cache_dirs",
        "smoke_tests.test_clip_cache_key_stable",
        "smoke_tests.test_ffmpeg_filter_build",
    ]

    ok = 0
    fail = 0
    for mod_name in tests:
        try:
            mod = importlib.import_module(mod_name)
            run = getattr(mod, "run", None)
            if not callable(run):
                raise RuntimeError(f"{mod_name} has no run()")
            run()
            print(f"✅ {mod_name}")
            ok += 1
        except Exception as e:
            print(f"❌ {mod_name}: {e}")
            fail += 1

    print(f"\nSmoke tests: {ok} passed, {fail} failed")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
