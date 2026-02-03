# Wan Studio smoke tests

Quick sanity checks you can run on Rocky Linux (no GPU required):

```bash
cd wan_studio_cinema_clean_v1
python -m smoke_tests.run_all
```

What it checks:
- Config dataclasses roundtrip (to_dict/from_dict)
- Storyboard apply keeps object identity (in-place update) + respects locks
- Init-frame cache key stability
- Clip render cache key stability
- FFmpeg postprocess filter graph string build
