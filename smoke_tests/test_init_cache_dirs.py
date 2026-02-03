from wan_studio.config import ProjectConfig
from wan_studio.flux2_init import get_init_cache_dirs


def test_get_init_cache_dirs_assets(tmp_path):
    cfg = ProjectConfig()
    cfg.output_dir = str(tmp_path)
    cfg.project_name = "TEST"
    cfg.init_image.cache_location = "assets"
    init_dir, cache_json = get_init_cache_dirs(cfg)
    assert "init_frames" in init_dir
    assert cache_json.endswith("init_frames_cache.json")


def test_get_init_cache_dirs_project(tmp_path):
    cfg = ProjectConfig()
    cfg.output_dir = str(tmp_path)
    cfg.project_name = "TEST"
    cfg.init_image.cache_location = "project"
    init_dir, cache_json = get_init_cache_dirs(cfg)
    assert "init_frames" in init_dir
    assert cache_json.endswith("init_frames_cache.json")


def run() -> None:
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        test_get_init_cache_dirs_assets(p)
        test_get_init_cache_dirs_project(p)
