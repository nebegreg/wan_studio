from __future__ import annotations


def run() -> None:
    from wan_studio.postprocess import PostProcessConfig, _ffmpeg_filter

    cfg = PostProcessConfig(deflicker=True, denoise=True, sharpen=True)
    f = _ffmpeg_filter(cfg)
    assert "deflicker" in f
    assert "hqdn3d" in f
    assert "unsharp" in f

    cfg2 = PostProcessConfig(deflicker=False, denoise=False, sharpen=False)
    f2 = _ffmpeg_filter(cfg2)
    assert f2 == "null"
