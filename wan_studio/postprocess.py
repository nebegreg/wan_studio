from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional, Tuple

from .utils import which, safe_makedirs
import re

@dataclass
class ToolStatus:
    ffmpeg: Optional[str] = None
    rife: Optional[str] = None
    realesrgan: Optional[str] = None

def detect_tools(tools_dir: str) -> ToolStatus:
    # Be robust: user projects may carry an old absolute tools_dir from another folder.
    # We search in:
    # 1) cfg.tools_dir
    # 2) this project's local ./tools
    tools_dir = os.path.abspath(tools_dir)
    project_tools = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tools"))

    st = ToolStatus()
    st.ffmpeg = which("ffmpeg")

    def _iter_tool_roots():
        seen = set()
        for d in (tools_dir, project_tools):
            if not d:
                continue
            d = os.path.abspath(d)
            if d in seen:
                continue
            seen.add(d)
            yield d

    def _has_rife_models(bin_dir: str) -> bool:
        # RIFE models are shipped as rife-v* directories next to the binary.
        try:
            for name in os.listdir(bin_dir):
                if not name.startswith("rife-v"):
                    continue
                mdir = os.path.join(bin_dir, name)
                if not os.path.isdir(mdir):
                    continue
                # Assets naming differs across nihui releases.
                # Common variants include:
                # - rife.param / rife.bin
                # - flownet.param / flownet.bin
                # Some builds ship other *.param/*.bin pairs.
                has_param = any(f.endswith(".param") for f in os.listdir(mdir))
                has_bin = any(f.endswith(".bin") for f in os.listdir(mdir))
                if has_param and has_bin:
                    return True
        except Exception:
            return False
        return False

    # RIFE
    for root in _iter_tool_roots():
        for p in [
            os.path.join(root, "rife", "rife-ncnn-vulkan"),
            os.path.join(root, "rife-ncnn-vulkan"),
        ]:
            if os.path.exists(p) and os.access(p, os.X_OK):
                bin_dir = os.path.dirname(p)
                # Avoid picking a stale install that lacks model assets (common cause of ncnn blob errors).
                if _has_rife_models(bin_dir):
                    st.rife = p
                    break
        if st.rife:
            break

    # Real-ESRGAN
    for root in _iter_tool_roots():
        for p in [
            os.path.join(root, "realesrgan", "realesrgan-ncnn-vulkan"),
            os.path.join(root, "realesrgan-ncnn-vulkan"),
        ]:
            if os.path.exists(p) and os.access(p, os.X_OK):
                st.realesrgan = p
                break
        if st.realesrgan:
            break

    return st
@dataclass
class PostProcessConfig:
    enabled: bool = True
    tools_dir: str = "./tools"
    use_rife_if_available: bool = True
    use_realesrgan_if_available: bool = True

    fps_in: int = 24
    target_fps: int = 48
    out_width: int = 1920
    out_height: int = 1080

    deflicker: bool = True
    denoise: bool = True
    denoise_strength: Tuple[float, float, float, float] = (1.5, 1.0, 3.0, 2.0)
    sharpen: bool = True
    unsharp: Tuple[int, int, float] = (5, 5, 0.8)

    export_master_prores: bool = True
    export_delivery_h265_main10: bool = True
    h265_crf: int = 18
    h265_preset: str = "slow"

def _ffmpeg_filter(cfg: PostProcessConfig) -> str:
    f = []
    if cfg.deflicker:
        f.append("deflicker=s=5:m=am")
    if cfg.denoise:
        l, c, tl, tc = cfg.denoise_strength
        f.append(f"hqdn3d={l}:{c}:{tl}:{tc}")
    if cfg.sharpen:
        mx, my, amt = cfg.unsharp
        f.append(f"unsharp={mx}:{my}:{amt}")
    return ",".join(f) if f else "null"

def ffmpeg_polish(in_path: str, out_path: str, cfg: PostProcessConfig) -> None:
    ffmpeg = which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg introuvable.")
    safe_makedirs(os.path.dirname(os.path.abspath(out_path)))
    filt = _ffmpeg_filter(cfg)
    cmd = [
        ffmpeg, "-y",
        "-i", in_path,
        "-vf", filt,
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        out_path
    ]
    subprocess.run(cmd, check=True)

def ffmpeg_scale(in_path: str, out_path: str, w: int, h: int, fps: int, keep_aspect: bool = True) -> None:
    """Scale to a target canvas.

    keep_aspect=True prevents stretching by preserving the original aspect ratio and padding to fit.
    This reduces 'blended edges' / warped geometry when upscaling longform clips with slight AR drift.
    """
    ffmpeg = which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg introuvable.")
    safe_makedirs(os.path.dirname(os.path.abspath(out_path)))

    if keep_aspect:
        vf = f"scale={w}:{h}:flags=lanczos:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black"
    else:
        vf = f"scale={w}:{h}:flags=lanczos"

    cmd = [
        ffmpeg, "-y",
        "-i", in_path,
        "-r", str(int(fps)),
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        out_path
    ]
    subprocess.run(cmd, check=True)

def ffmpeg_interpolate(in_path: str, out_path: str, fps: int) -> None:
    ffmpeg = which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg introuvable.")
    safe_makedirs(os.path.dirname(os.path.abspath(out_path)))
    cmd = [
        ffmpeg, "-y",
        "-i", in_path,
        "-vf", f"minterpolate=fps={fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1",
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        out_path
    ]
    subprocess.run(cmd, check=True)

def _extract_frames(in_path: str, frames_dir: str) -> None:
    ffmpeg = which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg introuvable.")
    safe_makedirs(frames_dir)
    subprocess.run([ffmpeg, "-y", "-i", in_path, os.path.join(frames_dir, "%06d.png")], check=True)

def _encode_frames(frames_dir: str, out_path: str, fps: int) -> None:
    ffmpeg = which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg introuvable.")
    safe_makedirs(os.path.dirname(os.path.abspath(out_path)))
    subprocess.run([ffmpeg, "-y", "-framerate", str(int(fps)), "-i", os.path.join(frames_dir, "%06d.png"),
                    "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path], check=True)

def rife_interpolate(rife_bin: str, in_path: str, out_path: str, src_fps: int, target_fps: int) -> None:
    # RIFE (ncnn-vulkan) compatibility:
    # - Many builds only support 2x per pass.
    # - For 4x/8x/... we chain multiple 2x passes.
    ratio = 0
    try:
        if src_fps > 0 and target_fps > src_fps and target_fps % src_fps == 0:
            ratio = target_fps // src_fps
    except Exception:
        ratio = 0

    # Helper: one 2x pass (extract frames -> rife -> encode)
    def _rife_pass(in_mp4: str, out_mp4: str, in_fps: int, out_fps: int) -> None:
        tmp = os.path.join(os.path.dirname(out_mp4), "_rife_tmp")
        inp = os.path.join(tmp, "in")
        outp = os.path.join(tmp, "out")
        shutil.rmtree(tmp, ignore_errors=True)
        _extract_frames(in_mp4, inp)
        safe_makedirs(outp)

        # Pick the newest available rife-v* model directory explicitly (prevents ncnn blob-index errors)
        model_dir = None
        try:
            bin_dir = os.path.dirname(rife_bin)
            candidates = []
            for name in os.listdir(bin_dir):
                if name.startswith('rife-v') and os.path.isdir(os.path.join(bin_dir, name)):
                    candidates.append(name)
            def _ver_key(n: str):
                m = re.findall(r"\d+", n)
                return tuple(int(x) for x in m) if m else (0,)
            candidates.sort(key=_ver_key, reverse=True)
            if candidates:
                model_dir = os.path.join(bin_dir, candidates[0])
        except Exception:
            model_dir = None

        cmd = [rife_bin, "-i", inp, "-o", outp]
        if model_dir:
            cmd += ["-m", model_dir]
        # Try enabling -n 2 (some builds support it). We'll retry without -n if needed.
        cmd_n = cmd + ["-n", "2"]
        try:
            subprocess.run(cmd_n, check=True, cwd=os.path.dirname(rife_bin), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        except subprocess.CalledProcessError as e:
            out = (e.stdout or "") if hasattr(e, "stdout") else ""
            if "only rife-v4" in out.lower() or "custom numframe" in out.lower():
                subprocess.run(cmd, check=True, cwd=os.path.dirname(rife_bin))
            else:
                raise

        _encode_frames(outp, out_mp4, fps=out_fps)
        shutil.rmtree(tmp, ignore_errors=True)

    # Validate ratio: allow powers of two (2,4,8,16...).
    if ratio < 2:
        raise RuntimeError(f"RIFE interpolation ratio invalid (src_fps={src_fps}, target_fps={target_fps}).")
    if ratio & (ratio - 1) != 0:
        # Not a power of two (e.g., 24->60). Let caller fall back.
        raise RuntimeError(f"RIFE supports only power-of-two ratios via chaining (src_fps={src_fps}, target_fps={target_fps}).")

    # Chain 2x passes until we reach target_fps.
    cur_in = in_path
    cur_fps = int(src_fps)
    intermediates = []
    try:
        while cur_fps < int(target_fps):
            nxt_fps = cur_fps * 2
            tmp_out = out_path if nxt_fps == int(target_fps) else os.path.join(os.path.dirname(out_path), f"_rife_{nxt_fps}fps.mp4")
            if tmp_out != out_path:
                intermediates.append(tmp_out)
            _rife_pass(cur_in, tmp_out, cur_fps, nxt_fps)
            cur_in = tmp_out
            cur_fps = nxt_fps
    finally:
        for p in intermediates:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

def realesrgan_upscale(re_bin: str, in_path: str, out_path: str, fps: int, scale: int = 2) -> None:
    tmp = os.path.join(os.path.dirname(out_path), "_re_tmp")
    inp = os.path.join(tmp, "in")
    outp = os.path.join(tmp, "out")
    _extract_frames(in_path, inp)
    safe_makedirs(outp)
    r = subprocess.run([re_bin, "-i", inp, "-o", outp, "-s", str(scale), "-t", "256"], cwd=os.path.dirname(re_bin), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = r.stdout or ""
    # Detect common ncnn model/asset mismatch (spam: find_blob_index_by_name ... failed)
    if "find_blob_index_by_name" in out.lower() or r.returncode != 0:
        raise RuntimeError(f"Real-ESRGAN failed or missing assets. Output:\n{out[-2000:]}")
    _encode_frames(outp, out_path, fps=fps)
    shutil.rmtree(tmp, ignore_errors=True)

def postprocess_pipeline(in_raw: str, out_dir: str, cfg: PostProcessConfig, log=None) -> dict:
    safe_makedirs(out_dir)
    tools = detect_tools(cfg.tools_dir)
    def _log(msg: str):
        try:
            if log:
                log(msg)
        except Exception:
            pass
    steps = {}
    cur = in_raw

    polished = os.path.join(out_dir, "polished.mp4")
    ffmpeg_polish(cur, polished, cfg)
    steps["polished"] = polished
    cur = polished

    fps_work = int(cfg.fps_in)

    if cfg.target_fps and cfg.target_fps > 0 and cfg.target_fps != fps_work:
        interp = os.path.join(out_dir, f"interp_{cfg.target_fps}fps.mp4")
        if cfg.use_rife_if_available and tools.rife:
            try:
                _log(f"RIFE: {tools.rife}")
                rife_interpolate(tools.rife, cur, interp, src_fps=fps_work, target_fps=cfg.target_fps)
            except Exception as e:
                _log(f"⚠️ RIFE failed ({type(e).__name__}: {e}). Falling back to ffmpeg interpolation.")
                try:
                    out = getattr(e, 'stdout', None) or ''
                    if out:
                        _log(out[-2000:])
                except Exception:
                    pass
                ffmpeg_interpolate(cur, interp, cfg.target_fps)
        else:
            ffmpeg_interpolate(cur, interp, cfg.target_fps)
        steps["interpolated"] = interp
        cur = interp
        fps_work = int(cfg.target_fps)

    if cfg.out_width and cfg.out_height and cfg.out_width > 0 and cfg.out_height > 0:
        up = os.path.join(out_dir, f"up_{cfg.out_width}x{cfg.out_height}.mp4")
        if cfg.use_realesrgan_if_available and tools.realesrgan:
            tmp_up = os.path.join(out_dir, "up_realesrgan_x2.mp4")
            realesrgan_upscale(tools.realesrgan, cur, tmp_up, fps=fps_work, scale=2)
            ffmpeg_scale(tmp_up, up, cfg.out_width, cfg.out_height, fps=fps_work)
        else:
            ffmpeg_scale(cur, up, cfg.out_width, cfg.out_height, fps=fps_work)
        steps["upscaled"] = up
        cur = up

    return {
        "steps": steps,
        "exports": {"raw": in_raw, "intermediate": cur, "fps": fps_work},
        "tools": {"ffmpeg": tools.ffmpeg, "rife": tools.rife, "realesrgan": tools.realesrgan},
    }



def export_delivery_pack(ffmpeg_bin: str, src_mp4: str, out_dir: str, base_name: str = "delivery", prores_profile: str = "hq", log=None):
    """Create a studio-friendly delivery pack:
    - ProRes master (.mov)
    - H.265 review (.mp4)
    - ProRes Proxy (.mov)
    - Thumbnails strip (.jpg)
    """
    import os, subprocess, shlex
    os.makedirs(out_dir, exist_ok=True)

    prof_map = {
        "proxy": "0", "lt": "1", "standard": "2", "hq": "3", "4444": "4", "xq": "5"
    }
    prof = prof_map.get(prores_profile.lower(), "3")

    master_mov = os.path.join(out_dir, f"{base_name}_master_prores.mov")
    proxy_mov  = os.path.join(out_dir, f"{base_name}_proxy_prores.mov")
    review_mp4 = os.path.join(out_dir, f"{base_name}_review_h265.mp4")
    thumbs_jpg = os.path.join(out_dir, f"{base_name}_thumbs.jpg")

    # ProRes master (prores_ks)
    cmd_master = [ffmpeg_bin, "-y", "-i", src_mp4, "-c:v", "prores_ks", "-profile:v", prof, "-pix_fmt", "yuv422p10le", "-c:a", "pcm_s16le", master_mov]
    # ProRes proxy
    cmd_proxy  = [ffmpeg_bin, "-y", "-i", src_mp4, "-c:v", "prores_ks", "-profile:v", "0", "-pix_fmt", "yuv422p10le", "-c:a", "pcm_s16le", proxy_mov]
    # H.265 review (Main10 if possible)
    cmd_review = [ffmpeg_bin, "-y", "-i", src_mp4, "-c:v", "libx265", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "slow", "-tag:v", "hvc1", "-c:a", "aac", "-b:a", "192k", review_mp4]
    # Thumbnails strip (1 frame every ~2 seconds, max 32 thumbs in one strip)
    cmd_thumbs = [ffmpeg_bin, "-y", "-i", src_mp4, "-vf", "fps=1/2,scale=640:-1,tile=8x4", "-frames:v", "1", thumbs_jpg]

    for cmd in (cmd_master, cmd_proxy, cmd_review, cmd_thumbs):
        if log: log("Export: " + " ".join(shlex.quote(c) for c in cmd))
        subprocess.run(cmd, check=True)

    return {"master_prores": master_mov, "proxy_prores": proxy_mov, "review_h265": review_mp4, "thumbs": thumbs_jpg}