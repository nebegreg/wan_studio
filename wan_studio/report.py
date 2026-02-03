from __future__ import annotations

import os
import csv
from typing import Optional, Dict, Any
from datetime import datetime
from PIL import Image

from .config import ProjectConfig
from .utils import safe_makedirs, write_json

def export_shotlist_csv(cfg: ProjectConfig, out_csv: str) -> None:
    safe_makedirs(os.path.dirname(os.path.abspath(out_csv)))
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "label", "tags", "seconds", "prompt", "negative_prompt", "notes"])
        for i, s in enumerate(cfg.scenes, start=1):
            w.writerow([i, s.label, s.tags, s.seconds, s.prompt, s.negative_prompt or "", s.notes])

def _copy_thumb(frames_dir: str, idx: int, out_path: str) -> Optional[str]:
    p = os.path.join(frames_dir, f"frame_{idx:06d}.png")
    if not os.path.exists(p):
        return None
    img = Image.open(p).convert("RGB")
    img.thumbnail((960, 540))
    img.save(out_path, "PNG")
    return out_path

def build_thumbnails(frames_dir: str, out_dir: str, last_frame_idx: int) -> Dict[str, str]:
    safe_makedirs(out_dir)
    thumbs: Dict[str, str] = {}
    if last_frame_idx <= 0:
        return thumbs
    thumbs["first"] = _copy_thumb(frames_dir, 1, os.path.join(out_dir, "thumb_first.png")) or ""
    mid = max(1, last_frame_idx // 2)
    thumbs["mid"] = _copy_thumb(frames_dir, mid, os.path.join(out_dir, "thumb_mid.png")) or ""
    thumbs["last"] = _copy_thumb(frames_dir, last_frame_idx, os.path.join(out_dir, "thumb_last.png")) or ""
    return thumbs

def create_html_report(cfg: ProjectConfig, proj_dir: str, outputs: Dict[str, str], thumbs: Dict[str, str], extra: Dict[str, Any]) -> str:
    safe_makedirs(proj_dir)
    report_path = os.path.join(proj_dir, "report.html")
    dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    write_json(os.path.join(proj_dir, "report.json"), {
        "generated_at": dt,
        "project": cfg.to_dict(),
        "outputs": outputs,
        "thumbs": thumbs,
        "extra": extra,
    })

    def rel(p: str) -> str:
        try:
            return os.path.relpath(p, proj_dir)
        except Exception:
            return p

    thumbs_html = ""
    for k in ["first", "mid", "last"]:
        if k in thumbs and thumbs[k]:
            thumbs_html += f'<div class="thumb"><div class="tlabel">{k}</div><img src="{rel(thumbs[k])}"/></div>'

    outs_html = ""
    for k, p in outputs.items():
        if not p:
            continue
        outs_html += f'<li><b>{k}</b>: <a href="{rel(p)}">{rel(p)}</a></li>'

    scenes_rows = ""
    for i, s in enumerate(cfg.scenes, start=1):
        scenes_rows += f"<tr><td>{i}</td><td>{s.label}</td><td>{s.tags}</td><td>{s.seconds}</td><td>{s.prompt}</td></tr>"

    html = f'''<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>Wan Studio Report - {cfg.project_name}</title>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu; margin:24px; background:#0b1220; color:#e5e7eb}}
a{{color:#93c5fd}}
.card{{background:#111827; border:1px solid #1f2937; border-radius:16px; padding:16px; margin:16px 0}}
.grid{{display:flex; gap:12px; flex-wrap:wrap}}
.thumb{{background:#0b1220; border:1px solid #1f2937; border-radius:12px; padding:8px}}
.thumb img{{display:block; border-radius:8px; max-width:480px; height:auto}}
.tlabel{{font-size:12px; opacity:.8; margin-bottom:6px}}
table{{width:100%; border-collapse:collapse}}
th,td{{border-bottom:1px solid #1f2937; padding:8px; vertical-align:top}}
th{{text-align:left; opacity:.85}}
code{{background:#0b1220; padding:2px 6px; border-radius:8px}}
</style>
</head>
<body>
<h1>Wan Studio Report</h1>
<div class="card">
  <div><b>Project</b>: {cfg.project_name}</div>
  <div><b>Generated</b>: {dt}</div>
  <div><b>Model</b>: <code>{cfg.model_id}</code> | Mode: <code>{cfg.mode}</code></div>
  <div><b>Base</b>: {cfg.width}x{cfg.height} @ {cfg.fps}fps | Chunk {cfg.chunk_seconds}s | Overlap {cfg.overlap_frames}</div>
</div>

<div class="card">
  <h2>Thumbnails</h2>
  <div class="grid">{thumbs_html}</div>
</div>

<div class="card">
  <h2>Outputs</h2>
  <ul>{outs_html}</ul>
</div>

<div class="card">
  <h2>Shots</h2>
  <table>
    <thead><tr><th>#</th><th>Label</th><th>Tags</th><th>Seconds</th><th>Prompt</th></tr></thead>
    <tbody>{scenes_rows}</tbody>
  </table>
</div>

<div class="card">
  <h2>Notes</h2>
  <div>Shotlist CSV: <code>shotlist.csv</code> (if enabled)</div>
  <div>Machine report: <code>report.json</code></div>
</div>
</body>
</html>'''
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    return report_path
