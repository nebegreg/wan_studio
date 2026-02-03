#!/usr/bin/env python3
from __future__ import annotations
import os, sys, stat, shutil, zipfile, tarfile, tempfile, urllib.request

TOOLS = {
    "rife-ncnn-vulkan": {
        "url": "https://github.com/nihui/rife-ncnn-vulkan/releases/download/20221029/rife-ncnn-vulkan-20221029-ubuntu.zip",
        "bin_names": ["rife-ncnn-vulkan"],
        "subdir": "rife",
    },
    "realesrgan-ncnn-vulkan": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip",
        "bin_names": ["realesrgan-ncnn-vulkan"],
        "subdir": "realesrgan",
    },
    "piper": {
        "url": "https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_amd64.tar.gz",
        "bin_names": ["piper"],
        "subdir": "piper",
    },
}

DEFAULT_PIPER_VOICE = os.environ.get("PIPER_VOICE", "fr_FR-upmc-medium")


def _piper_voice_urls(voice: str) -> tuple[str, str] | None:
    """Return (onnx_url, json_url) for a voice name like 'fr_FR-upmc-medium'."""
    try:
        locale, speaker, quality = voice.split("-", 2)
        lang = locale.split("_")[0]
        base = (
            "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
            f"{lang}/{locale}/{speaker}/{quality}/{voice}.onnx"
        )
        return base, base + ".json"
    except Exception:
        return None


def install_piper_voice(tools_dir: str, voice: str = DEFAULT_PIPER_VOICE) -> None:
    urls = _piper_voice_urls(voice)
    if not urls:
        print(f"  ⚠️ piper voice name not understood: {voice}")
        return
    onnx_url, json_url = urls
    voice_dir = os.path.join(tools_dir, "piper", "voices")
    os.makedirs(voice_dir, exist_ok=True)
    onnx_path = os.path.join(voice_dir, f"{voice}.onnx")
    json_path = os.path.join(voice_dir, f"{voice}.onnx.json")
    if not os.path.isfile(onnx_path):
        download(onnx_url, onnx_path)
    if not os.path.isfile(json_path):
        download(json_url, json_path)


def download(url: str, out_path: str):
    print(f"  downloading: {url}")
    with urllib.request.urlopen(url) as r, open(out_path, "wb") as f:
        total = int(r.headers.get("Content-Length", "0") or "0")
        got = 0
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            if total:
                pct = got * 100.0 / total
                print(f"\r  {pct:5.1f}% ({got/1e6:.1f}MB/{total/1e6:.1f}MB)", end="", flush=True)
        if total:
            print()

def extract(archive: str, dst: str):
    if archive.endswith(".zip"):
        with zipfile.ZipFile(archive, "r") as z:
            z.extractall(dst)
    elif archive.endswith(".tar.gz") or archive.endswith(".tgz"):
        with tarfile.open(archive, "r:gz") as t:
            t.extractall(dst)
    else:
        raise RuntimeError(f"Unknown archive type: {archive}")

def find_file(root: str, filename: str) -> str | None:
    for base, dirs, files in os.walk(root):
        if filename in files:
            return os.path.join(base, filename)
    return None

def install(tool_key: str, tools_dir: str):
    spec = TOOLS[tool_key]
    dst_dir = os.path.join(tools_dir, spec["subdir"])
    os.makedirs(dst_dir, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="wan_tools_") as td:
        # Keep the archive extension (zip vs tar.gz) so extract() can detect it.
        base_name = os.path.basename(spec["url"].split("?")[0]) or "tool.zip"
        arc = os.path.join(td, base_name)
        download(spec["url"], arc)
        ex = os.path.join(td, "extract")
        os.makedirs(ex, exist_ok=True)
        extract(arc, ex)

        bin_name = spec["bin_names"][0]
        found = find_file(ex, bin_name)
        if not found:
            raise RuntimeError(f"Binary not found: {bin_name}")
        dst = os.path.join(dst_dir, bin_name)
        shutil.copy2(found, dst)
        os.chmod(dst, os.stat(dst).st_mode | stat.S_IEXEC)

        # Copy model assets required by ncnn tools (folders located next to the binary in the release zip)
        bin_dir = os.path.dirname(found)
        if tool_key == "realesrgan-ncnn-vulkan":
            src_models = os.path.join(bin_dir, "models")
            if os.path.isdir(src_models):
                dst_models = os.path.join(dst_dir, "models")
                if os.path.exists(dst_models):
                    shutil.rmtree(dst_models, ignore_errors=True)
                shutil.copytree(src_models, dst_models)
                print(f"  assets: {dst_models}")
            else:
                print("  ⚠️ assets missing: models/ not found in archive")
        if tool_key == "rife-ncnn-vulkan":
            # RIFE models are shipped as rife-v* directories next to the binary
            for name in os.listdir(bin_dir):
                if name.startswith("rife-v") and os.path.isdir(os.path.join(bin_dir, name)):
                    src_m = os.path.join(bin_dir, name)
                    dst_m = os.path.join(dst_dir, name)
                    if os.path.exists(dst_m):
                        shutil.rmtree(dst_m, ignore_errors=True)
                    shutil.copytree(src_m, dst_m)
            # Also copy any models/ directory if present
            src_models = os.path.join(bin_dir, "models")
            if os.path.isdir(src_models):
                dst_models = os.path.join(dst_dir, "models")
                if os.path.exists(dst_models):
                    shutil.rmtree(dst_models, ignore_errors=True)
                shutil.copytree(src_models, dst_models)
        print(f"  installed: {dst}")

def main():
    tools_dir = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "./tools")
    os.makedirs(tools_dir, exist_ok=True)
    for key in TOOLS:
        print(f"==> {key}")
        try:
            install(key, tools_dir)
            if key == "piper":
                print(f"  voice: {DEFAULT_PIPER_VOICE}")
                try:
                    install_piper_voice(tools_dir, DEFAULT_PIPER_VOICE)
                except Exception as e:
                    print(f"  ⚠️ voice download failed: {e}")
        except Exception as e:
            print(f"  ⚠️ failed: {e}")
    print("Done.")

if __name__ == "__main__":
    main()
