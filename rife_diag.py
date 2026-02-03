#!/usr/bin/env python3
import os, subprocess, sys
from wan_studio.postprocess import detect_tools

def main():
    tools_dir = os.path.abspath("./tools")
    st = detect_tools(tools_dir)
    print("tools_dir:", tools_dir)
    print("ffmpeg:", st.ffmpeg)
    print("rife:", st.rife)
    if not st.rife:
        print("❌ RIFE not found. Run: python get_tools.py rife-ncnn-vulkan")
        return 2
    bin_dir = os.path.dirname(st.rife)
    models = [d for d in os.listdir(bin_dir) if d.startswith("rife-v") and os.path.isdir(os.path.join(bin_dir,d))]
    print("models:", models)
    try:
        r = subprocess.run([st.rife, "-h"], cwd=bin_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print("---- rife -h ----")
        print(r.stdout[:2000])
    except Exception as e:
        print("⚠️ Could not run rife -h:", e)
    print("✅ RIFE detected. If interpolation still fails, ensure each rife-v* folder contains a matching *.param + *.bin (e.g. flownet.param/bin or rife.param/bin).")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
