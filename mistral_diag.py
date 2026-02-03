"""Tiny diagnostic for Mistral connectivity + JSON mode.

Usage:
  export MISTRAL_API_KEY=...
  python mistral_diag.py

If this fails, copy/paste the full output when asking for help.
"""

from __future__ import annotations

import os
import json

from wan_studio.mistral_storyboard import call_mistral_chat


def main() -> int:
    key = os.getenv("MISTRAL_API_KEY", "").strip()
    if not key:
        print("❌ MISTRAL_API_KEY is not set")
        return 2

    model = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
    print(f"Model: {model}")

    messages = [
        {"role": "system", "content": "Return only JSON."},
        {"role": "user", "content": "Return a JSON object with keys ok=true and ping=\"pong\"."},
    ]

    try:
        resp = call_mistral_chat(
            api_key=key,
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=60,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        print("❌ Request failed:", e)
        return 1

    # Try to print the model content cleanly
    content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
    print("Raw content:")
    print(content)
    try:
        obj = json.loads(content)
        print("✅ JSON parsed:", obj)
        return 0
    except Exception as e:
        print("⚠️ JSON parse failed:", e)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
