from __future__ import annotations

import json
import os
import re
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


MISTRAL_CHAT_URL = "https://api.mistral.ai/v1/chat/completions"


def _estimate_max_tokens(num_shots: int, *, include_dialogue: bool = False) -> int:
    """Heuristic to avoid truncated JSON (common cause of JSONDecodeError).

    - Can be overridden via env var MISTRAL_STORYBOARD_MAX_TOKENS.
    """
    # user override
    env = os.getenv("MISTRAL_STORYBOARD_MAX_TOKENS")
    if env:
        try:
            return max(512, int(env))
        except Exception:
            pass

    # Rough budget: prompts + tags + neg + notes per shot
    per_shot = 320 + (80 if include_dialogue else 0)
    base = 900
    mt = base + int(max(1, num_shots)) * per_shot
    # Keep within reasonable bounds (depends on model context, but 8k is usually safe on large models)
    return int(min(8000, max(2048, mt)))




@dataclass
class StoryboardShot:
    label: str
    seconds: float
    tags: str = ""
    prompt: str = ""
    negative_prompt: str = ""
    notes: str = ""

    # Phase C (cinéma)
    characters: List[str] = field(default_factory=list)
    location: str = ""
    hard_cut: bool = False

    # Structured shot hints
    shot_type: str = ""
    camera_move: str = ""
    mood: str = ""
    music_intensity: float = 0.5

    mode: str = ""          # "T2V" or "I2V" (optional)
    model_id: str = ""      # optional hint

    # Optional dialogue/narration attached to this shot (used by Audio tab / TTS).
    dialogue: str = ""
    speaker: str = ""
    dialogue_language: str = ""
    dialogue_voice: str = ""


@dataclass
class Storyboard:
    title: str
    logline: str
    style: str
    shots: List[StoryboardShot]


def _extract_json_object(text: str) -> Dict[str, Any]:
    """
    Robust JSON extractor:
    - if response is JSON already: parse directly
    - else find ```json ... ``` block
    - else take first {...} span
    """
    text = (text or "").strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    i = text.find("{")
    j = text.rfind("}")
    if i != -1 and j != -1 and j > i:
        try:
            return json.loads(text[i : j + 1])
        except Exception:
            pass

    # Best-effort repair
    repaired = _repair_json_like(text)
    if repaired:
        obj = json.loads(repaired)
        if isinstance(obj, dict):
            return obj

    raise ValueError("Could not parse JSON object from model output.")


def _find_balanced_object(text: str) -> Optional[str]:
    """Return the first balanced {...} JSON object substring found in text.

    Handles braces in strings heuristically. Returns None if no balanced object.
    """
    if not text:
        return None
    s = text
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start : i + 1]
    return None


def _repair_json_like(text: str) -> str:
    """Best-effort repair for "almost JSON" outputs.

    - strips markdown fences
    - removes trailing commas
    - removes ASCII control chars
    """
    if not text:
        return ""
    t = text.strip()
    # unwrap ```json fences
    m = re.search(r"```(?:json)?\s*(\{.*)```", t, flags=re.S)
    if m:
        t = m.group(1).strip()
    span = _find_balanced_object(t)
    if span:
        t = span
    # remove trailing commas before ] or }
    t = re.sub(r",\s*([\]}])", r"\1", t)
    # remove control chars except \n\r\t
    t = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", t)
    return t


def call_mistral_chat(
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    *,
    temperature: float = 0.6,
    max_tokens: int = 2500,
    timeout_s: int = 120,
    response_format: Optional[Dict[str, Any]] = None,
    retries: int = 2,
    backoff_s: float = 1.5,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    if response_format is not None:
        payload["response_format"] = response_format

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        MISTRAL_CHAT_URL,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "WanStudio/14 (Storyboard; Python urllib)",
        },
        method="POST",
    )

    last_err: Optional[Exception] = None
    for attempt in range(int(retries) + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                return json.loads(raw)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
            msg = f"Mistral HTTP {e.code}: {body[:1200]}"

            # Common transient errors: retry a couple times
            if e.code in (408, 409, 425, 429, 500, 502, 503, 504) and attempt < int(retries):
                last_err = RuntimeError(msg)
                time.sleep(float(backoff_s) * (2 ** attempt))
                continue

            raise RuntimeError(msg) from e
        except Exception as e:
            last_err = e
            if attempt < int(retries):
                time.sleep(float(backoff_s) * (2 ** attempt))
                continue
            raise RuntimeError(f"Mistral request failed: {e}") from e

    # Shouldn't reach here
    raise RuntimeError(f"Mistral request failed: {last_err}")


def _safe_float(x: Any, default: float = 3.0) -> float:
    try:
        return float(x)
    except Exception:
        s = str(x or "").strip()
        m = re.search(r"(-?\d+(?:\.\d+)?)", s)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
    return float(default)


def generate_storyboard(
    brief: str,
    total_seconds: float,
    fps: int,
    num_shots: int,
    *,
    style: str = "publicité premium cinématique",
    language: str = "fr",
    model: str = "mistral-large-latest",
    api_key: Optional[str] = None,
    project_hint: Optional[Dict[str, Any]] = None,
    temperature: float = 0.6,
) -> Storyboard:
    """
    Returns a structured storyboard suitable for Wan Studio timelines.
    Uses Mistral JSON mode to improve parseability.
    """
    api_key = api_key or os.getenv("MISTRAL_API_KEY", "")
    if not api_key:
        raise RuntimeError("Missing Mistral API key. Set MISTRAL_API_KEY env var or paste it in the dialog.")

    project_hint = project_hint or {}
    hint_txt = json.dumps(project_hint, ensure_ascii=False)

    # Avoid truncated JSON (common cause of JSONDecodeError).
    max_tokens_storyboard = _estimate_max_tokens(int(num_shots or 0) or 1, include_dialogue=True)
    max_tokens_repair = max(4000, int(max_tokens_storyboard))



    schema_desc = {
        "title": "string",
        "logline": "string",
        "style": "string",
        "shots": [
            {
                "label": "string (e.g. S01_SH01)",
                "seconds": "number (float)",
                "tags": "string (comma-separated: EXT,NIGHT,WS etc.)",
                "prompt": "string (Wan prompt, include camera move, realism cues)",
                "negative_prompt": "string (anti-artefact / anti-text issues)",
                "notes": "string (shot notes for editor)",
                "characters": "array of strings (character names) — can be []",
                "location": "string (location name) — can be empty",
                "hard_cut": "boolean (true to reset continuity)",
                "shot_type": "string optional (e.g. close-up, wide shot)",
                "camera_move": "string optional (e.g. dolly-in, pan)",
                "mood": "string optional",
                "music_intensity": "number 0..1 optional",
                "mode": "string optional: T2V or I2V",
                "model_id": "string optional: suggested model id",
                "dialogue": "string optional: one short line of dialogue/narration for this shot (can be empty)",
                "speaker": "string optional: who speaks (character name or NARRATOR)",
                "dialogue_language": "string optional: fr/en/... (if empty uses project default)",
                "dialogue_voice": "string optional: voice id/name (if empty uses character/global voice)",
            }
        ],
    }

    system = (
        "Tu es un réalisateur publicitaire senior et un superviseur storyboard. "
        "Tu dois produire un storyboard exploitable par un outil vidéo IA. "
        "Réponds UNIQUEMENT en JSON valide (un objet JSON), sans texte autour. Utilise des chaînes courtes. Aucun retour à la ligne brut dans les chaînes (utilise \\n si besoin)."
    )

    user = f"""
Langue: {language}
Style: {style}

Objectif:
- Écris une mini-histoire + storyboard pour une durée totale de {total_seconds:.1f} secondes à {fps} fps.
- Génère exactement {num_shots} plans (shots) avec des durées qui additionnent exactement {total_seconds:.1f}.
- Chaque shot doit contenir un prompt détaillé (photoréaliste, mouvement caméra, ambiance), un negative_prompt anti-artefacts, tags et notes.
- Si le brief implique du dialogue ou une voix off, remplis le champ "dialogue" pour les plans concernés (sinon ""). Mets UNE phrase max par plan.
- Utilise les noms EXACTS présents dans characters_library et locations_library si ces listes existent (sinon invente des noms cohérents).
- Remplis characters[] et location pour chaque plan même si vide. Mets hard_cut=true seulement si tu veux un reset narratif/visuel.
- Réutilise si possible les personnages/lieux présents dans le contexte projet (characters_library / locations_library).

Contexte projet (JSON, si utile):
{hint_txt}

Brief:
{brief}

Format de sortie (JSON) à respecter:
{json.dumps(schema_desc, ensure_ascii=False)}

Important:
- Donne des durées réalistes (plans de 2 à 8 secondes typiquement, sauf demande contraire).
- Le dernier plan doit conclure clairement l'histoire (packshot/logo, etc.).
""".strip()

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    def _get_content(resp_obj: Dict[str, Any]) -> str:
        """Extract model text content from the API response."""
        try:
            return resp_obj["choices"][0]["message"]["content"]
        except Exception:
            msg = resp_obj.get("choices", [{}])[0].get("message", {})
            c = msg.get("content", "")
            if isinstance(c, list):
                parts = []
                for it in c:
                    if isinstance(it, dict) and it.get("type") == "text":
                        parts.append(it.get("text", ""))
                return "".join(parts).strip()
            return str(c)

    last_parse_err: Optional[Exception] = None
    last_content: str = ""
    obj: Dict[str, Any]

    # JSON mode is the best path, but some accounts/models may reject it and
    # some generations still produce slightly invalid JSON (missing comma, truncated strings, etc.).
    # We therefore try:
    #  0) normal request (JSON mode if available)
    #  1) re-emit the full storyboard as STRICT JSON only
    #  2) "repair" the last JSON-like object by asking the model to output a valid JSON object
    for attempt in range(3):
        try:
            try:
                resp = call_mistral_chat(
                    api_key=api_key,
                    model=model,
                    messages=messages,
                    temperature=temperature if attempt == 0 else min(0.3, float(temperature)),
                    max_tokens=max_tokens_storyboard,
                    response_format={"type": "json_object"},
                )
            except RuntimeError as e:
                msg = str(e)
                if "response_format" in msg or "json_object" in msg:
                    resp = call_mistral_chat(
                        api_key=api_key,
                        model=model,
                        messages=messages,
                        temperature=temperature if attempt == 0 else min(0.3, float(temperature)),
                        max_tokens=max_tokens_storyboard,
                        response_format=None,
                    )
                else:
                    raise

            last_content = _get_content(resp)
            obj = _extract_json_object(last_content)
            break
        except Exception as e:
            last_parse_err = e

            if attempt == 0:
                # Ask for a clean re-emit.
                excerpt = (last_content or "").strip().replace("\r", "")
                excerpt = excerpt[:800] + ("…" if len(excerpt) > 800 else "")
                messages = messages + [
                    {"role": "assistant", "content": excerpt},
                    {
                        "role": "user",
                        "content": (
                            "Ta réponse précédente n'était pas du JSON valide ou était incomplète. "
                            "Renvoie à nouveau le storyboard COMPLET en JSON STRICT uniquement (un seul objet), "
                            "sans markdown, sans texte autour, sans commentaires. "
                            "Évite les champs très longs; garde negative_prompt concis et sans caractères exotiques."
                        ),
                    },
                ]
                continue

            if attempt == 1:
                # Repair pass: provide the JSON-ish object and ask for a fixed JSON object.
                span = _find_balanced_object(last_content or "") or (last_content or "")
                span = span.strip()[:18000]  # guard size
                fix_messages = [
                    {
                        "role": "system",
                        "content": (
                            "Tu es un validateur JSON strict. "
                            "Tu reçois un objet JSON potentiellement invalide. "
                            "Ta tâche est de renvoyer le MÊME contenu corrigé en JSON STRICT (un seul objet), "
                            "sans texte autour, sans markdown. "
                            "Ne change pas la structure attendue (title, logline, style, shots[]...)."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Corrige ce JSON pour qu'il soit valide. "
                            "Règles: pas de virgule finale, guillemets doubles, pas de caractères de contrôle.\n\n"
                            f"JSON À CORRIGER:\n{span}"
                        ),
                    },
                ]

                resp = call_mistral_chat(
                    api_key=api_key,
                    model=model,
                    messages=fix_messages,
                    temperature=0.0,
                    max_tokens=max_tokens_repair,
                    response_format={"type": "json_object"},
                )
                last_content = _get_content(resp)
                obj = _extract_json_object(last_content)
                break

            raise

    if last_parse_err and not isinstance(locals().get("obj", None), dict):
        excerpt = (last_content or "").strip().replace("\r", "")
        excerpt = excerpt[:1200] + ("…" if len(excerpt) > 1200 else "")
        raise RuntimeError(f"Storyboard JSON parse failed: {last_parse_err}. Output excerpt: {excerpt}") from last_parse_err

    shots: List[StoryboardShot] = []
    raw_shots = obj.get("shots", [])
    if not isinstance(raw_shots, list):
        raw_shots = []


    def _safe_str_list(x) -> List[str]:
        if x is None:
            return []
        if isinstance(x, str):
            x = x.replace(";", ",")
            return [t.strip() for t in x.split(",") if t.strip()]
        if isinstance(x, list):
            return [str(t).strip() for t in x if str(t).strip()]
        return []

    for i, s in enumerate(raw_shots, start=1):
        if not isinstance(s, dict):
            continue
        label = str(s.get("label", "")).strip() or f"S01_SH{i:02d}"
        shots.append(
            StoryboardShot(
                label=label,
                seconds=_safe_float(s.get("seconds", 3.0), default=3.0),
                tags=str(s.get("tags", "")),
                prompt=str(s.get("prompt", "")),
                negative_prompt=str(s.get("negative_prompt", "")),
                notes=str(s.get("notes", "")),
                characters=_safe_str_list(s.get("characters", s.get("character", []))),
                location=str(s.get("location", s.get("background", s.get("set", "")))),
                hard_cut=bool(s.get("hard_cut", s.get("hardCut", False))),
                shot_type=str(s.get("shot_type", s.get("shotType", ""))),
                camera_move=str(s.get("camera_move", s.get("cameraMove", ""))),
                mood=str(s.get("mood", "")),
                music_intensity=_safe_float(s.get("music_intensity", s.get("musicIntensity", 0.5)), default=0.5),
                mode=str(s.get("mode", "")),
                model_id=str(s.get("model_id", "")),
                dialogue=str(s.get("dialogue", s.get("dialogue_text", ""))),
                speaker=str(s.get("speaker", "")),
                dialogue_language=str(s.get("dialogue_language", s.get("language", ""))),
                dialogue_voice=str(s.get("dialogue_voice", s.get("voice", ""))),
            )
        )

    # Enforce exact shot count (the model sometimes returns too few/many).
    if num_shots and len(shots) != int(num_shots):
        if len(shots) > int(num_shots):
            shots = shots[: int(num_shots)]
        else:
            # Pad missing shots with simple placeholders.
            start = len(shots) + 1
            for i in range(start, int(num_shots) + 1):
                shots.append(
                    StoryboardShot(
                        label=f"S01_SH{i:02d}",
                        seconds=max(2.0, float(total_seconds) / max(1, int(num_shots))),
                        tags="",
                        prompt="",
                        negative_prompt="",
                        notes="(auto) shot ajouté pour compléter le compte.",
                    )
                )

    sb = Storyboard(
        title=str(obj.get("title", "Storyboard")),
        logline=str(obj.get("logline", "")),
        style=str(obj.get("style", style)),
        shots=shots,
    )

    # Normalize durations to match total_seconds (small drift correction)
    target = float(total_seconds)
    tot = sum(max(0.0, sh.seconds) for sh in sb.shots) or 1.0
    if abs(tot - target) > 1e-2 and sb.shots:
        scale = target / tot
        for sh in sb.shots:
            sh.seconds = max(0.5, sh.seconds * scale)
        drift = target - sum(sh.seconds for sh in sb.shots)
        sb.shots[-1].seconds = max(0.5, sb.shots[-1].seconds + drift)

    return sb
