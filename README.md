# Wan Studio (Verified)

## Run
```bash
python app.py
```

## Setup (recommended)
Reproducible install (lockfile) with **uv**:

```bash
bash setup_uv.sh
```

Classic venv install:

```bash
bash setup_pro.sh
```

## Tools (optional)
```bash
bash get_tools.sh ./tools
```

Par défaut ça installe maintenant aussi **Piper (TTS)** et télécharge une voix
(`fr_FR-upmc-medium`). Tu peux choisir une autre voix via:

```bash
PIPER_VOICE=en_US-lessac-medium bash get_tools.sh ./tools
```

## Audio (Piper)
Une fois les tools installés, ouvre l'onglet **Audio**:
- texte par plan (`dialogue_text`)
- génération Piper + auto-fit à la durée du plan
- import d'une **musique globale** (optionnel)
- import d'un **VFX/SFX wav par plan** (optionnel)
- export **stems** pour montage:
  - `audio/shots/*_dialogue.wav` + `audio/shots/*_vfx.wav`
  - `audio/scenes/S01_dialogue.wav` + `audio/scenes/S01_vfx.wav` (si labels `Sxx_SHxx`)
- construction de 3 pistes globales: `dialogue_track.wav`, `vfx_track.wav`, `music_track.wav`
- mix automatique (ducking musique sous les dialogues) → `master_mix.wav`
- sous-titres simples (1 ligne par plan) → `audio/subtitles.srt` (optionnel)
- mux de `master_mix.wav` dans un rendu vidéo

CLI équivalente:

```bash
python -m wan_studio.audio_cli gen --project mon_projet.json --out outputs/run1
python -m wan_studio.audio_cli track --project mon_projet.json --out outputs/run1
python -m wan_studio.audio_cli mix --project mon_projet.json --out outputs/run1
python -m wan_studio.audio_cli mux --video outputs/run1/final.mp4 --audio outputs/run1/audio/master_mix.wav
```

## Outputs
Rendus dans `outputs/project_YYYYMMDD_HHMMSS/`
- `raw.mp4`
- `post/final.mp4` (si post-prod ON)
- `master_prores.mov` (si export ProRes ON)

## Quick presets
- Draft Preview (Low Res): validation rapide du prompt (VRAM mini)
- Fast Preview
- SVI Longform (stable)
- 1080p Premium


## Notes
- `ftfy` est requis par le pipeline WAN I2V de Diffusers pour nettoyer les prompts.


## Notes
- Diffusers a déprécié `torch_dtype` au profit de `dtype`. Cette version utilise `dtype=`.

- Certaines LoRA WAN sont publiées sans préfixe `transformer.`. Pour éviter le warning
  "No LoRA keys associated … prefix='transformer'", l'app force `prefix=None` sur le backend **wan**.


## Multi-LoRA (adapters)
Pour activer `set_adapters()` / multi-LoRA dans Diffusers, installez PEFT:

    pip install -U peft transformers


## OOM / fragmentation (PyTorch)
Si tu vois des erreurs du type "reserved but unallocated" et des OOM aléatoires,
tu peux essayer:

    export PYTORCH_ALLOC_CONF=expandable_segments:True


## Tools (RIFE / Real-ESRGAN)
Assure-toi d'installer les tools via `./get_tools.sh ./tools`.
Cette version copie aussi les dossiers `models/` nécessaires aux binaires ncnn.


## v12 tools/gui postfix
- Fixed get_tools.py (key variable) and ensured model asset folders are copied.
- Fixed GUI preview reload (_reload_last_output).
