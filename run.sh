#!/bin/bash
# Local Audio Transcription & Annotation Tool
# One model resident at a time: each stage is its own process and exits before
# the next loads, so 16 GB is enough.
set -e
cd "$(dirname "$0")"

F="$1"
if [ -z "$F" ]; then echo "usage: ./run.sh audio/call.wav [lang] [second-lang]"; exit 1; fi
LANG_ARG="${2:-auto}"
SECOND_LANG="${3:-}"   # optional: the other language spoken on the call
FAST_FLAG=""           # FAST=1 ./run.sh ... skips the LLM annotation pass
# ASR_HOTWORDS="CT PNS, sinus, ENT" ./run.sh ...  seeds domain vocabulary
# OLLAMA_MODEL=qwen2.5:14b ./run.sh ...           a stronger annotation model
[ "${FAST:-}" = "1" ] && FAST_FLAG="--fast"
NAME=$(basename "${F%.*}")
WD="out/$NAME"
PY="./.venv/bin/python"

mkdir -p "$WD"
export OLLAMA_KEEP_ALIVE=0

echo "== stage 0  prepare audio"           && $PY src/stage0_prep.py "$F" "$WD"

echo "== stage 1  transcription"          && $PY src/stage1_asr.py          "$WD" "$LANG_ARG"
echo "== stage 2  speaker separation"     && { $PY src/stage2_speakers.py "$WD" \
    || $PY src/stage2_pyannote.py "$WD" || $PY src/stage2_mono.py "$WD"; }
echo "== stage 3  pauses and silences"    && $PY src/stage3_acoustics.py    "$WD"
echo "== stage 4  environment events"     && $PY src/stage4_events.py       "$WD"
echo "== stage 5  speaker metadata"       && $PY src/stage5_speaker_meta.py "$WD"
echo "== stage 7a merge into turns"       && $PY src/stage7_merge.py        "$WD"
echo "== stage 7b language per turn"      && $PY src/stage7b_language.py    "$WD" "$SECOND_LANG"
echo "== stage 6  annotation (ollama)"    && $PY src/stage6_annotate.py     "$WD" $FAST_FLAG || true
echo "== stage 8  format"                 && $PY src/stage8_format.py       "$WD"

echo
echo "done -> $WD/transcript.txt"
