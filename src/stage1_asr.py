"""Stage 1 - words + per-word timestamps.

Word-level times are what pause detection and the [unclear] logic depend on,
and a temperature of 0 disables the fallback that makes whisper invent fluent
text over silence.

On a stereo call each channel is transcribed separately. That beats
transcribing the mix twice over: whisper hallucinates where two voices
overlap, and the channel then names the speaker exactly instead of by energy.

The engine behind this is chosen in asr_backend - mlx-whisper on Apple
Silicon, faster-whisper everywhere else.
"""
import json, sys, pathlib
import soundfile as sf
import asr_backend

CHANNEL_SPEAKER = {0: "CUSTOMER", 1: "AGENT"}   # left track, right track


def read16k(path):
    audio, sr = sf.read(path, dtype="float32")
    assert sr == asr_backend.SR, f"expected 16 kHz, got {sr}"
    return audio


def run_asr(workdir: pathlib.Path, language=None):
    per_channel = [workdir / f"ch{c}_16k.wav" for c in (0, 1)]
    if all(p.exists() for p in per_channel):
        words, lang = [], None
        for ch, path in enumerate(per_channel):
            got, detected = asr_backend.transcribe_words(read16k(path), language)
            lang = lang or detected
            for w in got:
                w["channel"] = ch
                w["speaker"] = CHANNEL_SPEAKER[ch]
            words += got
        words.sort(key=lambda w: w["start"])
        source = "per-channel"
    else:
        words, lang = asr_backend.transcribe_words(
            read16k(workdir / "mono16k.wav"), language)
        source = "mono mix"

    (workdir / "asr.json").write_text(json.dumps(
        {"words": words, "language": lang or "en", "source": source,
         "engine": asr_backend.backend()}, indent=2))
    print(f"stage1: {len(words)} words, language={lang}, from {source} "
          f"({asr_backend.backend()})")


if __name__ == "__main__":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    lang = sys.argv[2] if len(sys.argv) > 2 else None
    run_asr(pathlib.Path(sys.argv[1]), None if lang in (None, "auto") else lang)
