"""Stage 7b - mark turns spoken in another language.

Two sources of evidence, treated very differently.

**Script (certain).** When whisper writes a turn in Devanagari, Kannada, Tamil,
Arabic, CJK and so on, the language is not in doubt. Those turns are wrapped:

    <Kannada>...</Kannada>

**Audio language ID (advisory only).** Whisper's detector is unreliable on
8 kHz telephone speech: on a real call it scored plainly English turns - "can
check your CT scan" - as Indonesian at 0.87, and a ten-second window of
English as Spanish at 0.88. So it never rewrites a turn and never re-runs the
ASR. It only raises a >> REVIEW flag, and only when the evidence is strong
enough that every false positive seen on that call is excluded: a long turn,
very high confidence, and the main language all but ruled out.

This is the same stance the guide takes on accent - no local model separates
them dependably, so the tool refuses to assert rather than guessing.
"""
import json, re, sys, pathlib, zlib
import numpy as np, soundfile as sf
import asr_backend

SR = asr_backend.SR

# Thresholds chosen against observed false positives, not from taste.
#
# Two things were measured on real audio. First, the detector cannot name an
# Indian language: on clean Kannada it answered hi 0.33 / ta 0.32 and never
# offered kn at all, and re-transcribing in each candidate does not settle it
# either - kn scored -0.100 against hi -0.101. Second, on 8 kHz telephone
# English it confidently answered Spanish and Indonesian, and Spanish even beat
# English on log-probability.
#
# What survived both tests is narrow but solid: when the main language is all
# but absent from the distribution AND the leading candidates use a non-Latin
# script, something other than the main language is being spoken. Which one it
# is stays a question for the reviewer. Latin-script candidates are dropped,
# because that is exactly where the detector was shown to be wrong.
ADVISE_MIN_DUR = 1.2         # seconds
ADVISE_MAX_PRIMARY = 0.25    # the main language must be nearly absent
# Naming the language is unreliable on a 2-second clip but becomes dependable
# with enough audio, so the suspect turns are pooled and identified together
# rather than one at a time.
POOL_MIN_DUR = 6.0
# Whisper destabilises on a tightly cropped clip and starts looping, so each
# segment is widened before it is re-transcribed.
CONTEXT_PAD = 0.4
# Even pooled, the model struggles to tell the Indic languages apart - 6.5s of
# clean Kannada came back as Hindi at 0.29. Below this confidence the turn is
# still tagged as "not the main language", but the language is not named.
AUTO_MIN_PROB = 0.50
UNNAMED = "OtherLanguage"

NON_LATIN = {"hi", "kn", "ta", "te", "ml", "mr", "bn", "gu", "pa", "ur", "or",
             "as", "ne", "si", "sa", "ar", "fa", "he", "th", "zh", "ja", "ko",
             "ru", "uk", "el", "hy", "ka", "am", "my", "km", "lo", "bo"}

LANGUAGE_NAME = {
    "en": "English", "hi": "Hindi", "kn": "Kannada", "ta": "Tamil",
    "te": "Telugu", "ml": "Malayalam", "mr": "Marathi", "bn": "Bengali",
    "gu": "Gujarati", "pa": "Punjabi", "ur": "Urdu", "or": "Odia",
    "as": "Assamese", "ne": "Nepali", "si": "Sinhala", "sa": "Sanskrit",
    "ar": "Arabic", "fa": "Persian", "he": "Hebrew", "th": "Thai",
    "fr": "French", "es": "Spanish", "de": "German", "zh": "Chinese",
    "ja": "Japanese", "ko": "Korean", "ru": "Russian", "pt": "Portuguese",
    "it": "Italian", "nl": "Dutch", "tr": "Turkish", "id": "Indonesian",
    "ms": "Malay", "vi": "Vietnamese",
}

# Unicode ranges are decisive: if the text is in this script, it is this
# language family, whatever the audio detector thinks.
SCRIPTS = [
    ("Devanagari", "Hindi",     0x0900, 0x097F),
    ("Bengali",    "Bengali",   0x0980, 0x09FF),
    ("Gurmukhi",   "Punjabi",   0x0A00, 0x0A7F),
    ("Gujarati",   "Gujarati",  0x0A80, 0x0AFF),
    ("Odia",       "Odia",      0x0B00, 0x0B7F),
    ("Tamil",      "Tamil",     0x0B80, 0x0BFF),
    ("Telugu",     "Telugu",    0x0C00, 0x0C7F),
    ("Kannada",    "Kannada",   0x0C80, 0x0CFF),
    ("Malayalam",  "Malayalam", 0x0D00, 0x0D7F),
    ("Sinhala",    "Sinhala",   0x0D80, 0x0DFF),
    ("Thai",       "Thai",      0x0E00, 0x0E7F),
    ("Arabic",     "Arabic",    0x0600, 0x06FF),
    ("Hebrew",     "Hebrew",    0x0590, 0x05FF),
    ("Cyrillic",   "Russian",   0x0400, 0x04FF),
    ("Hangul",     "Korean",    0xAC00, 0xD7AF),
    ("CJK",        "Chinese",   0x4E00, 0x9FFF),
    ("Kana",       "Japanese",  0x3040, 0x30FF),
]


def name_of(code):
    return LANGUAGE_NAME.get(code, (code or "unknown").upper())


def degenerate(text):
    """Whisper loops on short clips, repeating a token - or a single
    character - over and over.

    Such a transcription is worse than none: it looks like content. The
    compression ratio catches character-level loops that word counting misses
    ("...ವಿವಿವಿವಿ"), which is the form the loop actually took on Kannada.
    """
    t = (text or "").strip()
    if len(t) < 20:
        return False
    # a short unit repeated six times or more in a row
    if re.search(r"(.{1,4}?)\1{5,}", t):
        return True
    raw = t.encode("utf-8")
    if len(raw) / max(1, len(zlib.compress(raw))) > 2.4:
        return True
    toks = t.split()
    return (len(toks) >= 6
            and max(toks.count(x) for x in set(toks)) / len(toks) > 0.4)


def script_language(text):
    """The language name if the text is written in a non-Latin script."""
    counts = {}
    letters = 0
    for ch in text:
        if not ch.isalpha():
            continue
        letters += 1
        cp = ord(ch)
        for _, lang, lo, hi in SCRIPTS:
            if lo <= cp <= hi:
                counts[lang] = counts.get(lang, 0) + 1
                break
    if not counts or not letters:
        return None
    lang, n = max(counts.items(), key=lambda kv: kv[1])
    return lang if n / letters >= 0.5 else None


if __name__ == "__main__":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    wd = pathlib.Path(sys.argv[1])
    # The detector can tell that a turn is not English; it cannot reliably say
    # which Indian language it is. When the caller names the second language
    # of the call, that settles it and the turn gets a real tag instead of a
    # question for the reviewer.
    second = (sys.argv[2].lower() if len(sys.argv) > 2 else "") or ""
    if second in ("none", "auto", "-"):
        second = ""
    turns = json.load(open(wd / "merged_turns.json"))
    primary = (json.load(open(wd / "asr.json")).get("language") or "en").lower()

    audio, sr = sf.read(wd / "mono16k.wav", dtype="float32")
    assert sr == SR, f"expected 16 kHz, got {sr}"

    # Script first - it needs no model at all.
    wrapped = 0
    for t in turns:
        lang = script_language(t["text"])
        if lang and lang != name_of(primary):
            t["lang_name"] = lang
            t["lang_certain"] = True
            wrapped += 1

    advised = 0
    remaining = [t for t in turns
                 if not t.get("lang_certain")
                 and t["end"] - t["start"] >= ADVISE_MIN_DUR]
    if remaining:
        asr_backend.warm()

    # Pass 1 - which turns are not in the main language.
    suspects = []
    for t in remaining:
        a0 = max(0, int((t["start"] - CONTEXT_PAD) * SR))
        a1 = min(len(audio), int((t["end"] + CONTEXT_PAD) * SR))
        seg = audio[a0:a1]
        probs = asr_backend.language_probs(seg)
        if probs.get(primary, 0.0) > ADVISE_MAX_PRIMARY:
            continue
        ranked = sorted(probs.items(), key=lambda kv: -kv[1])
        if ranked[0][0] not in NON_LATIN:
            continue
        t["lang_prob"] = round(float(ranked[0][1]), 3)
        suspects.append((t, [c for c, _ in ranked[:3] if c in NON_LATIN], seg))

    # Pass 2 - name the language once, from all the suspect audio together.
    pooled_dur = sum(len(seg) for _, _, seg in suspects) / SR
    auto, unnamed = "", ""
    if suspects and not second and pooled_dur >= POOL_MIN_DUR:
        pool = np.concatenate([seg for _, _, seg in suspects])
        probs = asr_backend.language_probs(pool)
        ranked_pool = sorted(((c, p) for c, p in probs.items() if c in NON_LATIN),
                             key=lambda kv: -kv[1])
        best = ranked_pool[0] if ranked_pool else (None, 0.0)
        pool_names = "/".join(name_of(c) for c, _ in ranked_pool[:3])
        if best[0] and best[1] >= AUTO_MIN_PROB:
            auto = best[0]
            print(f"  pooled {pooled_dur:.1f}s of non-{name_of(primary)} speech "
                  f"-> {name_of(auto)} ({best[1]:.2f})")
        elif best[0]:
            unnamed = pool_names
            print(f"  pooled {pooled_dur:.1f}s of non-{name_of(primary)} speech "
                  f"- best guess {name_of(best[0])} at only {best[1]:.2f}, "
                  f"too low to name. Tagging <{UNNAMED}>; candidates "
                  f"{pool_names}. Pick the language in the UI to label it.")
    elif suspects and not second:
        print(f"  only {pooled_dur:.1f}s of non-{name_of(primary)} speech - too "
              f"little to name the language, flagging for review instead")

    chosen = second or auto
    for t, cands, seg in suspects:
        hint = ""
        if chosen or cands:
            try:
                hint = asr_backend.transcribe_text(seg, chosen or cands[0])
            except Exception as e:                      # noqa: BLE001
                print(f"  (hint unavailable at {t['start']:.2f}s: {e})")
        if hint:
            t["lang_hint"] = hint

        if not chosen and unnamed:
            # Certain it is not the main language, not certain which one.
            t["lang_certain"] = True
            t["lang_name"] = UNNAMED
            t["lang_source"] = "unnamed"
            t["lang_suspected"] = unnamed
            # Keep the best-effort transcription rather than discarding it.
            # It is in the leading candidate's script, which may be the wrong
            # one, so the flag tells the reviewer not to trust the spelling.
            if hint and not degenerate(hint):
                t["text"] = hint
            else:
                t["text"] = "[unclear]"
                t["lang_failed"] = True
            wrapped += 1
            print(f"  {t['start']:7.2f}s {t['speaker']:9s} -> "
                  f"<{UNNAMED}> {t['text'][:40]}  (candidates {unnamed})")
            continue

        if chosen:
            t["lang_certain"] = True
            t["lang_name"] = name_of(chosen)
            t["lang_source"] = "given" if second else "detected"
            # The main pass transcribed this turn as if it were the call's main
            # language, so its text is not to be trusted. Prefer the
            # re-transcription in the language actually spoken - unless that
            # collapsed into a repetition loop, in which case say so.
            if hint and not degenerate(hint):
                t["text"] = hint
            else:
                t["text"] = "[unclear]"
                t["lang_failed"] = True
            wrapped += 1
            print(f"  {t['start']:7.2f}s {t['speaker']:9s} -> "
                  f"<{name_of(chosen)}> {t['text'][:45]}")
        else:
            t["lang_suspected"] = "/".join(name_of(c) for c in cands)
            advised += 1
            print(f"  {t['start']:7.2f}s {t['speaker']:9s} not "
                  f"{name_of(primary)} -> {t['lang_suspected']}"
                  f"  hint: {hint[:45]}")

    json.dump(turns, open(wd / "merged_turns.json", "w"), indent=2)
    print(f"stage7b: {wrapped} turns wrapped by script, "
          f"{advised} flagged for review out of {len(turns)} "
          f"(main language {name_of(primary)})")
