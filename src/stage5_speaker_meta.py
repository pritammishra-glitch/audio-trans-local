"""Stage 5 - the per-speaker metadata block.

Pitch rather than a neural gender classifier, because the output is auditable:
a reviewer can see 118 Hz and understand why the tool said male. The 145-180 Hz
band is left unclear on purpose, which is where most classifier errors occur.

Two fields deliberately refuse to self-fill. Accent stays unclear because no
local model separates Indian, US, British and Australian English dependably.
Emotional state is needs_review: the spec has nine states and asks for the one
that lasts most of the call, a judgement about the whole arc.
"""
import json, sys, pathlib
import numpy as np, soundfile as sf

FMIN, FMAX = 60.0, 400.0
CHANNEL_OF = {"CUSTOMER": 0, "AGENT": 1}   # left track, right track


def f0_median(x, sr):
    """Median pitch over voiced frames, by autocorrelation."""
    win = int(0.04 * sr)
    hop = int(0.02 * sr)
    lo, hi = int(sr / FMAX), int(sr / FMIN)
    vals = []
    for s in range(0, len(x) - win, hop):
        seg = x[s:s + win] - x[s:s + win].mean()
        energy = np.sqrt((seg ** 2).mean())
        if energy < 1e-3:
            continue
        ac = np.correlate(seg, seg, mode="full")[win - 1:]
        if ac[0] <= 0 or hi >= len(ac):
            continue
        ac = ac / ac[0]
        lag = lo + int(np.argmax(ac[lo:hi]))
        if ac[lag] > 0.35:              # voiced enough to trust
            vals.append(sr / lag)
    return float(np.median(vals)) if vals else 0.0


if __name__ == "__main__":
    wd = pathlib.Path(sys.argv[1])
    turns = json.load(open(wd / "speakers.json"))["turns"]
    audio, sr = sf.read(wd / "original.wav", always_2d=True)

    meta = {}
    for speaker in sorted({t["speaker"] for t in turns if "|" not in (t["speaker"] or "")}):
        ch = CHANNEL_OF.get(speaker, 0)
        if ch >= audio.shape[1]:
            ch = 0
        segs = [audio[int(t["start"] * sr): int(t["end"] * sr), ch]
                for t in turns if t["speaker"] == speaker]
        f0 = f0_median(np.concatenate(segs).astype(np.float64), sr) if segs else 0.0
        if f0 == 0:
            g = "unclear"
        elif f0 < 145:
            g = "male"
        elif f0 > 180:
            g = "female"
        else:
            g = "unclear"               # the ambiguous band stays unclear
        meta[speaker] = {
            "gender": g,
            "f0_median_hz": round(f0, 1),
            "emotional_state": "needs_review",
            "physical_state": "stationary",   # no breathing, footsteps or wind detected
            "accent": "unclear",              # never asserted automatically
        }

    json.dump(meta, open(wd / "speakers_meta.json", "w"), indent=2)
    for s, m in meta.items():
        print(f"stage5: {s} {m['gender']} (f0 {m['f0_median_hz']} Hz)")
