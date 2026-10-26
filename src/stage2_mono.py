"""Stage 2 fallback - speaker separation for mono recordings.

The build guide reaches for pyannote 3.1 here. That needs a Hugging Face token
and a licence accepted in a browser, so this is a local stand-in: segment the
call on the word timings stage 1 already produced, describe each segment with
MFCCs and pitch, and cluster the segments into two speakers.

It is weaker than neural diarization and it says so - every run writes a flag
telling the reviewer that speaker labels on mono are a guess. Roles follow the
guide's heuristic: whoever speaks first is usually the agent on an inbound
call.
"""
import json, sys, pathlib
import numpy as np, soundfile as sf
from scipy.fftpack import dct
from scipy.cluster.vq import kmeans2

SR = 16000
WIN, HOP = 0.025, 0.010
N_MEL, N_MFCC = 26, 13
SEG_GAP = 0.45      # a gap this long between words starts a new segment
# Segments are cut finely so clustering has clean material, but consecutive
# segments from the same speaker are then rejoined across a gap of up to
# MERGE_GAP. Without this a turn can never contain a pause, because every gap
# over 0.45 s would end the turn - and stage 3 would report silences where the
# spec wants [pause: N.Ns].
MERGE_GAP = 2.0
MIN_SEG = 0.35      # segments shorter than this are too thin to cluster on


def mel_filters(n_fft, sr, n_mel):
    def to_mel(f):
        return 2595 * np.log10(1 + f / 700)

    def to_hz(m):
        return 700 * (10 ** (m / 2595) - 1)

    pts = to_hz(np.linspace(to_mel(50), to_mel(sr / 2), n_mel + 2))
    bins = np.floor((n_fft + 1) * pts / sr).astype(int)
    fb = np.zeros((n_mel, n_fft // 2 + 1))
    for m in range(1, n_mel + 1):
        a, b, c = bins[m - 1], bins[m], bins[m + 1]
        if b == a or c == b:
            continue
        fb[m - 1, a:b] = (np.arange(a, b) - a) / (b - a)
        fb[m - 1, b:c] = (c - np.arange(b, c)) / (c - b)
    return fb


def mfcc(x, sr):
    n_fft = 512
    win, hop = int(WIN * sr), int(HOP * sr)
    if len(x) < win:
        return np.zeros((0, N_MFCC))
    n = 1 + (len(x) - win) // hop
    frames = np.lib.stride_tricks.as_strided(
        x, shape=(n, win), strides=(x.strides[0] * hop, x.strides[0])).copy()
    frames *= np.hamming(win)
    spec = np.abs(np.fft.rfft(frames, n_fft)) ** 2
    # macOS ships numpy against Accelerate, whose BLAS raises spurious
    # divide-by-zero and overflow flags on any matmul - a plain matmul of two
    # random matrices trips it. The values are fine, so the flags are ignored
    # here and the result is checked for finiteness by the caller instead.
    with np.errstate(all="ignore"):
        energy = mel_filters(n_fft, sr, N_MEL) @ spec.T
    return dct(np.log(energy + 1e-10), type=2, axis=0, norm="ortho")[:N_MFCC].T


def f0_median(x, sr):
    win, hop = int(0.04 * sr), int(0.02 * sr)
    lo, hi = int(sr / 400), int(sr / 60)
    vals = []
    for s in range(0, max(0, len(x) - win), hop):
        seg = x[s:s + win] - x[s:s + win].mean()
        if np.sqrt((seg ** 2).mean()) < 1e-3:
            continue
        ac = np.correlate(seg, seg, mode="full")[win - 1:]
        if ac[0] <= 0 or hi >= len(ac):
            continue
        ac /= ac[0]
        lag = lo + int(np.argmax(ac[lo:hi]))
        if ac[lag] > 0.35:
            vals.append(sr / lag)
    return float(np.median(vals)) if vals else 0.0


def segments_from_words(words):
    """Group words into stretches that are very likely one speaker."""
    segs, cur = [], None
    for w in words:
        if cur and w["start"] - cur["end"] <= SEG_GAP:
            cur["end"] = w["end"]
        else:
            if cur:
                segs.append(cur)
            cur = {"start": w["start"], "end": w["end"]}
    if cur:
        segs.append(cur)
    return segs


def describe(audio, sr, seg):
    x = audio[int(seg["start"] * sr): int(seg["end"] * sr)].astype(np.float64)
    m = mfcc(x, sr)
    if len(m) == 0:
        return None
    f0 = f0_median(x, sr)
    return np.concatenate([m.mean(axis=0), m.std(axis=0), [f0 / 100.0]])


if __name__ == "__main__":
    wd = pathlib.Path(sys.argv[1])
    words = json.load(open(wd / "asr.json"))["words"]
    audio, sr = sf.read(wd / "mono16k.wav", dtype="float32")

    segs = [s for s in segments_from_words(words) if s["end"] - s["start"] >= MIN_SEG]
    feats, kept = [], []
    for s in segs:
        v = describe(audio, sr, s)
        if v is not None and np.isfinite(v).all():
            feats.append(v)
            kept.append(s)

    if len(kept) < 2:
        json.dump({"turns": [{"speaker": "UNKNOWN", "start": 0.0,
                              "end": round(len(audio) / sr, 2)}],
                   "overlaps": [],
                   "flags": ["FLAG: too little speech to separate speakers"]},
                  open(wd / "speakers.json", "w"), indent=2)
        print("stage2-mono: too little speech to cluster")
        sys.exit(0)

    X = np.vstack(feats)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    centroid, label = kmeans2(X, 2, minit="++", seed=0)

    # Whoever speaks first is usually the agent on an inbound call. This is the
    # guide's heuristic and the reviewer is told to check it.
    first = label[0]
    role = {first: "AGENT", 1 - first: "CUSTOMER"}

    turns, cur = [], None
    for s, lab in zip(kept, label):
        who = role[lab]
        if cur and cur["speaker"] == who and s["start"] - cur["end"] < MERGE_GAP:
            cur["end"] = round(s["end"], 2)
        else:
            if cur:
                turns.append(cur)
            cur = {"speaker": who, "start": round(s["start"], 2),
                   "end": round(s["end"], 2)}
    if cur:
        turns.append(cur)

    sizes = {r: sum(1 for l in label if role[l] == r) for r in ("AGENT", "CUSTOMER")}
    flags = ["FLAG: mono recording - speaker labels come from clustering, not "
             "channels. Check who is AGENT and who is CUSTOMER."]
    if min(sizes.values()) < 0.1 * len(label):
        flags.append("FLAG: one speaker holds under 10% of segments - the split "
                     "may be wrong, or there may be only one speaker")

    json.dump({"turns": turns, "overlaps": [], "flags": flags,
               "method": "mfcc+f0 k-means (pyannote not installed)"},
              open(wd / "speakers.json", "w"), indent=2)
    print(f"stage2-mono: {len(turns)} turns from {len(kept)} segments, "
          f"AGENT {sizes['AGENT']} / CUSTOMER {sizes['CUSTOMER']}")
