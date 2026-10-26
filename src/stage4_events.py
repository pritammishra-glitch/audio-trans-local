"""Stage 4 - environment events, without TensorFlow.

The build guide reaches for YAMNet here. YAMNet needs tensorflow-macos, which
this machine has no working toolchain for, so this stage does the part that can
be measured honestly and refuses the part that cannot:

  * [packet_drop] candidates - energy collapsing for 40-200 ms mid-word, or
    consecutive frames that are bit-identical, which is what a frozen jitter
    buffer produces. The guide already calls this advisory.
  * steady-noise candidates - a narrowband low-frequency peak well above the
    rest of the spectrum (fan_hum), or a flat high floor (static).

Nothing here is written into the transcript as an asserted [noise: ...] tag.
Everything lands on a >> REVIEW line. Install YAMNet to assert the full
17-label set; see README.
"""
import json, sys, pathlib
import numpy as np, soundfile as sf
from scipy import signal

FRAME = 0.02
# The guide suggests 40-200 ms. 40 ms is below the length of an ordinary stop
# consonant closure (the /k/ in "credit"), so the floor is raised to 100 ms and
# the collapse must reach the channel noise floor: a plosive closure does not,
# a muted packet does.
DROP_MIN, DROP_MAX = 0.10, 0.30   # seconds
DROP_TO_FLOOR_DB = 6.0


TONAL_FLATNESS = 0.02   # below this the frame is a tone, not speech or noise


def tonality(frame, sr):
    """Spectral flatness and dominant frequency.

    A pure tone concentrates all its power in one bin, so the geometric mean
    of the spectrum collapses relative to the arithmetic mean.
    """
    spec = np.abs(np.fft.rfft(frame * np.hanning(len(frame)))) ** 2 + 1e-20
    flat = float(np.exp(np.log(spec).mean()) / spec.mean())
    peak = float(np.fft.rfftfreq(len(frame), 1.0 / sr)[int(np.argmax(spec))])
    return flat, peak


def frames(x, sr):
    n_f = int(FRAME * sr)
    n = len(x) // n_f
    return x[: n * n_f].reshape(n, n_f)


def packet_drops(x, sr, word_spans):
    """Energy collapse mid-word, or repeated identical frames.

    "Mid-word" is the load-bearing part. A gap between words is just somebody
    pausing, and scoring those produces a wall of candidates the reviewer has
    to delete one by one - so a collapse only counts when it sits strictly
    inside a word whisper transcribed.
    """

    def mid_word(t0, t1):
        return any(ws + 0.02 < t0 and t1 < we - 0.02 for ws, we in word_spans)

    f = frames(x, sr)
    rms = np.sqrt((f ** 2).mean(axis=1)) + 1e-10
    db = 20 * np.log10(rms)
    speech = db > -45.0
    floor = float(np.percentile(db, 10))
    out, i = [], 1
    while i < len(db) - 1:
        if speech[i - 1] and db[i] < db[i - 1] - 25:
            j = i
            while j < len(db) and db[j] < db[i - 1] - 25:
                j += 1
            dur = (j - i) * FRAME
            if (DROP_MIN <= dur <= DROP_MAX and j < len(db) and speech[j]
                    and db[i:j].max() < floor + DROP_TO_FLOOR_DB
                    and mid_word(i * FRAME, j * FRAME)):
                out.append({"t": round(i * FRAME, 1), "dur": round(dur, 2),
                            "why": "energy collapse mid-word"})
            i = j + 1
        else:
            i += 1
    # Frozen jitter buffer: consecutive frames identical sample for sample.
    # A steady tone also repeats exactly, so runs are classified before they
    # are reported - a 1 kHz ringback tone is a hold tone, not a dropout.
    tones = []
    i = 1
    while i < len(f):
        if rms[i] > 1e-5 and np.array_equal(f[i], f[i - 1]):
            j = i
            while j + 1 < len(f) and np.array_equal(f[j + 1], f[j]):
                j += 1
            dur = (j - i + 2) * FRAME
            flat, peak_hz = tonality(f[i], sr)
            if flat < TONAL_FLATNESS:
                tones.append({"t": round((i - 1) * FRAME, 1), "dur": round(dur, 2),
                              "tag": "beeps", "confirmed": False,
                              "why": f"steady {peak_hz:.0f} Hz tone for {dur:.1f}s "
                                     f"- ringback, hold tone or beep"})
            elif mid_word((i - 1) * FRAME, (j + 1) * FRAME):
                out.append({"t": round((i - 1) * FRAME, 1), "dur": round(dur, 2),
                            "why": f"{j - i + 2} identical consecutive frames"})
            i = j + 1
        else:
            i += 1
    return out, tones


def steady_noise(x, sr):
    """A tonal low-frequency peak or a flat loud floor, during non-speech."""
    f = frames(x, sr)
    rms = np.sqrt((f ** 2).mean(axis=1)) + 1e-10
    quiet = f[20 * np.log10(rms) < -45.0]
    if len(quiet) < 25:
        return []
    seg = quiet.reshape(-1)
    freqs, psd = signal.welch(seg, sr, nperseg=min(1024, len(seg)))
    psd_db = 10 * np.log10(psd + 1e-20)
    out = []
    low = (freqs > 40) & (freqs < 300)
    if low.any():
        peak, median = psd_db[low].max(), float(np.median(psd_db))
        if peak > median + 18:
            out.append({"t": 0.0, "tag": "fan_hum", "confirmed": False,
                        "why": f"tonal peak at {freqs[low][psd_db[low].argmax()]:.0f} Hz, "
                               f"{peak - median:.0f} dB above the spectrum median"})
    if float(np.median(psd_db)) > -70:
        out.append({"t": 0.0, "tag": "static", "confirmed": False,
                    "why": "broadband floor is loud and flat"})
    return out


if __name__ == "__main__":
    wd = pathlib.Path(sys.argv[1])
    audio, sr = sf.read(wd / "original.wav", always_2d=True)
    word_spans = [(w["start"], w["end"])
                  for w in json.load(open(wd / "asr.json"))["words"]]
    events, drops = [], []
    for ch in range(audio.shape[1]):
        x = audio[:, ch].astype(np.float64)
        found, tones = packet_drops(x, sr, word_spans)
        for d in found:
            drops.append({**d, "channel": ch})
        for e in tones + steady_noise(x, sr):
            events.append({**e, "channel": ch})
    json.dump({"events": events, "packet_drop_candidates": drops,
               "classifier": "none (YAMNet not installed)"},
              open(wd / "events.json", "w"), indent=2)
    print(f"stage4: {len(events)} noise candidates, {len(drops)} packet-drop candidates")
