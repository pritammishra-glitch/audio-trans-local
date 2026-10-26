"""Stage 3 - pauses, silences and the noise floor, measured not guessed.

Three thresholds straight from the spec: a gap of 0.5 s or more inside one
speaker's turn is [pause]; a gap of 2 s or more with nobody speaking is
[silence]; everything shorter is untagged. Rounding to one decimal happens
here, so the number the reviewer sees is a measurement.
"""
import json, sys, pathlib
import numpy as np, soundfile as sf

PAUSE_MIN, SILENCE_MIN = 0.5, 2.0


def gaps_from_words(words, turns):
    """In-turn gaps -> pause; between-turn gaps -> silence."""
    pauses, silences = [], []
    for a, b in zip(words, words[1:]):
        gap = b["start"] - a["end"]
        if gap < PAUSE_MIN:
            continue
        same_turn = any(t["start"] <= a["end"] and b["start"] <= t["end"]
                        for t in turns)
        entry = {"start": round(a["end"], 1), "dur": round(gap, 1)}
        if same_turn:
            pauses.append(entry)
        elif gap >= SILENCE_MIN:
            silences.append(entry)
    return pauses, silences


def noise_floor(path):
    audio, sr = sf.read(path, always_2d=True)
    out = []
    for ch in range(audio.shape[1]):
        frame = int(0.02 * sr)
        n = len(audio) // frame
        x = audio[: n * frame, ch].reshape(n, frame)
        rms = np.sqrt((x ** 2).mean(axis=1)) + 1e-10
        out.append(round(float(np.percentile(20 * np.log10(rms), 10)), 1))
    return out


if __name__ == "__main__":
    wd = pathlib.Path(sys.argv[1])
    words = json.load(open(wd / "asr.json"))["words"]
    turns = json.load(open(wd / "speakers.json"))["turns"]
    pauses, silences = gaps_from_words(words, turns)
    info = sf.info(wd / "original.wav")
    floors = noise_floor(wd / "original.wav")
    json.dump({"pauses": pauses,
               "silences": silences,
               "duration": round(info.duration, 2),
               "noise_floor_db": floors,
               # a floor above about -40 dBFS means the reviewer should listen
               # for background sound the event stage may have missed
               "noise_floor_advisory": [f"channel {i} floor {f} dBFS is high"
                                        for i, f in enumerate(floors) if f > -40.0]},
              open(wd / "acoustics.json", "w"), indent=2)
    print(f"stage3: {len(pauses)} pauses, {len(silences)} silences, floor {floors} dBFS")
