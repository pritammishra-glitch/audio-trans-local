"""Stage 2 - speaker separation from stereo channel energy.

Customer on the left track, agent on the right. Channel energy tells you who is
speaking with near-perfect accuracy and costs no model at all. Overlaps are
kept, because they are what the [interrupt] rule needs.
"""
import json, sys, pathlib
import numpy as np, soundfile as sf

HOP = 0.02          # 20 ms frames
SPEECH_DB = -45.0   # above this, the channel is active
MIN_TURN = 0.10     # drop frame-level flicker shorter than this


def channel_activity(path: pathlib.Path):
    audio, sr = sf.read(path, always_2d=True)
    if audio.shape[1] != 2:
        return None, sr
    frame = int(HOP * sr)
    n = len(audio) // frame
    db = np.zeros((n, 2))
    for ch in (0, 1):
        x = audio[: n * frame, ch].reshape(n, frame)
        rms = np.sqrt((x ** 2).mean(axis=1)) + 1e-10
        db[:, ch] = 20 * np.log10(rms)
    return db, sr


def smooth(active, fill=int(0.5 / HOP)):
    """Bridge gaps shorter than `fill` frames.

    0.5 s is not arbitrary: the spec starts a new line when a speaker stops for
    more than half a second, so anything shorter belongs to the same turn and
    should not fragment the activity mask."""
    out = active.copy()
    for ch in range(out.shape[1]):
        col, run = out[:, ch], 0
        for i, v in enumerate(col):
            if v:
                if 0 < run <= fill:
                    col[i - run:i] = True
                run = 0
            else:
                run += 1
    return out


def to_turns(db):
    """Frame-level activity -> labelled turns, with overlaps kept."""
    active = smooth(db > SPEECH_DB)
    turns, cur = [], None
    for i, (left, right) in enumerate(active):
        who = ("CUSTOMER" if left else "") + ("|AGENT" if right else "")
        who = who.strip("|") or None
        t = i * HOP
        if who != (cur["speaker"] if cur else None):
            if cur:
                cur["end"] = round(t, 2)
                turns.append(cur)
            cur = {"speaker": who, "start": round(t, 2), "end": round(t, 2)} if who else None
    if cur:
        cur["end"] = round(len(active) * HOP, 2)
        turns.append(cur)
    return [t for t in turns if t["end"] - t["start"] >= MIN_TURN]


def track_check(db):
    """Flag-the-file rule: one track empty means the recording is unusable."""
    problems = []
    for ch, name in ((0, "CUSTOMER/left"), (1, "AGENT/right")):
        if float(np.percentile(db[:, ch], 99)) < SPEECH_DB:
            problems.append(f"FLAG: {name} track is empty or silent")
    return problems


if __name__ == "__main__":
    wd = pathlib.Path(sys.argv[1])
    db, sr = channel_activity(wd / "original.wav")
    if db is None:
        print("mono input -> stereo split unavailable; see README for the pyannote fallback")
        sys.exit(2)
    turns = to_turns(db)
    overlaps = [t for t in turns if "|" in t["speaker"]]
    json.dump({"turns": turns, "overlaps": overlaps, "flags": track_check(db)},
              open(wd / "speakers.json", "w"), indent=2)
    print(f"stage2: {len(turns)} turns, {len(overlaps)} overlaps")
