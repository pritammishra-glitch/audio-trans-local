"""Stage 2, best case - neural diarization with pyannote 3.1.

This is what the build guide specified for mono recordings, and it is better
than the clustering in stage2_mono because it was trained to do exactly this
job, including telling apart two speakers who overlap.

It is optional because it asks for two things the other paths do not: the
`pyannote.audio` package, and a free Hugging Face token with the model
licence accepted once in a browser at
huggingface.co/pyannote/speaker-diarization-3.1

    pip install pyannote.audio
    export HF_TOKEN=hf_...

run.sh tries this first and falls back to stage2_mono when either is missing,
so nothing breaks without it.

NOTE: written against the pyannote 3.1 API but not executed here - this
machine has no Hugging Face token. Expect to need a small fix on first run.
"""
import json, os, sys, pathlib

MIN_SPEAKERS, MAX_SPEAKERS = 2, 4   # room for a [handover] to a third voice
CHANNEL_ROLE = {0: "CUSTOMER", 1: "AGENT"}


def main(wd: pathlib.Path):
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        print("stage2-pyannote: no HF_TOKEN set")
        return 2
    try:
        import torch
        from pyannote.audio import Pipeline
    except Exception as e:                       # noqa: BLE001
        print(f"stage2-pyannote: not installed ({e})")
        return 2

    pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                    use_auth_token=token)
    if torch.backends.mps.is_available():
        pipe.to(torch.device("mps"))

    # min/max rather than num_speakers=2: pinning it to two hides a third
    # voice, which is exactly the [handover] case the guidelines describe.
    dia = pipe(str(wd / "mono16k.wav"),
               min_speakers=MIN_SPEAKERS, max_speakers=MAX_SPEAKERS)

    raw = [{"label": lbl, "start": round(s.start, 2), "end": round(s.end, 2)}
           for s, _, lbl in dia.itertracks(yield_label=True)]
    if not raw:
        print("stage2-pyannote: no speech found")
        return 2

    # pyannote returns SPEAKER_00, SPEAKER_01... Map by who speaks first,
    # which is the guide's heuristic, and tell the reviewer to check it.
    order, seen = [], set()
    for t in sorted(raw, key=lambda t: t["start"]):
        if t["label"] not in seen:
            seen.add(t["label"])
            order.append(t["label"])
    role = {}
    for i, lbl in enumerate(order):
        role[lbl] = CHANNEL_ROLE.get(1 - i if i < 2 else i,
                                     f"OTHER_{i - 1}") if i < 2 else f"OTHER_{i - 1}"
    role[order[0]] = "AGENT"
    if len(order) > 1:
        role[order[1]] = "CUSTOMER"

    turns = [{"speaker": role[t["label"]], "start": t["start"], "end": t["end"]}
             for t in sorted(raw, key=lambda t: t["start"])]

    # Genuine overlapped speech - the one thing clustering cannot see.
    overlaps = []
    for a, b in zip(turns, turns[1:]):
        if b["start"] < a["end"] and a["speaker"] != b["speaker"]:
            overlaps.append({"speaker": f"{a['speaker']}|{b['speaker']}",
                             "start": b["start"], "end": min(a["end"], b["end"])})

    flags = ["FLAG: speaker roles come from who spoke first - check which is "
             "AGENT and which is CUSTOMER"]
    if len(order) > 2:
        flags.append(f"FLAG: {len(order)} voices detected - check for a "
                     f"[handover]")

    json.dump({"turns": turns, "overlaps": overlaps, "flags": flags,
               "method": "pyannote 3.1"},
              open(wd / "speakers.json", "w"), indent=2)
    print(f"stage2-pyannote: {len(turns)} turns, {len(order)} voices, "
          f"{len(overlaps)} true overlaps")
    return 0


if __name__ == "__main__":
    sys.exit(main(pathlib.Path(sys.argv[1])))
