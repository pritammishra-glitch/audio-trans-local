"""Stage 7a - assemble words, speakers, pauses and events into turns.

Runs before the LLM. Turn segmentation follows the spec's rule: start a new
line when the speaker changes, or when the same speaker stops for more than
half a second. Measured pauses are dropped into the text at the right word
boundary, so the LLM never has to invent a duration.
"""
import json, sys, pathlib

wd = pathlib.Path(sys.argv[1])
words = json.load(open(wd / "asr.json"))["words"]
spk = json.load(open(wd / "speakers.json"))
ac = json.load(open(wd / "acoustics.json"))
ev = json.load(open(wd / "events.json"))


NEAR = 1.0   # how far a word may sit outside a turn and still belong to it


def speaker_of(word):
    """Which channel a word belongs to.

    Whisper's word times and the energy-derived turn edges disagree by a
    fraction of a second, so match on overlap rather than on containment, and
    fall back to the nearest turn within NEAR seconds. Overlapping turns
    resolve to whoever started first.
    """
    ws, we = word["start"], word["end"]
    best, best_overlap = None, 0.0
    for s in spk["turns"]:
        ov = min(we, s["end"]) - max(ws, s["start"])
        if ov > best_overlap or (ov == best_overlap and best
                                 and s["start"] < best["start"]):
            best, best_overlap = s, ov
    if best_overlap > 0:
        return best["speaker"].split("|")[0]

    near, dist = None, NEAR
    for s in spk["turns"]:
        d = s["start"] - we if s["start"] > we else ws - s["end"]
        if 0 <= d < dist:
            near, dist = s, d
    return near["speaker"].split("|")[0] if near else "UNKNOWN"


def build(stream, fixed_speaker=None):
    """Words -> turns. New turn when the speaker changes, or after 0.5 s."""
    out, cur = [], None
    for w in stream:
        who = fixed_speaker or w.get("speaker") or speaker_of(w)
        # "UNKNOWN" only means no energy turn matched this word. Starting a
        # new speaker on that is worse than carrying on with the current one -
        # it produced a phantom third speaker on a two-party call.
        if who == "UNKNOWN" and cur is not None:
            who = cur["speaker"]
        if cur is None or who != cur["speaker"] or w["start"] - cur["end"] > 0.5:
            if cur:
                out.append(cur)
            cur = {"speaker": who, "start": w["start"], "end": w["end"],
                   "text": w["word"]}
        else:
            cur["text"] += " " + w["word"]
            cur["end"] = w["end"]
    if cur:
        out.append(cur)
    return out


# With per-channel transcription each speaker has their own word stream, so
# turns are built inside each stream and only then interleaved. Building them
# from the merged stream let one speaker's word land in the middle of the
# other's sentence and split it: "It clears on the 12th, no sorry, the 14th"
# came out as three turns reading "mm it", "clears", "on the 12th...".
by_speaker = {}
for w in words:
    if w.get("speaker"):
        by_speaker.setdefault(w["speaker"], []).append(w)

if by_speaker:
    turns = []
    for who_, stream in by_speaker.items():
        turns += build(stream, fixed_speaker=who_)
    turns.sort(key=lambda t: t["start"])
else:
    turns = build(words)

# measured pauses, inserted at the word boundary they were measured at
for t in turns:
    inside = sorted((p for p in ac["pauses"] if t["start"] <= p["start"] <= t["end"]),
                    key=lambda p: p["start"])
    parts = t["text"].split()
    for n, p in enumerate(inside):
        n_before = len([w for w in words
                        if t["start"] <= w["start"] and w["end"] <= p["start"]])
        parts.insert(min(n_before + n, len(parts)), f"[pause: {p['dur']}s]")
    t["text"] = " ".join(parts)

# [interrupt]: on an overlap, the speaker who started later gets the tag and
# the earlier line ends with a dash
for ov in spk.get("overlaps", []):
    inside = [t for t in turns if ov["start"] <= t["start"] <= ov["end"]]
    if not inside:
        continue
    later = max(inside, key=lambda t: t["start"])
    earlier = [t for t in turns if t["start"] < later["start"] <= t["end"]]
    later["interrupt"] = True
    for e in earlier:
        e["cut_off"] = True

for t in turns:
    t["events"] = [e for e in ev["events"] if t["start"] <= e["t"] <= t["end"]]
    for e in t["events"]:
        e["_used"] = True
    t["packet_drops"] = [d for d in ev["packet_drop_candidates"]
                         if t["start"] <= d["t"] <= t["end"]]

# Events that fall between turns - the ringback tone before anyone speaks,
# a beep in a silence - belonged to no turn and were silently dropped. They
# are written out separately so the formatter can give them their own line.
loose = [e for e in ev["events"] if not e.pop("_used", False)]
json.dump(loose, open(wd / "standalone_events.json", "w"), indent=2)

json.dump(turns, open(wd / "merged_turns.json", "w"), indent=2)
print(f"stage7a: {len(turns)} turns, {len(loose)} events outside any turn")
