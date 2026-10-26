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


turns, cur = [], None
for w in words:
    # A per-channel transcription already knows who spoke; only fall back to
    # matching against energy turns when the words came from the mixed track.
    who = w.get("speaker") or speaker_of(w)
    # "UNKNOWN" only means no energy turn matched this word. Starting a new
    # speaker on that is worse than carrying on with the current one - it
    # produced a phantom third speaker on a two-party call.
    if who == "UNKNOWN" and cur is not None:
        who = cur["speaker"]
    if cur is None or who != cur["speaker"] or w["start"] - cur["end"] > 0.5:
        if cur:
            turns.append(cur)
        cur = {"speaker": who, "start": w["start"], "end": w["end"],
               "text": w["word"]}
    else:
        cur["text"] += " " + w["word"]
        cur["end"] = w["end"]
if cur:
    turns.append(cur)

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
    t["packet_drops"] = [d for d in ev["packet_drop_candidates"]
                         if t["start"] <= d["t"] <= t["end"]]

json.dump(turns, open(wd / "merged_turns.json", "w"), indent=2)
print(f"stage7a: {len(turns)} turns")
