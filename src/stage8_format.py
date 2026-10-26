"""Stage 8 - render the annotated turns.

Two files, from the same data:

  transcript.txt         one line per turn, [start - end] and Speaker A/B,
                         with every tag inline. This is the delivery format.
  transcript_review.txt  the guidelines' two-line As spoken / Clean format,
                         with the metadata block and the >> REVIEW handoff
                         lines that tell a reviewer where to listen.
"""
import json, re, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from precheck import flags as file_flags

wd = pathlib.Path(sys.argv[1])
ann = json.load(open(wd / "annotated.json"))
turns = ann["turns"]
ac = json.load(open(wd / "acoustics.json"))
ev = json.load(open(wd / "events.json"))
spk = json.load(open(wd / "speakers.json"))
meta = json.load(open(wd / "speakers_meta.json"))
lang = json.load(open(wd / "asr.json")).get("language", "en")


def ts_ms(sec):
    """00:02.160 - minutes, seconds, milliseconds."""
    return f"{int(sec // 60):02d}:{sec % 60:06.3f}"


def ts_short(sec):
    return f"{int(sec // 60):02d}:{sec % 60:04.1f}"


# Speaker A, B, C... in order of first appearance, which is how the delivery
# format labels them. The role names are kept in the review file, where the
# per-speaker metadata block needs them.
letters, order = {}, []
for t in turns:
    if t["speaker"] == "UNKNOWN":
        letters["UNKNOWN"] = "?"
        continue
    if t["speaker"] not in letters:
        letters[t["speaker"]] = chr(ord("A") + len([v for v in letters.values()
                                                    if v != "?"]))
        order.append(t["speaker"])

# <pii:card>9014</pii> -> <CARD>9014</CARD>, to match the entity tag style
PII_TAG = re.compile(r"<pii:(\w+)>(.*?)</pii>", re.S)


def upper_tags(text):
    return PII_TAG.sub(lambda m: f"<{m.group(1).upper()}>{m.group(2)}"
                                 f"</{m.group(1).upper()}>", text)


# ---------------------------------------------------------------- delivery
lines, emitted = [], set()
for t in turns:
    for s in ac["silences"]:
        if s["start"] < t["start"] and s["start"] not in emitted:
            end = s["start"] + s["dur"]
            lines.append(f"[{ts_ms(s['start'])} - {ts_ms(end)}] [silence: {s['dur']}s]")
            emitted.add(s["start"])
    noise = " ".join(f"[noise: {e['tag']}]" for e in t.get("events", [])
                     if e.get("confirmed"))
    text = upper_tags((noise + " " + t["as_spoken"]).strip())
    lines.append(f"[{ts_ms(t['start'])} - {ts_ms(t['end'])}] "
                 f"Speaker {letters[t['speaker']]}: {text}")
(wd / "transcript.txt").write_text("\n".join(lines) + "\n")

# ------------------------------------------------------------------ review
rev = ["STATUS: machine pre-annotation, needs human review",
       f"LANGUAGE: {lang} | SCENARIO: [REVIEW]"]
for s in order:
    m = meta.get(s)
    if m:
        rev.append(f"Speaker {letters[s]} ({s}): {m['gender']} | "
                   f"{m['emotional_state']} | {m['physical_state']} | {m['accent']}")

warnings = (spk.get("flags", []) + ac.get("noise_floor_advisory", [])
            + file_flags(turns, ac["duration"], ev["packet_drop_candidates"]))
if ann.get("halted"):
    warnings.append(ann["halted"])
if warnings:
    rev.append("")
    rev += [f"!! {w}" for w in warnings]
rev.append("")

emitted = set()
for t in turns:
    for s in ac["silences"]:
        if s["start"] < t["start"] and s["start"] not in emitted:
            rev.append(f"{ts_short(s['start'])} [silence: {s['dur']}s]")
            rev.append("")
            emitted.add(s["start"])
    noise = " ".join(f"[noise: {e['tag']}]" for e in t.get("events", [])
                     if e.get("confirmed"))
    rev.append(f"{ts_short(t['start'])}  Speaker {letters[t['speaker']]} ({t['speaker']})")
    rev.append(f"  AS SPOKEN: {(noise + ' ' + t['as_spoken']).strip()}")
    rev.append(f"  CLEAN:     {t['clean']}")
    if t.get("flags"):
        rev.append(f"  >> REVIEW: {', '.join(t['flags'])}")
    if t.get("lang_hint"):
        rev.append(f"  >> HINT ({t.get('lang_suspected', '?')}): {t['lang_hint']}")
    rev.append("")
(wd / "transcript_review.txt").write_text("\n".join(rev))

print(f"wrote {wd / 'transcript.txt'} and {wd / 'transcript_review.txt'}")
