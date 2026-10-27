"""Stage 6 - the Ollama annotation layer.

It never touches the audio; it takes the words stage 1 produced and applies the
written rules - fillers, false starts, self-corrections, PII, and the Clean
line. One turn at a time: a 7B model given a full transcript drifts, drops tags
and starts paraphrasing.

The flags list is the honest part. The model proposes hesitation, sarcasm,
mispronunciation and pressure as candidates for the reviewer, and never writes
those tags itself.
"""
import json, os, sys, pathlib, re, requests
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from entities import tag_entities

# qwen2.5:14b handles the self-correction and false-start rules noticeably
# better than 7b and still fits 16 GB, since this stage runs with nothing
# else resident. Pull it and set OLLAMA_MODEL=qwen2.5:14b to use it.
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
URL = "http://localhost:11434/api/chat"

SYSTEM = """You annotate call transcripts. Apply these rules exactly.

You output JSON only: {"as_spoken": "...", "clean": "...", "flags": [...]}

AS_SPOKEN - every word as said. Never fix grammar. Never change words.
- Fillers written only as: um, uh, hmm, ah, mm-hm, uh-huh. Write "er" as uh.
- Filler words stay: like, you know, I mean, so, basically, actually.
- Keep the [pause: N.Ns] markers already in the input exactly where they are.
- False start: wrap dropped words <fs>...</fs>. Broken word: <fs>resched-</fs> reschedule.
- Repeats: wrap the extra copies. <fs>I I</fs> I wanted to.
- Self-correction: wrap the replaced words <sc>...</sc>. Signals: no, sorry,
  I mean, actually, wait, make that, nahi nahi.
- NOT a correction when the speaker means both things, or restates the same
  value another way for clarity ("ninety fourteen, so nine zero one four").
- PII: <pii:name>, <pii:phone>, <pii:email>, <pii:address>, <pii:account>,
  <pii:card>, <pii:id>, <pii:dob>, <pii:otp>.
- Hindi or other language: Latin letters, no translation, <lang:hi>haan ji</lang>.
- Place names: wrap them <LOCATION>Bangalore</LOCATION>. Do NOT tag dates or
  times yourself - those are tagged automatically after you.

CLEAN - what the speaker meant, written properly.
- Remove fillers and disfluency tags. Keep [unclear] and PII tags.
- Final answers only: after a self-correction keep only the corrected value.
- Never alter a digit. If AS_SPOKEN already contains digits, copy that exact
  sequence into CLEAN. "9014" stays 9014; it never becomes 9814.
- Numbers as digits. Phone "nine eight double four oh two" -> 984402.
  Amount "twelve eighty four thirty seven" -> $1,284.37.
  Spelled name "S as in Sam, H, A, R, M, A" -> SHARMA.
  Email "priya dot k at acme dot com" -> priya.k@acme.com.
- A backchannel gets an EMPTY clean line ONLY when the whole turn is nothing
  but a short listening sound (mm-hm, uh-huh, yeah, okay, right, haan, achha,
  ji) - three words at most, carrying no information. "Yes" answering a
  question is a real answer and gets a full clean line.
- Any turn that carries content gets a full CLEAN line. Never return an empty
  clean for a turn that states, asks or confirms anything, even if it opens
  with "okay" or "yeah".

FLAGS - add any that apply, for human review, never asserted as fact:
  possible_hesitation, possible_sarcasm, possible_mispron,
  possible_pressure, possible_handover

EXAMPLES - follow these exactly.

IN:  It clears on the 12th. No, sorry, the 14th.
OUT: {"as_spoken": "It clears on the <sc>12th</sc>. No, sorry, the 14th.",
      "clean": "It clears on the 14th.", "flags": []}

IN:  Machine, not machine. The guy operating the machine is the wrong person.
OUT: {"as_spoken": "<sc>Machine</sc>, not machine. The guy operating the machine is the wrong person.",
      "clean": "The guy operating the machine is the wrong person.", "flags": []}

IN:  Sure, it's ninety fourteen, so nine zero one four.
OUT: {"as_spoken": "Sure, it's ninety fourteen, so nine zero one four.",
      "clean": "Sure, it's 9014.", "flags": []}
      (NOT a correction - the same value restated for clarity.)

IN:  Uh, I'd like to pay the full amount [pause: 0.5s] from my account.
OUT: {"as_spoken": "Uh, I'd like to pay the full amount [pause: 0.5s] from my account.",
      "clean": "I'd like to pay the full amount from my account.", "flags": []}

Output the JSON object and nothing else."""


# Two kinds of false start have no judgement in them at all, and a 7B model
# was finding neither. They are matched here instead, the same way pauses and
# PII are: an immediate repeat of the same word, and a word broken off and
# restarted. Anything subtler stays with the model.
_TAGGED = re.compile(r"<[^>]*>|\[[^\]]*\]")


def _bare(tok):
    return re.sub(r"[^\w']", "", tok).lower()


# A whole tagged span, so matching never reaches inside one. Without this a
# repeat inside an <sc> span produced "<sc> <fs>I</fs> I </sc>" - nested and
# invalid.
_SPAN = re.compile(r"<(\w[^>\s]*)[^>]*>.*?</\1>|<[^>]*>|\[[^\]]*\]", re.S)


def mark_false_starts(text):
    """Wrap immediate repeats and broken words in <fs>, outside other tags."""
    if not text or "<fs>" in text:
        return text
    # process only the stretches that carry no tags
    out, last = "", 0
    for m in _SPAN.finditer(text):
        out += _mark_free(text[last:m.start()]) + m.group(0)
        last = m.end()
    return out + _mark_free(text[last:])


def _mark_free(text):
    if not text.strip():
        return text
    lead = " " if text[:1] == " " else ""
    trail = " " if text[-1:] == " " and text.strip() else ""
    parts, out = text.split(), []
    i = 0
    while i < len(parts):
        tok = parts[i]
        if _TAGGED.search(tok) or not _bare(tok):
            out.append(tok)
            i += 1
            continue

        # "I I I wanted" -> "<fs>I I</fs> I wanted": every copy but the last.
        # A copy that ends a sentence is left alone - "Hello. Hello." is
        # someone checking whether the line is live, not a dropped restart.
        j = i
        if not tok.rstrip().endswith((".", "!", "?")):
            while (j + 1 < len(parts) and _bare(parts[j + 1]) == _bare(tok)
                   and not _TAGGED.search(parts[j + 1])
                   and not parts[j].rstrip().endswith((".", "!", "?"))):
                j += 1
        if j > i:
            out.append("<fs>" + " ".join(parts[i:j]) + "</fs>")
            out.append(parts[j])
            i = j + 1
            continue

        # "resched- reschedule": a word cut off, then said in full
        if tok.endswith("-") and i + 1 < len(parts):
            stem = _bare(tok)
            if len(stem) >= 3 and _bare(parts[i + 1]).startswith(stem):
                out.append(f"<fs>{tok}</fs>")
                i += 1
                continue

        out.append(tok)
        i += 1
    return lead + " ".join(out) + trail


# Self-corrections, but only the kind that cannot be misread. "no" and
# "sorry" are everywhere in an unhappy call ("No sir, we don't have ties"),
# so a signal word alone proves nothing. What does is a value replaced by
# another of the same kind: a number for a number, a weekday for a weekday.
# Everything subtler is left to the model.
_NUM = (r"(?:\d+(?:st|nd|rd|th)?|one|two|three|four|five|six|seven|eight|"
        r"nine|ten|eleven|twelve|twenty|thirty|forty|fifty)")
_DAY = (r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|"
        r"today|tomorrow)")
_SIGNAL = r"(?:no\s+sorry|no|sorry|i\s+mean|actually|wait|make\s+that|nahi\s+nahi)"

SELF_CORRECTION = [
    re.compile(rf"\b({_NUM})\b([,.]?\s+{_SIGNAL}[,.]?\s+(?:the\s+)?{_NUM}\b)", re.I),
    re.compile(rf"\b({_DAY})\b([,.]?\s+{_SIGNAL}[,.]?\s+(?:the\s+)?{_DAY}\b)", re.I),
]


def mark_self_corrections(text):
    """Wrap a replaced value in <sc>, where the replacement is unambiguous."""
    if not text or "<sc>" in text:
        return text
    out, last = "", 0
    for m in _SPAN.finditer(text):
        out += _mark_sc_free(text[last:m.start()]) + m.group(0)
        last = m.end()
    return out + _mark_sc_free(text[last:])


def _mark_sc_free(text):
    for pat in SELF_CORRECTION:
        text = pat.sub(lambda m: f"<sc>{m.group(1)}</sc>{m.group(2)}", text)
    return text


FILLER = re.compile(r"\b(um|uh|hmm|ah|er)\b", re.I)


def hesitation_plausible(text):
    """The spec's hesitation is a stall: a drawn-out word, trailing off, or a
    filler followed by a pause. A single quick "um" is a filler, not a
    hesitation - so the model's guess is only kept when the turn actually
    carries a measured pause, which is the one part of it we can see."""
    return "[pause:" in text


BACKCHANNELS = {"mm-hm", "mm-hmm", "mmhmm", "uh-huh", "yeah", "yes", "okay",
                "ok", "right", "haan", "achha", "ji", "hmm", "mhm"}


def is_backchannel(text):
    bare = re.sub(r"[\[<].*?[\]>]", "", text)
    bare = re.sub(r"\s*-\s*", "-", bare)      # whisper writes "mm -hmm"
    words = [w.strip(".,!?").lower() for w in bare.split()]
    words = [w for w in words if w]
    return 0 < len(words) <= 3 and all(w in BACKCHANNELS for w in words)


def words_only(text):
    """The bare word sequence, with every tag and all punctuation removed."""
    text = re.sub(r"<[^>]*>", " ", str(text))
    text = re.sub(r"\[[^\]]*\]", " ", text)
    return re.sub(r"[^\w]+", " ", text).lower().split()


def digits_invented(as_spoken, clean):
    """True when CLEAN contains a digit AS_SPOKEN never had.

    Only meaningful once AS_SPOKEN itself carries digits - turning "twelve
    eighty four" into 1,284.37 is the rule working, not a corruption.
    """
    src = [c for c in as_spoken if c.isdigit()]
    dst = [c for c in clean if c.isdigit()]
    if not src or not dst:
        return False
    pool = list(src)
    for c in dst:
        if c in pool:
            pool.remove(c)
        else:
            return True
    return False


# Recognising "Bangalore" as a place needs world knowledge, so it stays with
# the model - but buried in the main 40-line prompt it was missed every time.
# It gets its own small question instead, asked only when the turn actually
# holds a capitalised word that could be a place.
NOT_A_PLACE = {
    "I", "I'm", "I've", "Mr", "Mrs", "Ms", "Dr", "Mister", "Miss", "Sir",
    "Madam", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
    "Saturday", "Sunday", "January", "February", "March", "April", "May",
    "June", "July", "August", "September", "October", "November", "December",
    "Okay", "Yes", "No", "Yeah", "Hello", "Hi", "Thanks", "Thank", "Please",
    "Actually", "Uh", "Um", "So", "And", "But", "The", "PM", "AM", "OTP",
}
PLACE_CUE = re.compile(r"\b(from|in|at|near|to|visit|branch|city|located|"
                       r"based|office|hospital|clinic|airport|station)\b", re.I)


def location_candidates(text):
    bare = re.sub(r"[\[<].*?[\]>]", " ", text)
    if not PLACE_CUE.search(bare):
        return []
    out, sentence_start = [], True
    for tok in re.findall(r"[^\w']+|[\w']+", bare):
        if not tok.strip():
            continue
        if not tok[0].isalnum():
            sentence_start = bool(re.search(r"[.!?]", tok))
            continue
        if (not sentence_start and tok[0].isupper() and tok not in NOT_A_PLACE
                and tok not in out):
            out.append(tok)
        sentence_start = False
    return out


def find_locations(text):
    cands = location_candidates(text)
    if not cands:
        return []
    try:
        r = requests.post(URL, json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content":
                 'Which of the given words are names of real geographic places '
                 '(a city, state, country, area or landmark)? A company, brand, '
                 'hospital or person name is NOT a place. '
                 'Answer JSON only: {"places": ["..."]}'},
                {"role": "user", "content":
                 f"Sentence: {text}\nWords: {', '.join(cands)}"}],
            "stream": False, "format": "json",
            "options": {"temperature": 0.0, "num_ctx": 1024},
        }, timeout=120)
        got = json.loads(r.json()["message"]["content"]).get("places") or []
        return [g for g in got if isinstance(g, str) and g in cands]
    except Exception:
        return []


# Phrases from our own corrective message. If one comes back inside a turn,
# the model echoed the instruction instead of following it, and the retry is
# thrown away rather than written into the transcript.
LEAKED = ("previous answer was wrong", "CLEAN line must not be empty",
          "Answer again", "does not appear in AS_SPOKEN",
          "Copy digit sequences", "measured pause markers")


def leaked(text):
    low = str(text).lower()
    return any(p.lower() in low for p in LEAKED)


def annotate(turn_text, speaker, history, extra=""):
    prompt = (f"Previous turns for context:\n{history}\n\n"
              f"Annotate this {speaker} turn:\n{turn_text}")
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt}]
    if extra:
        # a separate message, so it reads as correction rather than as text
        # belonging to the turn
        messages.append({"role": "user", "content": extra})
    r = requests.post(URL, json={
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.1, "num_ctx": 4096},
    }, timeout=300)
    r.raise_for_status()
    return json.loads(r.json()["message"]["content"])


# Belt and braces after the LLM: a missed card number is a compliance problem
# rather than a quality one.
PII_PATTERNS = [
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"), "email"),
    (re.compile(r"\b\d{12,19}\b"), "card"),
    (re.compile(r"\b\d{10}\b"), "phone"),
]
# Context-sensitive rules: the surrounding words decide the tag.
# The window spans sentence boundaries on purpose: the agent asks for the code
# in one turn and the customer reads it back in the next, so a pattern that
# stops at the first full stop never fires on a real call.
NAME_PAT = re.compile(r"\b(?:Mr|Mrs|Ms|Mister|Miss|Dr)\.?\s+([A-Z][a-z]+)")

# A trigger word opens a window; every bare digit group inside it is tagged,
# not just the first. The customer often reads a number back across two turns
# ("it's ninety fourteen" / "so nine zero one four"), and only tagging the
# first occurrence leaves the second one exposed.
TRIGGERS = [
    (re.compile(r"\bcards?\b", re.I), "card", re.compile(r"\b\d{4}\b")),
    (re.compile(r"\baccounts?\b", re.I), "account", re.compile(r"\b\d{4}\b")),
    (re.compile(r"\b(?:code|otp|pin)\b", re.I), "otp", re.compile(r"\b\d{4,8}\b")),
]
WINDOW = 160
# A currency amount is not an account number.
MONEY = re.compile(r"[$£€₹]\s?[\d,]*\d$|\d\.$")
CVV_NEAR = re.compile(r"\b(cvv|cvc|security code)\b", re.I)
LONG_CARD = re.compile(r"(?:\d[ -]?){13,19}")


def _wrap(text, value, kind):
    """Wrap `value` unless it already sits inside a tag.

    Entity kinds are written in upper case as bare tags (<LOCATION>Bangalore
    </LOCATION>); privacy kinds keep the guidelines' <pii:...> form, which the
    formatter upper-cases for the delivery file.
    """
    if not value or f">{value}<" in text:
        return text, False
    open_tag, close_tag = (f"<{kind}>", f"</{kind}>") if kind.isupper() \
        else (f"<pii:{kind}>", "</pii>")
    # Every occurrence, not just the first: a caller often says the number
    # twice ("it's nine zero one four, so nine zero one four") and the second
    # copy is just as sensitive. Already-tagged spans are excluded by the
    # lookarounds, so this cannot double-wrap.
    pat = re.compile(r"(?<!>)\b" + re.escape(value) + r"\b(?!<)")
    new = pat.sub(f"{open_tag}{value}{close_tag}", text)
    return new, new != text


def regex_pii(text, context=""):
    """Belt and braces after the LLM: a missed card number is a compliance
    problem rather than a quality one."""
    added = []
    for pat, kind in PII_PATTERNS:
        for m in set(pat.findall(text)):
            text, ok = _wrap(text, m, kind)
            if ok:
                added.append(kind)
    for m in NAME_PAT.findall(text):
        text, ok = _wrap(text, m, "name")
        if ok:
            added.append("name")

    joined = context + " " + text
    offset = len(context) + 1
    for trig, kind, digits in TRIGGERS:
        for t in trig.finditer(joined):
            window = joined[t.end(): t.end() + WINDOW]
            for d in digits.finditer(window):
                # only tag digits that live in this turn, not in the context
                if t.end() + d.start() < offset:
                    continue
                before = window[max(0, d.start() - 2): d.start()]
                if MONEY.search(before + d.group(0)):
                    continue
                text, ok = _wrap(text, d.group(0), kind)
                if ok:
                    added.append(kind)
    return text, added


def card_with_cvv(text):
    return bool(LONG_CARD.search(text)) and bool(CVV_NEAR.search(text))


if __name__ == "__main__":
    wd = pathlib.Path(sys.argv[1])
    # Fast mode skips the LLM entirely. The delivery transcript only ever
    # shows the as-spoken line, and the tags that line carries - PII, DATE,
    # TIME, pauses, interrupts - are all produced without a model. What is
    # lost is the Clean line, <fs>/<sc> and <LOCATION>, which is why this is
    # a choice rather than the default. It takes the LLM pass, which is most
    # of the runtime, down to nothing.
    fast = "--fast" in sys.argv[2:] or os.environ.get("FAST") == "1"
    turns = json.load(open(wd / "merged_turns.json"))
    out, history, halted = [], "", None

    for i, t in enumerate(turns, 1):
        if card_with_cvv(t["text"]):
            halted = (f"HALT at turn {i} ({t['speaker']}): a full card number with "
                      f"CVV was spoken. The guidelines say flag the file; it must "
                      f"not be annotated.")
            break
        # A turn stage 7b found in another language is wrapped and passed
        # through. The English rules - fillers, false starts, the Clean line -
        # do not apply to it, and asking a 7B model to annotate a language it
        # barely reads invents more than it finds.
        if t.get("lang_certain"):
            tag = t["lang_name"]
            wrapped = f"<{tag}>{t['text']}</{tag}>"
            out.append({**t, "as_spoken": wrapped, "clean": wrapped,
                        "flags": [f"other_language:{tag}"] +
                                 (["could_not_transcribe_CHECK"]
                                  if t.get("lang_failed")
                                  else ["transliterate_to_latin_CHECK"]) +
                                 ([f"language_unnamed_candidates:"
                                   f"{t['lang_suspected']}"]
                                  if t.get("lang_suspected") else [])})
            history += f"{t['speaker']}: [{tag}]\n"
            print(f"  turn {i}/{len(turns)} {t['speaker']} [{tag}]", flush=True)
            continue

        if fast:
            text, added = regex_pii(
                mark_self_corrections(mark_false_starts(t["text"])),
                history[-300:])
            text, _ = tag_entities(text)
            if t.get("interrupt"):
                text = "[interrupt] " + text
            if t.get("cut_off") and not text.rstrip().endswith("-"):
                text = text.rstrip() + "-"
            flags = ["fast_mode: no clean line, no <fs>/<sc>/<LOCATION>"]
            for d in t.get("packet_drops", []):
                flags.append(f"possible_packet_drop@{d['t']}s")
            out.append({**t, "as_spoken": text, "clean": "", "flags": flags})
            history += f"{t['speaker']}: {text}\n"
            if i % 25 == 0 or i == len(turns):
                print(f"  turn {i}/{len(turns)}", flush=True)
            continue

        try:
            a = annotate(t["text"], t["speaker"], history[-600:])
        except Exception as e:
            a = {"as_spoken": t["text"], "clean": "", "flags": [f"llm_error: {e}"]}

        a.setdefault("as_spoken", t["text"])
        a.setdefault("clean", "")
        flags = list(a.get("flags") or [])

        # One corrective retry for the two failures that matter: a content turn
        # whose clean line was emptied by the backchannel rule, and a digit the
        # model changed. Both are silent errors a reviewer could easily miss.
        problems = []
        # The words belong to the transcription, not to the model. Its only
        # job on the as-spoken line is to add tags around them. It was seen
        # rewriting "Brooks" to "Broog" and "9-0-1-4" to "90114", which turns
        # a transcript into a paraphrase - and on a name or a card number
        # that is the one error the guidelines call unacceptable.
        if words_only(a["as_spoken"]) != words_only(t["text"]):
            problems.append("AS_SPOKEN must contain exactly the same words as "
                            "the input, in the same order. Add tags around "
                            "them; never change, add or drop a word.")
        # Measured markers belong to the pipeline, not the model. It has been
        # seen to drop a [pause: 0.7s] and put an invented [unclear] in its
        # place, which turns a measurement into a fabrication.
        want_pauses = re.findall(r"\[pause: [\d.]+s\]", t["text"])
        got_pauses = re.findall(r"\[pause: [\d.]+s\]", str(a["as_spoken"]))
        if sorted(want_pauses) != sorted(got_pauses):
            problems.append(f"AS_SPOKEN must keep the measured pause markers "
                            f"exactly: expected {want_pauses}, got {got_pauses}.")
        if "[unclear" in str(a["as_spoken"]) and "[unclear" not in t["text"]:
            problems.append("Do not invent [unclear]; it was not in the input.")
        if not str(a["clean"]).strip() and not is_backchannel(t["text"]):
            problems.append("The CLEAN line must not be empty: this turn carries "
                            "content, it is not a bare backchannel.")
        if digits_invented(str(a["as_spoken"]), str(a["clean"])):
            problems.append("A digit in CLEAN does not appear in AS_SPOKEN. Copy "
                            "digit sequences across unchanged.")
        if problems:
            try:
                b = annotate(t["text"], t["speaker"], history[-600:],
                             "Your previous answer was wrong:\n- " +
                             "\n- ".join(problems) +
                             "\nAnswer again for the same turn, corrected. "
                             "Do not repeat these instructions in your output.")
                if (str(b.get("clean", "")).strip()
                        and not leaked(b.get("as_spoken", ""))
                        and not leaked(b.get("clean", ""))):
                    a["as_spoken"] = b.get("as_spoken", a["as_spoken"])
                    a["clean"] = b["clean"]
                    flags = list(b.get("flags") or [])
            except Exception:
                pass
            if words_only(a["as_spoken"]) != words_only(t["text"]):
                # Still rewritten after a correction: keep the transcription's
                # own words and only the tags we can place ourselves.
                a["as_spoken"] = mark_false_starts(t["text"])
                flags.append("as_spoken_rewritten_CHECK")
            if sorted(re.findall(r"\[pause: [\d.]+s\]", str(a["as_spoken"]))) != sorted(want_pauses):
                flags.append("pause_markers_altered_CHECK")
            if "[unclear" in str(a["as_spoken"]) and "[unclear" not in t["text"]:
                flags.append("unclear_invented_CHECK")
            if not str(a["clean"]).strip() and not is_backchannel(t["text"]):
                flags.append("clean_line_empty_CHECK")
            if digits_invented(str(a["as_spoken"]), str(a["clean"])):
                flags.append("digits_differ_from_as_spoken_CHECK")
        ctx = history[-300:]
        a["as_spoken"], added = regex_pii(str(a["as_spoken"]), ctx)
        a["clean"], added2 = regex_pii(str(a["clean"]), ctx)
        a["as_spoken"] = mark_self_corrections(
            mark_false_starts(str(a["as_spoken"])))
        for place in find_locations(a["as_spoken"]):
            a["as_spoken"], _ = _wrap(a["as_spoken"], place, "LOCATION")
            a["clean"], _ = _wrap(a["clean"], place, "LOCATION")
        a["as_spoken"], _ = tag_entities(a["as_spoken"])
        a["clean"], _ = tag_entities(a["clean"])
        if added or added2:
            flags.append("pii_added_by_regex")
        if t.get("interrupt"):
            a["as_spoken"] = "[interrupt] " + a["as_spoken"]
        if t.get("cut_off") and not a["as_spoken"].rstrip().endswith("-"):
            a["as_spoken"] = a["as_spoken"].rstrip() + "-"
        for d in t.get("packet_drops", []):
            flags.append(f"possible_packet_drop@{d['t']}s")
        for e in t.get("events", []):
            flags.append(f"possible_noise:{e['tag']}")
        if t.get("lang_suspected"):
            flags.append(f"possible_other_language:{t['lang_suspected']}"
                         f"@{t['lang_prob']}")
        if not hesitation_plausible(t["text"]):
            flags = [f for f in flags if f != "possible_hesitation"]
        if leaked(a["as_spoken"]) or leaked(a["clean"]):
            a["as_spoken"], a["clean"] = t["text"], ""
            flags.append("llm_echoed_instructions_CHECK")
        a["flags"] = sorted(set(flags))
        out.append({**t, **a})
        history += f"{t['speaker']}: {a.get('clean', '')}\n"
        print(f"  turn {i}/{len(turns)} {t['speaker']}", flush=True)

    json.dump({"turns": out, "halted": halted},
              open(wd / "annotated.json", "w"), indent=2)
    print(f"stage6: {len(out)} turns annotated" + (f" | {halted}" if halted else ""))
    if halted:
        sys.exit(3)
