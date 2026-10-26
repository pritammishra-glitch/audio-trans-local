"""<DATE>, <TIME> and <LOCATION> entity tags.

These are not in the annotation guidelines' tag set - they come from the
requested output format. Dates and times are handled here by regex, because
they have hard surface forms; locations are left to the LLM, because
recognising "Bangalore" as a place needs world knowledge.

Matching only ever happens on text that is not already inside a tag, so
<pii:...>, <fs>, [pause: ...] and entity tags never nest or double-wrap.
"""
import re

DAYS = r"Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday"
MONTHS = (r"January|February|March|April|May|June|July|August|September|"
          r"October|November|December")

# Tried in order; earlier patterns win, so the longest forms come first.
PATTERNS = [
    # AM/PM dots are paired, so a sentence-final "7:00 PM." keeps its period
    # outside the tag.
    ("TIME", re.compile(r"\b(\d{1,2}:\d{2}(?::\d{2})?(?:\s*(?:[AaPp]\.[Mm]\.|[AaPp][Mm]))?)")),
    ("TIME", re.compile(r"\b(\d{1,2}\s*(?:[AaPp]\.[Mm]\.|[AaPp][Mm]))(?!\w)")),
    ("TIME", re.compile(r"\b(\d{1,2}\s*o'?\s?clock(?:\s*(?:[AaPp]\.[Mm]\.|[AaPp][Mm]))?)", re.I)),
    # Whisper writes "at around 3:30" as "3 .30". Guarded so it cannot eat the
    # decimal part of an amount: the character before must not be a digit,
    # a currency symbol or a dot.
    ("TIME", re.compile(r"(?<![\d$£€₹.])\b(\d{1,2}\s?\.\s?[0-5]\d)\b(?!\d)")),
    ("TIME", re.compile(r"\b(noon|midnight)\b", re.I)),
    ("DATE", re.compile(r"\b(day after tomorrow|day before yesterday)\b", re.I)),
    ("DATE", re.compile(r"\b(today|tomorrow|yesterday|tonight)\b", re.I)),
    ("DATE", re.compile(r"\b(weekends?|weekdays?)\b", re.I)),
    ("DATE", re.compile(rf"\b((?:next |last |this )?(?:{DAYS}))\b", re.I)),
    ("DATE", re.compile(rf"\b((?:{MONTHS})\s+\d{{1,2}}(?:st|nd|rd|th)?)\b", re.I)),
    ("DATE", re.compile(rf"\b(\d{{1,2}}(?:st|nd|rd|th)\s+(?:of\s+)?(?:{MONTHS}))\b", re.I)),
    ("DATE", re.compile(r"\b(\d{1,2}/\d{1,2}/\d{2,4})\b")),
]

# An existing tag, or anything wrapped in one, is off limits.
PROTECTED = re.compile(r"<([^>\s/]+)[^>]*>.*?</\1>|<[^>]*>|\[[^\]]*\]", re.S)


def _tag_free(text, patterns):
    """Tag a stretch of text known to contain no tags."""
    if not text or not patterns:
        return text, []
    kind, pat = patterns[0]
    out, added, last = [], [], 0
    for m in pat.finditer(text):
        out.append(("free", text[last:m.start()]))
        out.append(("done", f"<{kind}>{m.group(1)}</{kind}>"))
        added.append(kind)
        last = m.end()
    out.append(("free", text[last:]))

    result = ""
    for what, s in out:
        if what == "free":
            s, more = _tag_free(s, patterns[1:])
            added += more
        result += s
    return result, added


def tag_entities(text):
    """Wrap dates and times. Returns (tagged_text, kinds_added)."""
    if not text:
        return text, []
    result, added, last = "", [], 0
    for m in PROTECTED.finditer(text):
        free, more = _tag_free(text[last:m.start()], PATTERNS)
        result += free + m.group(0)
        added += more
        last = m.end()
    free, more = _tag_free(text[last:], PATTERNS)
    return result + free, added + more
