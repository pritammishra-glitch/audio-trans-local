"""The flag-the-file rules, as an automatic pre-check.

Three of the five are measurable. Track emptiness is a dB check in stage 2, and
the card-with-CVV case halts stage 6 rather than warning, because the point of
that rule is that the file should not be annotated at all.
"""


def flags(turns, duration, packet_drops):
    out = []
    unclear = sum(t["as_spoken"].count("[unclear") for t in turns)
    words = sum(len(t["as_spoken"].split()) for t in turns)
    if words and unclear / words > 0.25:
        out.append("FLAG: more than a quarter of the call is unclear")
    drop_time = sum(d["dur"] for d in packet_drops)
    if duration and drop_time / duration > 0.10:
        out.append("FLAG: packet drops cover more than 10% of the call")
    speech = sum(t["end"] - t["start"] for t in turns)
    if duration and speech / duration < 0.15:
        out.append("FLAG: audio is mostly silent")
    return out
