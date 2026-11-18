"""Split input text into utterances before synthesis: every sentence is its own utterance, and a single sentence
longer than CLAUSE_MAX characters is split at clause boundaries (comma, semicolon, colon, dash), merging clauses
greedily so no piece exceeds CLAUSE_MAX. A piece that is still too long and has no clause punctuation is split once
before a conjunction ("and", "that", "which", ...) close to its middle. Standard library only."""
import re

CLAUSE_MAX = 120
FIRST_MAX = int(__import__("os").environ.get("WOOF_FIRST_MAX", "0"))   # >0: split the first utterance at a clause boundary when longer than this (tested at 70; off by default)
FIRST_MIN = 20      # ...but never into a first piece shorter than this
MIN_PIECE = 30
CONJ = r"\b(?:and|but|or|nor|that|which|who|whom|whose|when|where|while|because|although|though|so|if|as|than|until)\b"


def _split_conj(s, clause_max):
    if len(s) <= clause_max:
        return [s]
    cands = [m.start() for m in re.finditer(r"\s(?=" + CONJ + ")", s)]
    cands = [c for c in cands if c >= MIN_PIECE and len(s) - c >= MIN_PIECE]
    if not cands:
        return [s]
    c = min(cands, key=lambda i: abs(i - len(s) / 2))
    left = _split_conj(s[:c].strip(), clause_max)
    if not left[-1].endswith((",", ";", ":", "\u2014", "--")):
        left[-1] += ","
    return left + _split_conj(s[c:].strip(), clause_max)


def _split_first(s, first_max):
    """split the first utterance at the last clause boundary that keeps it within first_max characters (or the
    first boundary after that), leaving pieces of at least FIRST_MIN characters."""
    if len(s) <= first_max:
        return [s]
    cuts = [m.end() for m in re.finditer(r"[,;:](?=\s)|—|--", s)]
    q = [i for i, ch in enumerate(s) if ch in '"“”']
    cuts = [c for c in cuts if c >= FIRST_MIN and len(s) - c >= FIRST_MIN
            and not any(abs(i - c) < 12 for i in q)]          # not next to a quote mark ("Ah, | yes," style cuts)
    if not cuts:
        return [s]
    inside = [c for c in cuts if c <= first_max]
    c = max(inside) if inside else min(cuts)
    return [s[:c].strip(), s[c:].strip()]


def split_text(text, clause_max=CLAUSE_MAX, first_max=FIRST_MAX):
    out = _split_text(text, clause_max)
    if out and first_max:
        out = _split_first(out[0], first_max) + out[1:]
    return out


def _split_text(text, clause_max=CLAUSE_MAX):
    out = []
    for s in re.split(r"(?<=[.!?…])\s+", text.strip()):
        if not s:
            continue
        if len(s) <= clause_max:
            out.append(s)
            continue
        clauses = [c for c in re.split(r"(?<=[,;:])\s+|(?<=—)\s*|(?<=--)\s*", s) if c]
        merged, cur = [], ""
        for c in clauses:
            if cur and len(cur) + 1 + len(c) > clause_max:
                merged.append(cur)
                cur = c
            else:
                cur = c if not cur else (cur + c if cur.endswith(("—", "--")) else cur + " " + c)
        if cur:
            merged.append(cur)
        for m in merged:
            out += _split_conj(m, clause_max)
    return out
