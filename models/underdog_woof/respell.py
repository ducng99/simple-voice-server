"""Respellings for hard names and foreign words, so a voice that reads with English letter rules says them right
(Siobhan -> Shivawn, Saoirse Ronan -> Seersha Rohnan, gnocchi -> nyocky).

A plain dictionary lookup over respell.tsv: no model call at inference. Matching is case-insensitive, accent-insensitive
and on whole words only; the longest entry wins, so "Saoirse Ronan" beats "Saoirse". A possessive 's survives
(Eoin's -> Owen's). A word joined to another by a hyphen matches alone only when it is a capitalised name and the other
part is lower case (Worcestershire-glazed). Entries flagged "c" are names whose words are all ordinary English words;
they match only when every word is capitalised as written. Single words that collide with ordinary English (Will,
Grace, Nice, sake) are not in the lexicon at all.

Lexicon rows: written<TAB>respelling<TAB>flags. Built from general categories with GLM-5.3 and checked by a second
model (tools/lexgen), never from evaluation sentences.
"""
import os
import re
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
LEXICON = os.path.join(HERE, "respell.tsv")
WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*")


def fold(s):
    """Lookup key: no accents, lower case, straight apostrophes, hyphens read as spaces."""
    s = unicodedata.normalize("NFKD", s.replace("’", "'").replace("‘", "'"))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = s.replace("ł", "l").replace("ø", "o").replace("æ", "ae").replace("œ", "oe").replace("ß", "ss").replace("đ", "d")
    return " ".join(re.sub(r"[-‐–]", " ", s).split())


class Respeller:
    def __init__(self, path=LEXICON):
        self.lex, self.maxn = {}, 1
        if not os.path.exists(path):
            return
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip() or line.startswith("#"):
                    continue
                p = line.rstrip("\n").split("\t")
                key = fold(p[0])
                self.lex[key] = (p[1], "c" in (p[2] if len(p) > 2 else ""))
                self.maxn = max(self.maxn, key.count(" ") + 1)

    def lookup(self, src):
        key = fold(src)
        hit = self.lex.get(key)
        if hit:
            return hit, ""
        if key.endswith("'s") and key[:-2] in self.lex:
            return self.lex[key[:-2]], src[-2:]
        return None, ""

    @staticmethod
    def hyphen_ok(t, a, b, src):
        """A word joined by a hyphen to another word matches alone only when it is a capitalised name and the other part
        is lower case (Worcestershire-glazed, Siobhan-led); dove-cote and Saint-Jacques stay as written."""
        for side in ((t[a - 1:a] == "-" and t[a - 2:a - 1].isalpha(), lambda: t[:a - 1].split()[-1:]),
                     (t[b:b + 1] == "-" and t[b + 1:b + 2].isalpha(), lambda: t[b + 1:].split()[:1])):
            if side[0]:
                other = (side[1]() or [""])[0]
                if not (src[:1].isupper() and other[:1].islower()):
                    return False
        return True

    def __call__(self, t):
        if not self.lex:
            return t
        t = t.replace("’", "'")
        ws = list(WORD.finditer(t))
        out, pos, i = [], 0, 0
        while i < len(ws):
            for n in range(min(self.maxn, len(ws) - i), 0, -1):
                span = ws[i:i + n]
                if any(t[a.end():b.start()] not in (" ", "-") for a, b in zip(span, span[1:])):
                    continue
                a, b = span[0].start(), span[-1].end()
                src = t[a:b]
                if not self.hyphen_ok(t, a, b, src):
                    continue
                (hit, tail) = self.lookup(src)
                if not hit:
                    continue
                rep, caps = hit
                if caps and not all(w.group(0)[:1].isupper() for w in span):
                    continue
                if src[:1].isupper() and rep[:1].islower():
                    rep = rep[:1].upper() + rep[1:]
                out.append(t[pos:span[0].start()] + rep + tail)
                pos, i = span[-1].end(), i + n
                break
            else:
                i += 1
        return "".join(out) + t[pos:]
