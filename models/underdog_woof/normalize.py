"""Underdog English text front end: our rules for the cases general normalizers get wrong, then NeMo text processing
(Apache 2.0, optional: pip install nemo_text_processing) for whatever digits or symbols remain. Rules are written from how people read these strings aloud, and tuned on our own
hard-text dev sets (tn_dev.txt, pron_dev.jsonl), never on benchmark sentences.
"""
import re

DIG = "zero one two three four five six seven eight nine".split()
LETTER = {c: c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}


def digits(s):
    return " ".join(DIG[int(c)] for c in s if c.isdigit())


def spell(s):
    out = []
    for c in s:
        if c.isdigit():
            out.append(DIG[int(c)])
        elif c.isalpha():
            out.append(c.upper())
    return " ".join(out)


def pronounceable(w):
    """All-caps chunk that reads as a word (SAVE, WINTER, PROMO) rather than letters (ORD, XT, CS)."""
    w = w.lower()
    if len(w) < 4 or not re.search(r"[aeiouy]", w):
        return False
    if re.search(r"[^aeiouy]{3,}", w) and not re.search(r"(str|sch|thr|chr|spr|scr|ght|nds|nts)", w):
        return False
    return bool(re.fullmatch(r"(?:[^aeiouy]{0,3}[aeiouy]{1,2})+[^aeiouy]{0,3}", w))


def code_read(s):
    out = []
    for p in re.split(r"[-/]", s):
        if not p:
            continue
        if p.isdigit():
            out.append(p if len(p) <= 2 and p[0] != "0" else digits(p))
        elif p.isalpha() and p.islower() and len(p) > 3 and pronounceable(p):
            out.append(p)
        elif p.isalpha() and pronounceable(p):
            out.append(p.capitalize())
        else:
            out.append(spell(p))
    return " ".join(out)


def nread(s):
    """Digits of an identifier read the way people say room, flight and house numbers: 402 -> four oh two,
    229 -> two twenty nine, 1204 -> twelve oh four, 1600 -> sixteen hundred; 5+ digits or a leading 0 go digit by digit."""
    n = int(s)
    if s[0] == "0" and len(s) > 1 or len(s) > 4:
        return digits(s)
    if n < 100 or n % 100 == 0 and len(s) == 3 or n % 1000 == 0:
        return cardinal(n)
    hi, lo = divmod(n, 100)
    return cardinal(hi) + (" hundred" if lo == 0 else " oh " + ONES[lo] if lo < 10 else " " + cardinal(lo))


def split_ident(s):
    """getUserName -> get user name, user_id -> user id, OpenAI -> Open A I."""
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", s.replace("_", " "))
    return " ".join(spell(w) if w.isupper() and len(w) <= 4 else w for w in s.split())


ROMAN_RE = r"(?=[MDCLXVI]+\b)M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})"
# words after which a Roman numeral is a plain count (Chapter XIV, Super Bowl LIX, World War II, Type II)
ROMAN_CARD = set("""chapter chap ch part pt book volume vol act scene section sec phase stage level episode season article
    title class type grade war bowl apollo mark mk vatican rocky fantasy wars trek civilization gemini saturn soyuz annex
    appendix figure fig table plate tier round schedule unit division category series olympiad psalm
    canto lesson module form""".split())
ROMAN_CARD |= {w + "s" for w in ROMAN_CARD} | {"through", "thru"}
ROMAN_LABEL = set("appendix annex table figure fig plate schedule unit category class type grade tier section form division series".split())
ROMAN_STOP = set("the a an in on of and for to at by from with my our his her their this that its".split())
# names that take a regnal ordinal (Henry VIII -> Henry the eighth); any Title word before II+ also does (John Smith III)
REGNAL = set("""Henry Louis Elizabeth George Edward Richard Charles James William John Paul Benedict Pius Leo Gregory
    Innocent Clement Boniface Urban Alexander Peter Ivan Nicholas Frederick Wilhelm Ludwig Philip Felipe Carlos Gustav
    Gustaf Christian Olaf Haakon Harald Ramesses Ramses Thutmose Amenhotep Ptolemy Darius Xerxes Cyrus Constantine
    Justinian Mehmed Suleiman Selim Murad Napoleon Umberto Mary Anne Victoria Catherine Isabella Margaret Rama Abdullah
    Robert David Kenneth Duncan Alfonso Ferdinand Sancho Manuel Sebastian Stephen Matthias Sigismund Casimir Otto Conrad
    Rudolf Maximilian Leopold Joseph Francis Franz Karl Hugh Baldwin Basil Michael Theodore Edmund Harold Magnus Eric
    Erik Valdemar Frederik Albert Baudouin Philippe Rainier Hassan Faisal Fuad Pedro Afonso Jaime Sixtus Julius Adrian
    Hadrian Celestine Honorius Lothair Pepin Antiochus Seleucus Herod Cleopatra Wenceslaus Kamehameha Tewodros Menelik
    Nebuchadnezzar Shalmaneser Sargon Tiglath Artaxerxes Andronikos Romanos Leon Mohammed Muhammad Charlemagne""".split())
TITLES = set("King Queen Pope Emperor Empress Tsar Czar Tsarina Kaiser Pharaoh Sultan Shah Prince Princess Duke Duchess "
             "Count Countess Lord Lady Saint St Archduke Elector Emir Caliph Khan King's Queen's".split())


def roman(s):
    v = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    return sum(-v[a] if b and v[b] > v[a] else v[a] for a, b in zip(s, list(s[1:]) + [None]))


def roman_sub(m):
    prev, num = m.group(1), m.group(3)
    if not num:
        return m.group(0)
    before = m.string[:m.start()].split()
    prev2 = before[-1] if before and not re.search(r"[.,;:!?)\"]$", before[-1]) else ""
    key, n, title = prev.lower().rstrip("."), roman(m.group(3)), lambda w: bool(re.fullmatch(r"[A-Z][a-z]+", w))
    if key in ("through", "thru", "to", "and", "or") and not re.search(r"\b(?:" + "|".join(ROMAN_CARD) + r")\s+\S+\s+$", m.string[:m.start()], re.I):
        return m.group(0)
    if key in ROMAN_CARD and (len(num) > 1 or num == "I" and prev[:1].isupper() or num in "VX" and key not in ROMAN_LABEL):
        return f"{prev}{m.group(2)}{cardinal(n)}"
    if not re.fullmatch(r"[IVXL]+", num):
        return m.group(0)
    if prev in REGNAL or prev2 in TITLES and title(prev) and num != "I":
        return f"{prev}{m.group(2)}the {ordinal(n)}"
    if len(num) > 1 and title(prev) and not re.fullmatch(r"[IVXL]*L[IVXL]*", num):
        # John Smith III -> the third; Frozen II, Rocky IV -> two, four
        full = title(prev2) and prev2.lower() not in ROMAN_STOP
        return f"{prev}{m.group(2)}" + (f"the {ordinal(n)}" if full else cardinal(n))
    return m.group(0)


CURCODE = {"USD": "$", "US$": "$", "EUR": "€", "GBP": "£", "JPY": "yen", "¥": "yen", "CNY": "yuan", "RMB": "yuan",
           "INR": "rupees", "₹": "rupees", "Rs.": "rupees", "Rs": "rupees", "CAD": "Canadian dollars", "C$": "Canadian dollars",
           "AUD": "Australian dollars", "A$": "Australian dollars", "NZD": "New Zealand dollars", "CHF": "Swiss francs",
           "HKD": "Hong Kong dollars", "HK$": "Hong Kong dollars", "SGD": "Singapore dollars", "KRW": "won", "₩": "won",
           "MXN": "pesos", "BRL": "reais", "R$": "reais", "RUB": "rubles", "₽": "rubles", "SEK": "Swedish kronor",
           "NOK": "Norwegian kroner", "DKK": "Danish kroner", "ZAR": "rand", "₦": "naira", "NGN": "naira", "₺": "lira",
           "₱": "pesos", "₫": "dong", "BTC": "bitcoin", "ETH": "ether", "₿": "bitcoin"}
SCALE = {"k": "thousand", "K": "thousand", "m": "million", "M": "million", "mn": "million", "mm": "million",
         "bn": "billion", "b": "billion", "B": "billion", "tn": "trillion", "T": "trillion", "trn": "trillion",
         "thousand": "thousand", "million": "million", "billion": "billion", "trillion": "trillion"}
FRAC_DEN = {2, 3, 4, 5, 6, 8, 10, 16, 32, 64, 100}
FRAC_AFTER = (r"\s*(?:of\b|an?\b|cups?|c\.|teaspoons?|tsp|tablespoons?|tbsp|inch|inches|in\b|in\.|\"|miles?|mi\b|"
              r"pounds?|lbs?|ounces?|oz|hours?|hrs?|minutes?|mins?|seconds?|pints?|quarts?|gallons?|liters?|litres?|"
              r"kilos?|kg|feet|foot|ft|yards?|acres?|sticks?|pieces?|slices?|loaf|tons?|percent|points?|shares?|"
              r"the\b|times\b|marathon|price|full|empty|done|finished|complete)")
DATE_BEFORE = (r"(?:\b(?:on|by|until|till|from|since|before|after|due|dated?|deadline|through|thru|starting|effective|"
               r"expires?|expiry|born|died|week of|shipped|delivered|arrives?|arriving|ordered|paid|placed|scheduled|"
               r"filed|signed|posted|submitted|opens?|closes?|ends?|starts?|begins?|[Mm]onday|[Tt]uesday|[Ww]ednesday|[Tt]hursday|[Ff]riday|[Ss]aturday|"
               r"[Ss]unday|Mon|Tue|Wed|Thu|Fri|Sat|Sun)\.?[,:]?\s*)$")


def frac_words(a, b, lead=False):
    """3/4 -> three quarters; 1/2 -> one half; with lead (the 1 1/2 case): a half, a third."""
    if b == 2:
        name = "half" if a == 1 else "halves"
    else:
        name = ("quarter" if b == 4 else ordinal(b)) + ("" if a == 1 else "s")
    return ("a" if lead and a == 1 else cardinal(a)) + " " + name


ABBR = [
    (r"\bDr\.(?=\s+[A-Z])", "Doctor"), (r"\bMr\.", "Mister"), (r"\bMrs\.", "Missus"), (r"\bMs\.", "Miz"),
    (r"\bProf\.(?=\s+[A-Z])", "Professor"), (r"\bSt\.(?=\s+[A-Z])", "Saint"), (r"(?<=[a-z] )St\.", "Street"),
    (r"\bRd\.", "Road"), (r"\bAve\.", "Avenue"), (r"\bBlvd\.", "Boulevard"), (r"\bApt\.", "Apartment"),
    (r"\b[Aa]pprox\.", "approximately"), (r"\betc\.", "et cetera"), (r"\be\.g\.", "for example"), (r"\bi\.e\.", "that is"),
    (r"\bvs\.?(?=\s)", "versus"), (r"\b[Nn]o\.(?=\s*\d)", "number"), (r"\bdept\.", "department"), (r"\bext\.", "extension"),
    (r"(?<=\d)\s?min\.", " minutes"), (r"\bmin\.(?=\s+[a-z])", "minimum"), (r"\bmin\.", "minutes"),
    (r"\bhr?s?\.(?=\s)", "hours"), (r"(?<![\d\s])\s*\b[Ss]ec\.(?=\s*\d)", " section"), (r"\b([Ee])st\.(?=\s*\d{4})", r"\1stablished"),
    (r"\bSgt\.", "Sergeant"), (r"\bCapt\.", "Captain"), (r"\bLt\.", "Lieutenant"), (r"\bCol\.", "Colonel"),
    (r"\bGen\.(?=\s+[A-Z])", "General"), (r"\bGov\.", "Governor"), (r"\bSen\.", "Senator"), (r"\bRep\.(?=\s+[A-Z])", "Representative"),
    (r"\bRev\.", "Reverend"), (r"\bHon\.", "Honorable"), (r"\bJr\.", "Junior"), (r"\bSr\.", "Senior"),
    (r"\bLn\.", "Lane"), (r"\bDr\.(?=[\s,.]|$)", "Drive"), (r"\bCt\.", "Court"), (r"\bPl\.", "Place"), (r"\bSq\.", "Square"),
    (r"\bHwy\.", "Highway"), (r"\bPkwy\.", "Parkway"), (r"\bMt\.", "Mount"), (r"\bFt\.(?=\s+[A-Z])", "Fort"),
    (r"\bSte\.", "Suite"), (r"\bBldg\.", "Building"), (r"\bfl\.", "floor"), (r"\bFl\.", "Floor"), (r"\bRm\.", "Room"),
    (r"\bEst\.", "Estimated"), (r"\best\.", "estimated"), (r"\bInc\.", "Incorporated"), (r"\bLtd\.", "Limited"),
    (r"\bCo\.(?=\s)", "Company"), (r"\bCorp\.", "Corporation"), (r"\bDept\.", "Department"), (r"\bAssn\.", "Association"),
    (r"\bUniv\.", "University"), (r"\bqty\.?", "quantity"), (r"\bref\.", "reference"), (r"\btel\.", "telephone"),
    (r"\bTel\.", "Telephone"), (r"\bfig\.", "figure"), (r"\bvol\.", "volume"), (r"\bpp\.", "pages"), (r"\bp\.(?=\s*\d)", "page"),
    (r"\bwk\.", "week"), (r"\bmo\.", "month"), (r"\byr\.", "year"), (r"\bavg\.", "average"), (r"\bmax\.", "maximum"),
]
DAYS = {"Mon": "Monday", "Tue": "Tuesday", "Tues": "Tuesday", "Wed": "Wednesday", "Thu": "Thursday", "Thur": "Thursday",
        "Thurs": "Thursday", "Fri": "Friday", "Sat": "Saturday", "Sun": "Sunday"}
MON_ABBR = {"Jan": "January", "Feb": "February", "Mar": "March", "Apr": "April", "Jun": "June", "Jul": "July",
            "Aug": "August", "Sep": "September", "Sept": "September", "Oct": "October", "Nov": "November", "Dec": "December"}
SYMBOLS = [
    (r"\s*±\s*", " plus or minus "), (r"\s*≥\s*", " greater than or equal to "), (r"\s*≤\s*", " less than or equal to "),
    (r"\s*≠\s*", " not equal to "), (r"\s*≈\s*", " approximately "), (r"\s*÷\s*", " divided by "),
    (r"\s*→\s*", " to "), (r"\s*√\s*", " square root of "), (r"(?<=\d)\s*%", " percent"), (r"\s*‰", " per mille"),
    (r"\s+=\s+", " equals "), (r"(?<=\d)\s*<\s*(?=\d)", " less than "), (r"(?<=\d)\s*>\s*(?=\d)", " greater than "),
    (r"\s*&\s*", " and "),
]
UNITS = {"km": "kilometers", "m": "meters", "cm": "centimeters", "mm": "millimeters", "kg": "kilograms", "g": "grams",
         "mg": "milligrams", "l": "liters", "L": "liters", "ml": "milliliters", "mL": "milliliters", "km/h": "kilometers per hour",
         "mph": "miles per hour", "GB": "gigabytes", "MB": "megabytes", "TB": "terabytes", "kB": "kilobytes", "GHz": "gigahertz",
         "MHz": "megahertz", "kW": "kilowatts", "kWh": "kilowatt hours", "W": "watts", "V": "volts", "mAh": "milliamp hours",
         "ft": "feet", "in": "inches", "lb": "pounds", "lbs": "pounds", "oz": "ounces", "h": "hours", "min": "minutes",
         "s": "seconds", "ms": "milliseconds", "mi": "miles", "yd": "yards", "yds": "yards", "Hz": "hertz", "kHz": "kilohertz",
         "dB": "decibels", "mpg": "miles per gallon", "kph": "kilometers per hour", "tsp": "teaspoons", "tbsp": "tablespoons",
         "gal": "gallons", "qt": "quarts", "mcg": "micrograms", "µg": "micrograms", "μg": "micrograms", "nm": "nanometers",
         "µm": "micrometers", "μm": "micrometers", "ns": "nanoseconds", "µs": "microseconds", "μs": "microseconds",
         "sec": "seconds", "secs": "seconds", "Mbps": "megabits per second", "Gbps": "gigabits per second",
         "kbps": "kilobits per second", "kcal": "kilocalories", "mmHg": "millimeters of mercury", "dL": "deciliters",
         "dl": "deciliters", "PB": "petabytes", "MW": "megawatts", "GW": "gigawatts", "GWh": "gigawatt hours",
         "MWh": "megawatt hours", "kPa": "kilopascals", "hPa": "hectopascals", "mmol": "millimoles", "fl oz": "fluid ounces",
         "sq ft": "square feet", "sq m": "square meters", "km2": "square kilometers", "m2": "square meters", "cc": "C C",
         "mA": "milliamps", "ng": "nanograms", "pg": "picograms", "IU": "I U", "mEq": "milliequivalents", "kN": "kilonewtons", "Nm": "newton meters", "TFLOPS": "teraflops", "rpm": "R P M", "psi": "P S I"}
SING = {"feet": "foot", "inches": "inch", "hertz": "hertz", "kilohertz": "kilohertz", "teaspoons": "teaspoon",
        "C C": "C C", "R P M": "R P M", "P S I": "P S I", "teraflops": "teraflop", "I U": "I U"}


def sing(w):
    head, sep, rest = w.partition(" per ")
    if " of " in head:
        return w
    words = head.split(" ")
    words[-1] = SING.get(words[-1], words[-1][:-1] if words[-1].endswith("s") else words[-1])
    return " ".join(words) + sep + rest


def unit(k, n):
    return sing(UNITS[k]) if n == "1" else UNITS[k]


MON_RE = r"(January|February|March|April|May|June|July|August|September|October|November|December)"
STREET = {"St": "Street", "Rd": "Road", "Ave": "Avenue", "Av": "Avenue", "Blvd": "Boulevard", "Ln": "Lane", "Dr": "Drive",
          "Ct": "Court", "Pl": "Place", "Sq": "Square", "Hwy": "Highway", "Pkwy": "Parkway", "Cres": "Crescent",
          "Ter": "Terrace", "Terr": "Terrace", "Cir": "Circle", "Trl": "Trail", "Pk": "Park"}
STREET_RE = (r"(Street|Road|Avenue|Boulevard|Lane|Drive|Court|Place|Square|Highway|Parkway|Crescent|Terrace|Circle|Way|"
             r"Close|Row|Walk|Alley|Plaza|Trail|Pike|Loop|" + "|".join(STREET) + r")\b(\.?)")
# words after which a 2-4 digit number is an identifier read in pairs (room 402 -> four oh two, gate 47B -> forty seven B)
ID_WORDS = (r"[Rr]oom|[Rr]m\.?|[Oo]ffice|[Ss]uite|[Ss]te\.?|[Aa]partment|[Aa]pt\.?|[Uu]nit|[Gg]ate|[Pp]latform|[Ff]light|"
            r"[Bb]us|[Rr]oute|[Hh]ighway|[Hh]wy\.?|[Ii]nterstate|[Tt]erminal|[Bb]ay|[Dd]ock|[Pp]ier|[Ss]eat|[Cc]abin|"
            r"[Ww]ard|[Ll]ocker|[Bb]ooth|[Ss]tudio|[Bb]uilding|[Bb]ldg\.?|[Ee]xit|[Pp]ort|[Ee]xt\.?|[Ee]xtension|"
            r"[Mm]odel|[Bb]oeing|[Aa]irbus|[Cc]heckpoint|[Hh]angar|[Tt]rain|[Cc]oach|[Ff]erry|[Cc]ar")
ELEMENTS = set("""H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y
    Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg
    Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr""".split())
CODE_CTX = (r"(?i:\b(?:codes?|ref(?:erence)?|serial|PIN|password|passcode|quote|plate|registration|booking|confirmation|"
            r"tracking|order|voucher|coupon|promo|ticket|ID|key|token|account|policy|claim|case|invoice|SKU|part)\b)[^.;!?]*$")
# URL schemes read as words; the rest are spelled unless pronounceable
SCHEME_SAY = {"http": "H T T P", "https": "H T T P S", "file": "file", "git": "git", "data": "data", "blob": "blob",
              "chrome": "chrome", "spotify": "spotify", "slack": "slack", "zoom": "zoom", "steam": "steam"}
UNIT_PRE = r"(?<![\w.])(?<!\w-)(\d(?:[\d,]*\d)?(?:\.\d+)?)"


def handle(h):
    """@openai -> open A I, @Jane_Doe -> Jane underscore Doe."""
    out = []
    for p in h.split("_"):
        m = re.fullmatch(r"([a-z]{3,}[nlyex])(ai|hq)", p)
        out.append(m.group(1) + " " + spell(m.group(2)) if m else split_ident(p))
    return " underscore ".join(x for x in out if x)


def norm_tokens(t):
    """Rules that must see the raw string before the general passes: symbols outside the model's alphabet, money with
    scale words and ranges, decades, Roman numerals, powers, durations, code identifiers, hashtags and handles."""
    t = t.replace("−", "-").replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    t = re.sub(r"(?<=\d)([½¼¾⅓⅔⅛])", r" \1", t)
    t = re.sub(r"[½¼¾⅓⅔⅛]", lambda m: {"½": "1/2", "¼": "1/4", "¾": "3/4", "⅓": "1/3", "⅔": "2/3", "⅛": "1/8"}[m.group(0)], t)
    t = re.sub(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁻]+", lambda m: "^" + m.group(0).translate(str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")), t)
    # tech tokens: .NET, C#, C++, getUserName()
    t = re.sub(r"(?<![\w.])([A-Za-z]*)\.NET\b", lambda m: (spell(m.group(1)) + " " if m.group(1) else "") + "dot net", t)
    t = re.sub(r"\b([A-Za-z])#(?!\w)", r"\1 sharp", t)
    t = re.sub(r"\b([A-Za-z])\+\+(?!\w)", r"\1 plus plus", t)
    t = re.sub(r"\b([A-Za-z_][A-Za-z0-9_]*)\(\)", lambda m: split_ident(m.group(1)), t)
    t = re.sub(r"\b[a-z]{2,}(?:[A-Z][a-z]+){2,}\b", lambda m: split_ident(m.group(0)), t)
    t = re.sub(r"\bv(\d+(?:\.\d+)+)\b", lambda m: "version " + " point ".join(m.group(1).split(".")), t)
    # hashtags and handles
    t = re.sub(r"(?<![\w&#])#([A-Za-z][A-Za-z_]*)\b(?![\d-])", lambda m: "hashtag " + split_ident(m.group(1)), t)
    t = re.sub(r"(?<![\w.@])@([A-Za-z_][A-Za-z0-9_]{1,30})\b", lambda m: "at " + handle(m.group(1)), t)
    # UK and Canadian postcodes: SW1A 1AA, M14 5TG, K1A 0B6
    t = re.sub(r"\b([A-Z]{1,2}\d[A-Z\d]?|[A-Z]\d[A-Z])\s(\d[A-Z]{2}|\d[A-Z]\d)\b", lambda m: spell(m.group(1)) + ", " + spell(m.group(2)), t)
    # digits that are read one by one: card ending in 4459, PIN 0420
    t = re.sub(r"\b((?:ending|ends) in|last (?:four|4)(?: digits)?(?: of)?|PIN|[Pp]in|passcode|OTP|[Cc]ode)(\s+)(\d{3,8}(?:\s\d{2,4})*)\b(?![.,]\d|-\w)",
               lambda m: m.group(1) + m.group(2) + ", ".join(digits(g) for g in m.group(3).split()), t)
    # paths and code: C:\Users, SELECT *, .env, API_KEY
    t = re.sub(r"\b([A-Z]):\\", r"\1 colon backslash ", t)
    t = re.sub(r"\\", " backslash ", t)
    t = re.sub(r"(?<=\s)\*(?=\s)(?!\s*\d)", "star", t)
    t = re.sub(r"(?<![\w.])\.(env|gitignore|bashrc|zshrc|npmrc|htaccess|dockerignore)\b", r"dot \1", t)
    t = re.sub(r"\b(?=\w*_)(?![\w]*\.\w)[A-Za-z0-9]+(?:_[A-Za-z0-9]+)+\b", lambda m: " underscore ".join(
        spell(p) if re.search(r"\d", p) and re.search(r"[A-Za-z]", p) else p for p in m.group(0).split("_")), t)
    # US toll-free and long-distance numbers: 1-800-829-1040 -> one eight hundred, eight two nine, one zero four zero
    t = re.sub(r"(?<![\w+])1[-. ](\d{3})[-. ](?:(\d{3})[-. ](\d{4})|([A-Z]{3,}(?:-[A-Z]+)*))\b",
               lambda m: "one " + (cardinal(int(m.group(1))) if m.group(1) in ("800", "900") else digits(m.group(1))) + ", "
               + (digits(m.group(2)) + ", " + digits(m.group(3)) if m.group(2) else " ".join(w.capitalize() for w in m.group(4).split("-"))), t)
    # money: currency codes and other symbols, ranges, scale words ($4.2 million, $50-75, EUR 12.99, ¥300)
    NUM, SC = r"(\d[\d,]*(?:\.\d+)?)", r"(\s?(?:thousand|million|billion|trillion)\b|(?:k|K|mn|m|M|bn|b|B|tn|T)\b)?"
    codes = "|".join(re.escape(k) for k in sorted(CURCODE, key=len, reverse=True))
    def cur(c, num, sc):
        name = CURCODE[c]
        if len(name) == 1:
            return name + num + (sc or "")
        whole, _, cents = num.partition(".")
        if not sc and len(cents) == 2 and name.endswith("dollars"):
            return f"{whole} {name}" + (f" and {int(cents)} cents" if int(cents) else "")
        return f"{num}{' ' + SCALE[sc.strip()] if sc else ''} {name}"
    t = re.sub(r"(?<![\w$])(" + codes + r")\s?" + NUM + SC, lambda m: cur(m.group(1), m.group(2), m.group(3)), t)
    t = re.sub(NUM + SC + r"\s?(USD|EUR|GBP|JPY|CNY|INR|CAD|AUD|CHF|NZD|HKD|SGD|KRW|MXN|BRL|RUB|SEK|NOK|DKK|ZAR|BTC|ETH)\b",
               lambda m: cur(m.group(3), m.group(1), m.group(2)), t)
    t = re.sub(r"(?<!\w)(\d+)\s?¢", r"\1 cents", t)
    cw = {"$": "dollars", "€": "euros", "£": "pounds"}
    def mrange(m):
        c, a, s1, b, s2 = m.groups()
        if not (s1 or s2) and re.search(r"\.\d\d\b", a + " " + b):
            return f"{c}{a} to {c}{b}"
        s1 = " " + SCALE[s1.strip()] if s1 else ""
        s2 = " " + SCALE[s2.strip()] if s2 else ""
        return f"{a}{s1 if s1 != s2 else ''} to {b}{s2} {cw[c]}"
    t = re.sub(r"([$€£])" + NUM + SC + r"\s?[-–]\s?[$€£]?" + NUM + SC, mrange, t)
    t = re.sub(r"([$€£])" + NUM + r"(\s?(?:thousand|million|billion|trillion)\b|(?:k|K|mn|m|M|bn|b|B|tn|T)\b)",
               lambda m: f"{m.group(2)} {SCALE[m.group(3).strip()]} {cw[m.group(1)]}", t)
    # durations: 2h 15m, 1h30, 6m 30s; a bare 45m after for/last/takes is minutes
    pl = lambda n, w: f"{n} {w}{'' if n == '1' else 's'}"
    t = re.sub(r"\b(\d+)\s?h(?:rs?)?\s?(\d{1,2})(?:\s?m(?:ins?)?)?\b", lambda m: pl(m.group(1), "hour") + " " + pl(m.group(2), "minute"), t)
    t = re.sub(r"\b(\d+)\s?m(?:ins?)?\s?(\d{1,2})\s?s(?:ecs?)?\b", lambda m: pl(m.group(1), "minute") + " " + pl(m.group(2), "second"), t)
    t = re.sub(r"(\b(?:for|last|lasts|lasted|lasting|takes?|took|taking|every|within|wait|waited|remaining|runtime|ETA|"
               r"blocked)\s+(?:\w+\s+){0,2}?)(\d{1,3})m\b(?!\s*(?:dash|sprint|race|swim|hurdles|pool|freestyle|relay))",
               lambda m: m.group(1) + pl(m.group(2), "minute"), t)
    # decades: '80s, 1990s, the 2000s; '98 -> ninety eight
    def decade(m):
        s, n = m.group(2), int(m.group(2))
        if n % 10:
            return m.group(0)
        w = (year(n) if len(s) == 4 else "two thousand" if n == 0 and m.group(1) else cardinal(n) if n else "")
        if not w:
            return m.group(0)
        return w[:-1] + "ies" if w.endswith("y") else w + "s"
    t = re.sub(r"(?<![\w$€£.,'])(')?(\d{2}|1\d{3}|20\d{2})'?s\b", decade, t)
    t = re.sub(r"(?:(?<=\s)|^)'(\d{2})\b(?!['\w])", lambda m: ("oh " + ONES[int(m.group(1))]) if m.group(1)[0] == "0" else m.group(1), t)
    # Roman numerals by context: Henry VIII -> Henry the eighth, Chapter XIV -> Chapter fourteen
    t = re.sub(r"\bWW\s?(II|I|2|1)\b", lambda m: "World War " + ("Two" if m.group(1) in ("II", "2") else "One"), t)
    t = re.sub(r"\b([A-Za-z][A-Za-z']*\.?)(\s+)(" + ROMAN_RE + r")\b(?!'\w|-\d)", roman_sub, t)
    t = re.sub(r"(?<=[a-z])\((I|II|III|IV|V|VI)\)", lambda m: " " + cardinal(roman(m.group(1))), t)
    # powers: x^2 -> x squared, 10^23 -> 10 to the twenty third, 6.02 x 10^23 -> 6.02 times 10 to the ...
    t = re.sub(r"(\d)\s?[x×*]\s?(?=10\s?\^)", r"\1 times ", t)
    t = re.sub(UNIT_PRE + r"\s?(km|m|cm|mm|ft|mi|yd|in)\^([23])\b",
               lambda m: f"{m.group(1)} {'square' if m.group(3) == '2' else 'cubic'} {UNITS[m.group(2)]}", t)
    def power(m):
        b, e = m.group(1), m.group(2)
        neg, e = e.startswith("-"), e.lstrip("+-")
        b = " ".join(b) if b.isalpha() and len(b) > 1 else b
        if not e.isdigit():
            return f"{b} to the {e}"
        n = int(e)
        return f"{b} squared" if n == 2 and not neg else f"{b} cubed" if n == 3 and not neg else \
            f"{b} to the {'minus ' if neg else ''}{ordinal(n)}"
    t = re.sub(r"(\b[A-Za-z]{1,3}|\b\d+(?:\.\d+)?|\))\s?\^\s?\(?([+-]?(?:\d+|[a-z]))\)?", power, t)
    if re.search(r"[=^+]|\bsolve\b|\bequation", t):  # 3x - 2 = 11 -> 3 x minus 2
        t = re.sub(r"(?<=[\w)])\s+[-–]\s+(?=[\w(])", " minus ", t)
        t = re.sub(r"\b(\d+)([a-fi-jn-rt-z])\b", r"\1 \2", t)
    t = re.sub(r"\b(\d+(?:\.\d+)?)x(?=\s+(?:faster|slower|more|less|larger|smaller|bigger|higher|lower|cheaper|better|"
               r"worse|longer|shorter|the|as|over|speed|zoom)\b)", r"\1 times", t)
    t = re.sub(r"(\d)\s+[x×*]\s+(\d+(?:\.\d+)?)(?=\s*=)", r"\1 times \2", t)
    # bare scales: 4.5M visitors -> 4.5 million visitors (5K run stays)
    t = re.sub(r"(?<![\w$€£.])(\d+\.\d+|\d+(?=\s?\w+\s+[a-z]{3,}s\b))\s?(M|bn|mn|tn)\b(?=\s+[a-z]{3,})",
               lambda m: f"{m.group(1)} {SCALE[m.group(2)]}", t)
    return t


def norm_pre(t):
    t = norm_tokens(t)
    # host ports: localhost:8080 -> localhost colon eighty eighty
    t = re.sub(r"\b(localhost|(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}):(\d{2,5})\b", lambda m: f"{m.group(1)} colon {nread(m.group(2))}", t)
    # URL schemes: a bare one is read out (rather than http:// -> H T T P colon slash slash), http(s) before an address is
    # dropped as people do when reading one aloud, other schemes are read (sftp://host -> S F T P colon slash slash host)
    say = lambda s: SCHEME_SAY.get(s.lower(), s.lower() if pronounceable(s) else spell(s))
    t = re.sub(r"(?<![\w.+-])([A-Za-z][A-Za-z0-9+.-]{1,9})://(?=[\s,.;:!?)\"']|$)", lambda m: say(m.group(1)) + " colon slash slash", t)
    t = re.sub(r"(?<![\w.+-])(?!https?://)([A-Za-z][A-Za-z0-9+.-]{1,9})://(?=[\w/@\[])((?![\w.-]*@)[A-Za-z][\w-]*(?:\.[\w-]+)+\b)?",
               lambda m: say(m.group(1)) + " colon slash slash " + (m.group(2) or "").replace(".", " dot ").replace("-", " dash "), t)
    t = re.sub(r"\bmailto:(?=[\w.+-]+@)", "mail to ", t)
    # e-mail and web addresses
    t = re.sub(r"\b([\w.+-]+)@([\w-]+(?:\.[\w-]+)+)\b",
               lambda m: m.group(1).replace(".", " dot ").replace("_", " underscore ").replace("-", " dash ")
               + " at " + m.group(2).replace(".", " dot ").replace("-", " dash "), t)
    t = re.sub(r"\bhttps?://(?![\w.-]*[:@])([A-Za-z][\w-]*(?:\.[\w-]+)+\b)?", lambda m: re.sub(r"^www ", "w w w ", (m.group(1) or "")
               .replace(".", " dot ").replace("-", " dash ")) if m.group(1) and not re.search(r"\.(?:com|org|net|io|ai|co|uk|dev|app|gov|edu|ly|me|tv|us|eu|de|fr|info|biz)$", m.group(1))
               else m.group(1) or "", t)
    t = re.sub(r"\bwww\.", "w w w dot ", t)
    t = re.sub(r"\b((?:[\w-]+\.)+)(com|org|net|io|ai|co|uk|dev|app|gov|edu|ly|me|tv|us|eu|de|fr|info|biz)\b",
               lambda m: m.group(1).replace(".", " dot ").replace("-", " dash ") + m.group(2), t)
    # file names: name_with_parts.ext
    t = re.sub(r"\b([\w-]+)\.(pdf|png|jpe?g|gif|docx?|xlsx?|csv|txt|zip|mp3|mp4|wav|json|py|js|html?|gradle|ipynb|rtf|pptx?|webm|bin|dat|bat|ps1|vue|jsx|kts|cs|dart|lua|r)\b",
               lambda m: re.sub(r"_", " underscore ", m.group(1)) + " dot " + m.group(2), t)
    def fname(m):
        base, ext = m.group(1), m.group(2)
        base = base.replace("_", " underscore ").replace("/", " slash ").replace("-", " dash ")
        base = re.sub(r"\.", " dot ", base)
        return base + " dot " + ext
    t = re.sub(r"(?<![\w@])((?:/?[\w-]+/)*[\w-]*[a-z_][\w-]*)\.(conf|log|cfg|ini|yaml|yml|toml|md|sh|ts|tsx|css|xml|sql|db|bak|tmp|exe|dmg|apk|iso|tar|gz|rar|7z|svg|webp|heic|mov|avi|mkv|flac|ogg|m4a|key|pem|crt|env|lock|rs|go|java|kt|swift|rb|php|c|cpp|h|git)\b",
               fname, t)
    t = re.sub(r"(?<![\w])/((?:[\w.-]+/)*[\w.-]+)", lambda m: " slash " + m.group(1).replace("/", " slash "), t)
    # 12-25-1999 and 14.03.2027 are dates; give them the slashed form the date rules read
    t = re.sub(r"\b(\d{1,2})-(\d{1,2})-((?:19|20)\d\d)\b", r"\1/\2/\3", t)
    t = re.sub(r"\b(\d{1,2})\.(\d{1,2})\.((?:19|20)\d\d)\b",
               lambda m: f"the {ordinal(int(m.group(1)))} of {MONTHS[int(m.group(2)) - 1]} {m.group(3)}"
               if 1 <= int(m.group(1)) <= 31 and 1 <= int(m.group(2)) <= 12 else m.group(0), t)
    # day and month ranges / abbreviations
    t = re.sub(r"\b(" + "|".join(DAYS) + r")\.?\s*[-–]\s*(" + "|".join(DAYS) + r")\.?",
               lambda m: DAYS[m.group(1)] + " to " + DAYS[m.group(2)], t)
    t = re.sub(r"(?<!\bthe )(?<!\bThe )\b(" + "|".join(DAYS) + r")\.(?=[\s,]|$)",
               lambda m: DAYS[m.group(1)] + ("." if re.match(r"\s*(?:$|[A-Z])", m.string[m.end():]) else ""), t)
    t = re.sub(r"\b(on|by|until|till|every|from|next|this|last) (" + "|".join(DAYS) + r")\b(?!')", lambda m: m.group(1) + " " + DAYS[m.group(2)], t)
    t = re.sub(r"\b(" + "|".join(MON_ABBR) + r")\.(?=\s*\d)", lambda m: MON_ABBR[m.group(1)], t)
    t = re.sub(r"\b(" + "|".join(MON_ABBR) + r")\b(?=\s+\d{1,4}\b(?!:))", lambda m: MON_ABBR[m.group(1)], t)
    t = re.sub(r"(?<=\d )(" + "|".join(MON_ABBR) + r")\b\.?", lambda m: MON_ABBR[m.group(1)], t)
    # house numbers and street types: 742 Evergreen Terrace -> seven forty two, 4 Elm Rd -> Road
    def street(m):
        num, letter, name, kind, dot = m.groups()
        tail = "." if dot and re.match(r"\s*(?:$|[A-Z\n])", m.string[m.end():]) else ""
        return f"{nread(num)}{' ' + letter.upper() if letter else ''} {name} {STREET.get(kind, kind)}{tail}"
    t = re.sub(r"\b(\d{1,4})([A-Za-z]?)\s+((?:(?:North|South|East|West|N\.?|S\.?|E\.?|W\.?)\s+)?"
               r"(?:[A-Z][\w']*|\d+(?:st|nd|rd|th))(?:\s+[A-Z][\w']*){0,2})\s+" + STREET_RE, street, t)
    t = re.sub(r"(?<!Mount )(?<!Port )(?<!Fort )(?<!Lake )(?<!Cape )\b([A-Z][a-z]+|\d+(?:st|nd|rd|th))\s+"
               r"(St|Rd|Ave|Blvd|Ln|Hwy|Pkwy|Cres|Terr?)\b(\.?)",
               lambda m: m.group(0) if m.group(1).lower() in ROMAN_STOP | {"on", "at", "visit", "see"}
               or m.group(2) == "St" and re.match(r"\s+[A-Z]", m.string[m.end():]) else f"{m.group(1)} {STREET[m.group(2)]}"
               + ("." if m.group(3) and re.match(r"\s*(?:$|\n)", m.string[m.end():]) else ""), t)
    t = re.sub(r"\b((?:I|US|SR|Route|Highway|Hwy|Interstate)[- ]\d+)\s([NSEW])\b",
               lambda m: m.group(1) + " " + {"N": "North", "S": "South", "E": "East", "W": "West"}[m.group(2)], t)
    for pat, rep in ABBR:
        t = re.sub(pat, rep, t)
    # ISO dates 2026-10-21 -> October twenty first, 2026 (NeMo reads the day ordinal and the year)
    MN = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
    t = re.sub(r"\b(\d{4})-(\d{2})-(\d{2})\b",
               lambda m: f"{MN[int(m.group(2)) - 1]} {int(m.group(3))}, {m.group(1)}" if 1 <= int(m.group(2)) <= 12 else m.group(0), t)
    # 14 March 2027 -> the fourteenth of March 2027; 3-17 November -> the third to the seventeenth of November
    t = re.sub(r"\b(\d{1,2})(?:st|nd|rd|th)?\s*[-–]\s*(\d{1,2})(?:st|nd|rd|th)?\s+" + MON_RE + r"\b",
               lambda m: f"the {ordinal(int(m.group(1)))} to the {ordinal(int(m.group(2)))} of {m.group(3)}"
               if int(m.group(1)) < int(m.group(2)) <= 31 else m.group(0), t)
    t = re.sub(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?" + MON_RE + r"\b",
               lambda m: ("" if re.search(r"\bthe\s+$", m.string[:m.start()], re.I) else "the ") + f"{ordinal(int(m.group(1)))} of {m.group(2)}"
               if 1 <= int(m.group(1)) <= 31 else m.group(0), t)
    # year ranges 1968-2020, 1990-95 (before the phone rule can read them as digits)
    t = re.sub(r"\b(1[5-9]\d\d|20\d\d)\s*[-–]\s*(1[5-9]\d\d|20\d\d|\d\d)\b(?![-/.]\d)",
               lambda m: f"{m.group(1)} to {m.group(2)}" if len(m.group(2)) == 4 or int(m.group(2)) > int(m.group(1)[2:]) else m.group(0), t)
    # month/year 3/2027 -> March 2027; set phrases 24/7, 50/50, 9/11
    t = re.sub(r"(?<![\d/])\b(\d{1,2})/((?:19|20)\d\d)\b", lambda m: f"{MN[int(m.group(1)) - 1]} {m.group(2)}" if 1 <= int(m.group(1)) <= 12 else m.group(0), t)
    t = re.sub(r"(?<![\d/])\b(24/7|50/50|9/11)\b(?!/)", lambda m: {"24/7": "24 7", "50/50": "fifty fifty", "9/11": "nine eleven"}[m.group(1)], t)
    # fractions: 1 1/2 -> 1 and a half, 3/4 cup -> three quarters cup (dates need a date word before them)
    t = re.sub(r"(?<![\d/.,])\b(\d+)\s+(\d{1,2})/(\d{1,2})\b(?!/\d)",
               lambda m: f"{m.group(1)} and {frac_words(int(m.group(2)), int(m.group(3)), True)}"
               if 0 < int(m.group(2)) < int(m.group(3)) and int(m.group(3)) in FRAC_DEN else m.group(0), t)
    # day/month (or month/day) without a year: 14/03 -> the 14th of March, 03/14 -> March 14th; ambiguous -> day/month
    def dm(m):
        a, b = int(m.group(1)), int(m.group(2))
        after, before = m.string[m.end():m.end() + 12].lower(), m.string[max(0, m.start() - 16):m.start()]
        if re.match(r"\s*(stars?|points?|marks?|votes?)\b", after) or re.search(r"(rated|scored?|rating|got)\s*$", before.lower()):
            return f"{a} out of {b}"
        if 0 < a < b and b in FRAC_DEN and "0" not in (m.group(1)[0], m.group(2)[0]) and (
                not re.search(DATE_BEFORE, before, re.I) or re.match(FRAC_AFTER, after)):
            return frac_words(a, b)
        ordn = lambda n: f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"
        the = "" if re.search(r"\bthe\s+$", before, re.I) else "the "
        if 1 <= b <= 12 and 1 <= a <= 31:
            return f"{the}{ordn(a)} of {MN[b - 1]}"
        if 1 <= a <= 12 and 1 <= b <= 31:
            return f"{MN[a - 1]} {ordn(b)}"
        return m.group(0)
    t = re.sub(r"(?<![\d/])\b(\d{1,2})/(\d{1,2})\b(?!/\d)", dm, t)
    # multi-dot versions 4.0.1(-beta)
    t = re.sub(r"\b(\d+(?:\.\d+){2,})(?:-(alpha|beta|rc\d*|preview))?\b",
               lambda m: " point ".join(m.group(1).split(".")) + (" " + m.group(2) if m.group(2) else ""), t)
    # 2:03:45 -> two hours three minutes forty five seconds (a clock time after "at" keeps its seconds at the end)
    pl = lambda n, w: f"{int(n)} {w}{'' if int(n) == 1 else 's'}"
    t = re.sub(r"\b(\d{1,2}):(\d{2}):(\d{2})\b(?!:)", lambda m: f"{m.group(1)}:{m.group(2)} and {pl(m.group(3), 'second')}"
               if re.search(r"\b(at|by|until|from)\s+$", m.string[:m.start()]) else
               ", ".join(pl(x, w) for x, w in zip(m.groups(), ("hour", "minute", "second")) if int(x) or w == "second"), t)
    # time ranges: 10:00-16:30, 8 AM-6 PM, 9am-5pm
    t = re.sub(r"\b(\d{1,2}(?::\d{2})?\s?(?:[AaPp]\.?[Mm]\.?)?)\s?[-–]\s?(\d{1,2}(?::\d{2})?\s?[AaPp]\.?[Mm]\b\.?)", r"\1 to \2", t)
    t = re.sub(r"\b(\d{1,2}:\d{2})\s?[-–]\s?(\d{1,2}:\d{2})\b", r"\1 to \2", t)
    # ratios / scores written with a colon where the right side is one digit: 3:2, 16:9 is two digits -> handled as ratio after "ratio"
    t = re.sub(r"\b(\d{1,2}):(\d)\b(?!\d)", r"\1 to \2", t)
    t = re.sub(r"(?<=[Rr]atio of |[Rr]atio is )(\d{1,2}):(\d{1,2})\b", r"\1 to \2", t)
    # letter-led codes with digit groups (CS-2026-0813, AB-4471-Z) before the phone rule can grab their digits
    t = re.sub(r"\b[A-Z]{1,4}(?:-[A-Z0-9]{1,6})*-\d[A-Za-z0-9]*(?:-[A-Za-z0-9]+)*\b", lambda m: code_read(m.group(0)), t)
    # phone numbers: optional +CC, then digit groups split by spaces/dashes/dots/parens, 7+ digits in total
    def phone(m):
        s = m.group(0)
        lead = "plus " if s.strip().startswith("+") else ""
        groups = re.findall(r"\d+", s)
        return " " + lead + ", ".join(digits(g) for g in groups) + " "
    t = re.sub(r"(?<![\w.,])\+?\(?\d{2,4}\)?(?:[\s.-]\(?\d{2,4}\)?){1,4}(?![\w.,]*\d)",
               lambda m: phone(m) if sum(c.isdigit() for c in m.group(0)) >= 7 else m.group(0), t)
    # short service numbers preceded by call/dial/text
    t = re.sub(r"\b((?:[Cc]all|[Dd]ial|[Tt]ext|[Rr]ing)\s+)(\d{3,6})\b", lambda m: m.group(1) + digits(m.group(2)), t)
    # units after numbers (singular after exactly 1; 10 mg/kg -> milligrams per kilogram)
    t = re.sub(r"\b(\d+)\s?ft\.?\s?(\d{1,2})\s?in\b\.?", lambda m: f"{m.group(1)} {'foot' if m.group(1) == '1' else 'feet'} "
               f"{m.group(2)} {'inch' if m.group(2) == '1' else 'inches'}", t)
    t = re.sub(r"\b(\d)'\s?(\d{1,2})(?:\"|'')", r"\1 foot \2", t)
    ukeys = "|".join(re.escape(k) for k in sorted((k for k in UNITS if k not in ("in", "s")), key=len, reverse=True))
    t = re.sub(UNIT_PRE + r"\s?(" + ukeys + r")/(" + ukeys + r"|s|in)(?![\w/])",
               lambda m: f"{m.group(1)} {unit(m.group(2), m.group(1))} per {sing(UNITS.get(m.group(3), 'seconds'))}"
               if m.group(3) != "in" else m.group(0), t)
    t = re.sub(UNIT_PRE + r"\s?(" + ukeys + r")(?![\w/'])", lambda m: f"{m.group(1)} {unit(m.group(2), m.group(1))}", t)
    t = re.sub(r"(\b(?:squared|cubed|to the (?:minus )?(?:\w+ )?\w+(?:th|st|nd|rd))\s)(" + ukeys + r")(?:/(" + ukeys + r"|s))?(?![\w/'])",
               lambda m: m.group(1) + UNITS[m.group(2)] + (" per " + sing(UNITS.get(m.group(3), "seconds")) if m.group(3) else ""), t)
    # identifiers read in pairs: room 402, gate 47B, flight UA 482, B12, AA229, A380 (5+ digits stay digit by digit)
    t = re.sub(r"\b(" + ID_WORDS + r")(\s+(?:No\.?\s*|#\s*)?)([A-Z]{1,3}\s?)?(\d{2,4})([A-Z]{0,2})\b(?![.,]\d|\s?[-–]\s?\d)",
               lambda m: m.group(1) + m.group(2) + (spell(m.group(3)) + " " if m.group(3) else "") + nread(m.group(4))
               + (" " + spell(m.group(5)) if m.group(5) else ""), t)
    t = re.sub(r"\b([A-Z]{2,3})\s(\d{3,4})\b(?![.,:]\d|\s\d)", lambda m: m.group(1) + " " + nread(m.group(2)), t)
    # codes after code/reference/serial words stay letter by letter and digit by digit (handled below)
    pair = lambda m: not re.search(CODE_CTX, m.string[max(0, m.start() - 40):m.start()])
    t = re.sub(r"(?<![-/.\w])([A-Z]{1,3})(\d{1,4})([A-Z]{0,2})(?![-/\w])", lambda m: " ".join(
        x for x in (spell(m.group(1)), nread(m.group(2)), spell(m.group(3))) if x)
        if pair(m) and (len(m.group(2)) <= 2 or len(m.group(1)) == 2 and not m.group(3)) else m.group(0), t)
    t = re.sub(r"(?<![-/.\w])(\d{1,4})([A-Z]{1,2})(?![-/\w])",
               lambda m: nread(m.group(1)) + " " + spell(m.group(2)) if pair(m) else m.group(0), t)
    # chemical formulas: C6H12O6 -> C six H twelve O six
    t = re.sub(r"(?<![-/.\w])(?=[A-Za-z]*\d)(?:[A-Z][a-z]?\d{0,3}){2,}(?![-/\w])", lambda m: " ".join(
        spell(e) + (" " + cardinal(int(n)) if n else "") for e, n in re.findall(r"([A-Z][a-z]?)(\d*)", m.group(0)))
        if all(e in ELEMENTS for e in re.findall(r"[A-Z][a-z]?", m.group(0))) and pair(m) else m.group(0), t)
    # hash numbers: #1 -> number one, #4471 -> number four four seven one
    t = re.sub(r"#(\d+(?:-\d+)*)\b", lambda m: "number " + ", ".join(g if len(g) <= 3 else digits(g) for g in m.group(1).split("-")), t)
    # codes: letter/digit mixes with or without dashes (AB-4471-Z, X7K9, ORD-9Q2-77L), not plain words or plain numbers
    code = lambda m: code_read(m.group(0))
    t = re.sub(r"\b(?=[A-Za-z0-9-]*\d)(?=[A-Za-z0-9-]*[A-Za-z])[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+\b", code, t)
    t = re.sub(r"\b(?=\w*\d)(?=\w*[A-Z])[A-Z0-9]{3,}\b", lambda m: spell(m.group(0)), t)
    # random mixed IDs (4hQx8Z, a9F2k): letters + digits, 4+ chars, not ordinals / times / units like 3rd, 10am, 5kg
    t = re.sub(r"\b(?=[A-Za-z0-9]*\d)(?=[A-Za-z0-9]*[A-Za-z])[A-Za-z0-9]{4,}\b",
               lambda m: m.group(0) if re.fullmatch(r"\d+(?:st|nd|rd|th|am|pm|s|kg|km|mm|cm|ml|mg|gb|mb|tb|kb|ghz|mhz|kw|kwh|mah|hrs?|mins?|h|m|k|x|g|l|w|v)", m.group(0), re.I)
               else spell(m.group(0)), t)
    # long bare digit strings (order / account numbers): 5+ digits, not money, not years with commas
    t = re.sub(r"(?<![\d$€£.,])\b(\d{5,})\b(?![.,]\d)", lambda m: digits(m.group(1)), t)
    # ranges between small numbers: 9-5, 3-5 days, 0-2
    t = re.sub(r"(?<![\d/:.-])\b(\d{1,4})\s*[-–]\s*(\d{1,4})\b(?![/:.-]\d)", r"\1 to \2", t)
    # durations: 2h 15, 1h30, 45min, 3hrs
    t = re.sub(r"\b(\d+)\s*h\s*(\d{1,2})\b", lambda m: f"{m.group(1)} hour{'' if m.group(1) == '1' else 's'} {m.group(2)}", t)
    t = re.sub(r"\b(\d+)\s*hrs?\b", lambda m: f"{m.group(1)} hour{'' if m.group(1) == '1' else 's'}", t)
    t = re.sub(r"\b(\d+)\s*mins?\b", lambda m: f"{m.group(1)} minute{'' if m.group(1) == '1' else 's'}", t)
    t = re.sub(UNIT_PRE + r"\s?(" + ukeys + r")(?![\w/'])", lambda m: f"{m.group(1)} {unit(m.group(2), m.group(1))}", t)
    # dimensions with ×
    t = re.sub(r"\s*×\s*", " by ", t)
    t = re.sub(r"(?<=\d)\s+x\s+(?=\d)", " by ", t)
    for pat, rep in SYMBOLS:
        t = re.sub(pat, rep, t)
    # pounds and pence: £7.50 -> 7 pounds 50 (scaled money is read in norm_tokens)
    t = re.sub(r"£(\d[\d,]*)\.(\d{2})\b", lambda m: f"{m.group(1)} pounds {int(m.group(2))}" if m.group(2) != "00" else f"{m.group(1)} pounds", t)
    # leftover en dashes are not in the model's alphabet
    t = re.sub(r"\s+–\s+", " — ", t).replace("–", "-")
    return re.sub(r"\s{2,}", " ", t).strip()


ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
SCALES = [(10**12, "trillion"), (10**9, "billion"), (10**6, "million"), (1000, "thousand")]
MONTHS = "January February March April May June July August September October November December".split()
ORD = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth", "nine": "ninth", "twelve": "twelfth"}


def cardinal(n):
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    if n < 1000:
        return ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " and " + cardinal(n % 100))
    for v, name in SCALES:
        if n >= v:
            rest = n % v
            tail = "" if rest == 0 else (" and " if rest < 100 else " ") + cardinal(rest)
            return cardinal(n // v) + " " + name + tail
    return str(n)


def ordinal(n):
    w = cardinal(n).split(" ")
    last = w[-1]
    w[-1] = ORD.get(last, last[:-1] + "ieth" if last.endswith("y") else last + "th")
    return " ".join(w)


def year(n):
    if 2000 <= n <= 2009:
        return cardinal(n)
    hi, lo = divmod(n, 100)
    return cardinal(hi) + (" hundred" if lo == 0 else " oh " + ONES[lo] if lo < 10 else " " + cardinal(lo))


def number(s):
    s = s.replace(",", "")
    if "." in s:
        a, b = s.split(".", 1)
        return (cardinal(int(a)) if a else "zero") + " point " + " ".join(ONES[int(c)] for c in b)
    return cardinal(int(s))


def fallback(t):
    """Light verbalizer for numbers, money, times and ordinals, used when NeMo is not installed."""
    cur = {"$": ("dollar", "dollars", "cent", "cents"), "€": ("euro", "euros", "cent", "cents"), "£": ("pound", "pounds", "penny", "pence")}

    def money(m):
        one, many, c1, cn = cur[m.group(1)]
        whole, frac = int(m.group(2).replace(",", "")), m.group(3)
        if frac and len(frac) != 2:  # $1.5 -> one point five dollars
            return f"{number(m.group(2) + '.' + frac)} {many}"
        c = int(frac) if frac else 0
        if whole == 0 and c:
            return f"{cardinal(c)} {c1 if c == 1 else cn}"
        out = f"{cardinal(whole)} {one if whole == 1 else many}"
        return out + (f" and {cardinal(c)} {c1 if c == 1 else cn}" if c else "")
    t = re.sub(r"([$€£])(\d[\d,]*)(?:\.(\d+))?\b", money, t)

    def clock(m):
        h, mi = int(m.group(1)), int(m.group(2))
        if mi == 0 and not m.group(3) and re.match(r"\s*(noon|midnight)", m.string[m.end():]):
            return cardinal(h)
        if mi == 0 and not m.group(3) and (h > 12 or h == 0):  # 14:00 -> fourteen hundred
            return ("zero" if h == 0 else cardinal(h)) + " hundred"
        out = cardinal(h) + ((" o'clock" if not m.group(3) else "") if mi == 0 else " oh " + ONES[mi] if mi < 10 else " " + cardinal(mi))
        if m.group(3):
            out += " " + m.group(3).replace(".", "").upper()
        return out
    t = re.sub(r"\b(\d{1,2}):(\d{2})\s*([aApP]\.?[mM])?(?:\.(?=\s+[a-z]))?(?=\W|$)", clock, t)
    t = re.sub(r"\b(\d{1,2})\s*([aApP])\.?[mM](?:\.(?=\s+[a-z]))?(?=\W|$)", lambda m: f"{cardinal(int(m.group(1)))} {m.group(2).upper()}M", t)

    def date(m):  # 01/10/2026 or 1/10/26, read month first
        mo, d, y = int(m.group(1)), int(m.group(2)), m.group(3)
        y = int(y) + (2000 if len(y) == 2 else 0)
        if 12 < mo <= 31 and 1 <= d <= 12:  # 14/03/2027 is day first
            return f"the {ordinal(mo)} of {MONTHS[d - 1]} {year(y)}"
        if not (1 <= mo <= 12 and 1 <= d <= 31):
            return m.group(0)
        return f"{MONTHS[mo - 1]} {ordinal(d)} {year(y)}"
    t = re.sub(r"\b(\d{1,2})/(\d{1,2})/(\d{4}|\d{2})\b", date, t)
    t = re.sub(r"\b(" + "|".join(MONTHS) + r")\.? (\d{1,2})\b(?!:)", lambda m: f"{m.group(1)} {ordinal(int(m.group(2)))}", t)
    t = re.sub(r"(\d)\s?°\s?C\b", r"\1 degrees Celsius", t)
    t = re.sub(r"(\d)\s?°\s?F\b", r"\1 degrees Fahrenheit", t)
    t = re.sub(r"(\d)\s?°", r"\1 degrees", t)
    t = re.sub(r"(?<=\w)/(?=\w)", " slash ", t)
    t = re.sub(r"\b(\d+)(st|nd|rd|th)\b", lambda m: ordinal(int(m.group(1))), t)
    t = re.sub(r"\b(1[1-9]\d\d|20\d\d)\b(?![.,]\d)", lambda m: year(int(m.group(1))), t)
    t = re.sub(r"(?<![\w-])-(?=\d)", "minus ", t)
    t = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", t)  # v2 -> v two
    t = re.sub(r"(\d+(?:\.\d+)?)\s?%", lambda m: number(m.group(1)) + " percent", t)
    t = re.sub(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|\b\d+(?:\.\d+)?\b", lambda m: number(m.group(0)), t)
    t = t.replace("&", " and ").replace("+", " plus ").replace("=", " equals ").replace("@", " at ").replace("#", " number ")
    return t


class Normalizer:
    """Turns raw text into the spoken-form text the model reads. NeMo is used only if it is installed. Hard names and
    foreign words are then respelled from the lexicon in respell.tsv (respell=False turns that off)."""

    def __init__(self, use_nemo=True, respell=True):
        self.n = None
        self.respell = None
        if respell:
            try:
                from .respell import Respeller
            except ImportError:  # loaded as a plain file (tools/score_norm.py)
                import importlib.util, os
                spec = importlib.util.spec_from_file_location("respell", os.path.join(os.path.dirname(os.path.abspath(__file__)), "respell.py"))
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                Respeller = mod.Respeller
            self.respell = Respeller()
        if use_nemo:
            try:
                import logging
                logging.disable(logging.WARNING)
                from nemo_text_processing.text_normalization.normalize import Normalizer as N
                self.n = N(input_case="cased", lang="en")
            except ImportError:
                pass

    def __call__(self, text):
        t = norm_pre(text)
        if re.search(r"[0-9#%$€£@&/_+=<>±≥≤×°]", t):  # only where something is left to verbalize (NeMo is slow)
            if self.n is not None:
                try:
                    t = self.n.normalize(t, punct_post_process=True)
                except Exception:
                    t = fallback(t)
            else:
                t = fallback(t)
        t = re.sub(r"\s+([.,!?;:])", r"\1", t)
        t = re.sub(r"\s{2,}", " ", t).strip()
        if self.respell is not None:  # hard names and foreign words, last so no rule above sees a respelling
            t = self.respell(t)
        return t[:1].upper() + t[1:] if text[:1].isupper() else t
