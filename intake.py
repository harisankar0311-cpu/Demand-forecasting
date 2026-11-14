"""
intake.py - read messy files and clean them without throwing data away.

Lane 1: a clean table goes straight to forecasting (no changes).
Lane 2: a messy table gets a "recipe" of cleaning steps. Rows that cannot be used (for example no
readable date) are SET ASIDE with a reason instead of being deleted, and every change is logged.

Parts of this module:
  load_smart       read CSV / Excel with delimiter, encoding, sheet and header-row detection
  parse_*          turn many date, money, promo and text formats into one standard form
  OPS / apply_recipe   the fixed list of cleaning steps and the code that runs them
  analyze          inspect a table, propose steps and ask questions where it is unsure
"""
import csv
import datetime as _dt
import difflib
import io
import re
import uuid
from collections import Counter

import numpy as np
import pandas as pd

NA_TOKENS = {"", "na", "n/a", "nan", "null", "none", "nil", "-", "--", "\u2014", "#n/a", "missing", "?"}


def _blank(v):
    return (v is None or v is pd.NaT or (isinstance(v, float) and np.isnan(v))
            or (isinstance(v, str) and v.strip() == ""))


def _map_cells(df, fn):
    return df.map(fn) if hasattr(df, "map") else df.applymap(fn)


# =============================================================================
# 1. READING FILES
# =============================================================================
def _read_bytes(file):
    if isinstance(file, (bytes, bytearray)):
        return bytes(file), ""
    name = getattr(file, "name", str(file))
    if hasattr(file, "read"):
        try:
            file.seek(0)
        except Exception:
            pass
        data = file.read()
    else:
        with open(file, "rb") as fh:
            data = fh.read()
    return (data.encode("utf-8") if isinstance(data, str) else data), name


def _decode(data):
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1", errors="replace"), "latin-1"


def _sniff_delimiter(text):
    lines = [ln for ln in text.splitlines()[:40] if ln.strip()]
    best, best_score = ",", 0
    for d in (",", ";", "\t", "|"):
        counts = []
        for ln in lines:
            try:
                counts.append(len(next(csv.reader([ln], delimiter=d))))
            except Exception:
                counts.append(1)
        if not counts:
            continue
        fields, freq = Counter(counts).most_common(1)[0]
        score = freq * fields if fields > 1 else 0
        if score > best_score:
            best, best_score = d, score
    return best


_NUMERICISH = re.compile(r"^[\d.,\s\u20b9$\u20ac\u00a3%()+\-]+$")
_DATEISH = re.compile(r"^\d{1,4}[/\-.]\d{1,2}[/\-.]\d{1,4}$")


def _headerish(v):
    if isinstance(v, (pd.Timestamp, _dt.datetime, _dt.date)):
        return True
    if not isinstance(v, str):
        return False
    return bool(_DATEISH.match(v.strip())) or not _NUMERICISH.match(v.strip())


def _find_header(raw, max_scan=15):
    ncols = raw.shape[1]
    for i in range(min(max_scan, len(raw))):
        vals = [v for v in raw.iloc[i] if not _blank(v)]
        if len(vals) >= max(2, 0.6 * ncols) and all(_headerish(v) for v in vals):
            return i
    return 0


def _clean_names(header):
    names, seen = [], Counter()
    for i, v in enumerate(header):
        if _blank(v):
            n = f"column_{i + 1}"
        elif isinstance(v, (pd.Timestamp, _dt.datetime, _dt.date)):
            n = pd.Timestamp(v).strftime("%Y-%m-%d")
        else:
            n = re.sub(r"\s+", " ", str(v)).strip()
        seen[n] += 1
        names.append(n if seen[n] == 1 else f"{n}_{seen[n]}")
    return names


_ISO_STRICT = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?(?:Z|[+-]\d{2}:?\d{2})?$")


def infer_types(df):
    """Give clean columns proper types (numbers, ISO dates); leave anything messy as text."""
    out = {}
    for c in df.columns:
        s = df[c]
        nn = s.dropna()
        if nn.empty:
            out[c] = s
            continue
        if all(isinstance(v, (pd.Timestamp, _dt.datetime, _dt.date)) for v in nn):
            out[c] = pd.to_datetime(s)
            continue
        if all(isinstance(v, str) and _ISO_STRICT.match(v) for v in nn):
            out[c] = pd.to_datetime(s, errors="coerce", utc=True).dt.tz_convert(None)
            continue
        if not any(isinstance(v, str) and re.match(r"^[+-]?0\d", v) for v in nn):  # leading zero = an ID
            num = pd.to_numeric(s, errors="coerce")
            if num.notna().sum() == s.notna().sum():
                out[c] = num
                continue
        out[c] = s
    return pd.DataFrame(out, index=df.index)


def load_smart(file, sheet=None):
    """Read a CSV or Excel file. Returns (DataFrame, notes). Index = row number in the file."""
    data, name = _read_bytes(file)
    notes = {"name": name, "kind": "csv", "encoding": None, "delimiter": None, "sheet": None,
             "sheets": [], "header_row": 1, "messages": []}
    if name.lower().endswith((".xlsx", ".xlsm", ".xls")) or data[:2] == b"PK":
        try:
            xl = pd.ExcelFile(io.BytesIO(data))
        except Exception as e:
            raise ValueError("Could not open this Excel file. If it is an old .xls file, save it as .xlsx "
                             f"or CSV and try again. ({type(e).__name__})")
        sheets = xl.sheet_names
        if sheet not in sheets:
            sizes = {s: int(xl.parse(s, header=None, dtype=object).notna().sum().sum()) for s in sheets}
            sheet = max(sizes, key=sizes.get)
        raw = xl.parse(sheet, header=None, dtype=object)
        notes.update(kind="excel", sheet=sheet, sheets=sheets)
    else:
        text, enc = _decode(data)
        delim = _sniff_delimiter(text)
        rows = list(csv.reader(io.StringIO(text), delimiter=delim))
        width = max((len(r) for r in rows), default=0)
        raw = pd.DataFrame([r + [None] * (width - len(r)) for r in rows], dtype=object)
        notes.update(encoding=enc, delimiter=delim)

    raw = _map_cells(raw, lambda v: np.nan if _blank(v) else (v.strip() if isinstance(v, str) else v))
    raw = raw.dropna(how="all").dropna(axis=1, how="all")
    if raw.empty:
        raise ValueError("The file seems to be empty.")
    raw.columns = range(raw.shape[1])
    hdr = _find_header(raw)
    names = _clean_names(raw.iloc[hdr])
    body = raw.iloc[hdr + 1:].copy()
    body.columns = names
    body.index = body.index + 1  # row numbers as people see them in the file
    body = body.dropna(axis=1, how="all")
    if hdr > 0:
        notes["messages"].append(f"Skipped {hdr} title row(s) above the column names.")
    notes["header_row"] = int(raw.index[hdr]) + 1
    return infer_types(body), notes


# =============================================================================
# 2. PARSERS
# =============================================================================
_ISO_D = re.compile(r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?:[ T].*)?$")
_NUM_D = re.compile(r"^(\d{1,2})[-/.](\d{1,2})[-/.](\d{2}|\d{4})(?:[ T].*)?$")
_WEEK = re.compile(r"^(?:(\d{4})[-\s]?W(\d{1,2})|W(\d{1,2})[-\s]?(\d{4}))$", re.I)
_QTR = re.compile(r"^(?:Q([1-4])[-\s]?(\d{4})|(\d{4})[-\s]?Q([1-4]))$", re.I)
_YMD8 = re.compile(r"^(\d{4})(\d{2})(\d{2})$")


def _year(y):
    y = int(y)
    return y if y >= 100 else (2000 + y if y < 70 else 1900 + y)


def _mk(y, m, d):
    try:
        return pd.Timestamp(int(y), int(m), int(d))
    except Exception:
        return pd.NaT


def _date_one(v, day_first):
    if _blank(v):
        return pd.NaT
    if isinstance(v, (pd.Timestamp, _dt.datetime, _dt.date)):
        ts = pd.Timestamp(v)
        return ts.tz_convert(None) if ts.tzinfo else ts
    if isinstance(v, (int, float, np.integer, np.floating)):
        x = float(v)
        return pd.Timestamp("1899-12-30") + pd.Timedelta(days=x) if 20000 <= x <= 60000 else pd.NaT
    t = str(v).strip()
    if t.lower() in NA_TOKENS:
        return pd.NaT
    if re.fullmatch(r"\d{5}(\.\d+)?", t) and 20000 <= float(t) <= 60000:
        return pd.Timestamp("1899-12-30") + pd.Timedelta(days=float(t))
    m = _ISO_D.match(t)
    if m:
        return _mk(*m.groups())
    m = _NUM_D.match(t)
    if m:
        a, b, y = int(m.group(1)), int(m.group(2)), _year(m.group(3))
        if a > 12:
            return _mk(y, b, a)
        if b > 12:
            return _mk(y, a, b)
        return _mk(y, b, a) if day_first else _mk(y, a, b)
    m = _YMD8.match(t)
    if m:
        return _mk(*m.groups())
    m = _WEEK.match(t)
    if m:
        y, w = (m.group(1), m.group(2)) if m.group(1) else (m.group(4), m.group(3))
        try:
            return pd.Timestamp(_dt.date.fromisocalendar(int(y), int(w), 1))
        except Exception:
            return pd.NaT
    m = _QTR.match(t)
    if m:
        q, y = (m.group(1), m.group(2)) if m.group(1) else (m.group(4), m.group(3))
        return _mk(y, (int(q) - 1) * 3 + 1, 1)
    if not re.search(r"[A-Za-z]", t):
        return pd.NaT
    try:
        ts = pd.to_datetime(t, errors="coerce", format="mixed", dayfirst=bool(day_first))
        return ts.tz_convert(None) if (ts is not pd.NaT and getattr(ts, "tzinfo", None)) else ts
    except Exception:
        return pd.NaT


def date_order(series):
    """Look at slash/dash dates in a column: 'DMY', 'MDY', 'mixed', 'ambiguous' or None."""
    d_ev = m_ev = seen = False
    for v in pd.unique(series.dropna()):
        mt = _NUM_D.match(str(v).strip()) if not isinstance(v, (_dt.date, pd.Timestamp, int, float)) else None
        if mt:
            seen = True
            a, b = int(mt.group(1)), int(mt.group(2))
            d_ev |= a > 12
            m_ev |= b > 12
    if not seen:
        return None
    return "mixed" if d_ev and m_ev else "DMY" if d_ev else "MDY" if m_ev else "ambiguous"


def parse_dates(series, dayfirst=None):
    """Parse many date formats. Returns (datetime Series, info)."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series, {"order": None, "ambiguous": False, "unparsed": 0, "unparsed_examples": [], "parsed": int(series.notna().sum())}
    order = date_order(series)
    eff = dayfirst if dayfirst is not None else (order == "DMY")
    uniq = pd.unique(series.dropna())
    mp = {v: _date_one(v, eff) for v in uniq}
    out = pd.to_datetime(series.map(mp), errors="coerce")
    bad = series.notna() & out.isna() & ~series.map(lambda v: _blank(v) or str(v).strip().lower() in NA_TOKENS)
    return out, {"order": order, "ambiguous": order == "ambiguous" and dayfirst is None, "unparsed": int(bad.sum()),
                 "unparsed_examples": [str(x) for x in series[bad].unique()[:5]], "parsed": int(out.notna().sum())}


_CUR = re.compile(r"(?i)(\u20b9|\brs\.?|\binr\b|\busd\b|us\$|\$|\u20ac|\beur\b|\u00a3|\bgbp\b)")
_CUR_NAME = {"\u20b9": "INR", "rs": "INR", "rs.": "INR", "inr": "INR", "$": "USD", "us$": "USD", "usd": "USD",
             "\u20ac": "EUR", "eur": "EUR", "\u00a3": "GBP", "gbp": "GBP"}
_MULT = {"k": 1e3, "thousand": 1e3, "m": 1e6, "mn": 1e6, "mm": 1e6, "million": 1e6, "b": 1e9, "bn": 1e9,
         "billion": 1e9, "lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5, "crore": 1e7, "crores": 1e7, "cr": 1e7}
_SUFFIX = re.compile(r"(?i)^(.*?\d)\s*(k|thousand|m|mn|mm|million|b|bn|billion|lakhs?|lacs?|crores?|cr)$")
_US_GROUP = re.compile(r"\d{1,3}(,\d{3})+(\.\d+)?")
_IN_GROUP = re.compile(r"\d{1,2}(,\d{2})*,\d{3}(\.\d+)?")


def _number_one(v, decimal="auto"):
    """Returns (value, currencies, multiplied, was_percent)."""
    if _blank(v):
        return np.nan, [], False, False
    if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
        return float(v), [], False, False
    t = str(v).strip()
    if t.lower() in NA_TOKENS:
        return np.nan, [], False, False
    cur = [_CUR_NAME.get(c.lower().rstrip("."), _CUR_NAME.get(c.lower(), "")) for c in _CUR.findall(t)]
    t = _CUR.sub("", t).strip().replace("\u00a0", "").replace("\u2009", "")
    neg = False
    if t.startswith("(") and t.endswith(")"):
        neg, t = True, t[1:-1].strip()
    if t.endswith("-") and not t.startswith("-"):
        neg, t = True, t[:-1].strip()
    if t[:1] in "-\u2212":
        neg, t = (not neg), t[1:].strip()
    t = t.lstrip("+").strip()
    pct = t.endswith("%")
    if pct:
        t = t[:-1].strip()
    mult = 1.0
    m = _SUFFIX.match(t)
    if m:
        t, mult = m.group(1).strip(), _MULT[m.group(2).lower()]
    t = t.replace(" ", "").replace("'", "")
    has_c, has_d = "," in t, "." in t
    comma_locale = any(c in ("INR", "USD", "GBP") for c in cur)  # these currencies use comma for thousands
    if has_c and has_d:
        t = t.replace(".", "").replace(",", ".") if t.rfind(",") > t.rfind(".") else t.replace(",", "")
    elif has_c:
        if t.count(",") > 1:                       # 1,20,000 or 1,234,567: always thousands
            t = t.replace(",", "")
        elif re.fullmatch(r"\d{1,3},\d{3}", t):    # 1,500: thousands or decimal comma? use the column style
            t = t.replace(",", ".") if (decimal == "comma" and not comma_locale) else t.replace(",", "")
        elif re.fullmatch(r"\d+,\d{1,2}", t) and decimal != "dot":
            t = t.replace(",", ".")                 # 12,5 -> decimal comma
        else:
            t = t.replace(",", "")
    elif has_d and decimal == "comma" and not comma_locale and re.fullmatch(r"\d{1,3}(\.\d{3})+", t):
        t = t.replace(".", "")                      # 2.000 in a comma-decimal column
    try:
        val = float(t)
    except ValueError:
        return np.nan, cur, False, False
    val *= mult
    if pct:
        val /= 100.0
    return (-val if neg else val), cur, mult != 1.0, pct


def parse_number(series, decimal="auto"):
    """Parse money/number text (symbols, lakh/crore/K/M, 1,20,000, 1.200,50, (500), 20%). Returns (Series, info)."""
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        return series.astype(float), {"currencies": [], "unparsed": 0, "unparsed_examples": [], "percent": 0, "multiplied": 0, "decimal": "dot"}
    uniq = [v for v in pd.unique(series.dropna())]
    if decimal == "auto":
        eu = any(re.search(r"\d\.\d{3}(\.\d{3})*,\d", str(v)) or re.fullmatch(r"\s*\D{0,3}\d+,\d{1,2}\s*", str(v)) for v in uniq)
        decimal = "comma" if eu else "dot"
    mp, curs, mult_n, pct_n = {}, Counter(), 0, 0
    for v in uniq:
        val, cur, mu, pc = _number_one(v, decimal)
        mp[v] = val
        for c in set(cur):
            curs[c] += int((series == v).sum())
        mult_n += int(mu)
        pct_n += int(pc)
    out = pd.to_numeric(series.map(mp), errors="coerce")
    is_na = series.map(lambda v: _blank(v) or str(v).strip().lower() in NA_TOKENS)
    bad = series.notna() & out.isna() & ~is_na
    return out, {"currencies": [c for c, _ in curs.most_common() if c], "unparsed": int(bad.sum()),
                 "unparsed_examples": [str(x) for x in series[bad].unique()[:5]], "percent": pct_n,
                 "multiplied": mult_n, "decimal": decimal}


_PROMO_NO = {"", "n", "no", "false", "0", "off", "none", "nil", "-", "no promo", "na", "n/a", "nan", "null", "0%"}
_PROMO_YES = {"y", "yes", "true", "1", "on", "promo", "promotion", "active", "offer", "yes promo"}
_PROMO_WORDS = re.compile(r"(?i)\b(bogo|b1g1|buy\s*\d+\s*get\s*\d+|sale|festive|diwali|holi|christmas|black\s*friday|offer|deal|"
                          r"discount|coupon|clearance|flash|combo|cashback|promo\w*|season\w*)\b")


def parse_promo(series, custom=None):
    """Turn promo text/flags into (flag 0/1, discount fraction). Returns (flag, discount, info)."""
    custom = {str(k).strip().lower(): v for k, v in (custom or {}).items()}
    num = pd.to_numeric(series, errors="coerce") if not pd.api.types.is_numeric_dtype(series) else series
    numeric_col = series.notna().any() and num.notna().sum() == series.notna().sum()
    if numeric_col:
        vals = set(num.dropna().unique())
        if vals <= {0, 1, 0.0, 1.0}:
            flag = num.fillna(0).astype(int)
            return flag, pd.Series(0.0, index=series.index), {"unknown": [], "kind": "flag"}
        mx = num.max()
        frac = num.fillna(0) / (100.0 if mx > 1 else 1.0)
        return (frac > 0).astype(int), frac.clip(lower=0), {"unknown": [], "kind": "discount"}
    flags, discs, unknown = {}, {}, []
    for v in pd.unique(series.dropna()):
        t = re.sub(r"\s+", " ", str(v).strip().lower())
        if t in custom:
            c = custom[t]
            flags[v], discs[v] = 1, (float(c) if c not in (True, 1) or isinstance(c, float) else 0.0)
            if isinstance(c, (int, float)) and c in (0, False):
                flags[v], discs[v] = 0, 0.0
            continue
        pm = re.search(r"(\d+(?:\.\d+)?)\s*%", t)
        if t in _PROMO_NO:
            flags[v], discs[v] = 0, 0.0
        elif pm and float(pm.group(1)) > 0:
            flags[v], discs[v] = 1, float(pm.group(1)) / 100.0
        elif t in _PROMO_YES or _PROMO_WORDS.search(t):
            flags[v], discs[v] = 1, 0.0
        else:
            flags[v], discs[v] = 1, 0.0
            unknown.append(str(v))
    flag = series.map(flags).fillna(0).astype(int)
    disc = series.map(discs).fillna(0.0).astype(float)
    return flag, disc, {"unknown": unknown, "kind": "text"}


def _norm_key(s):
    return re.sub(r"\s+", " ", str(s).strip()).lower()


def standardise_text(series):
    """Merge spellings that differ only by case or spacing. Returns (Series, {variant: canonical})."""
    s = series.astype(object)
    cleaned = s.map(lambda v: re.sub(r"\s+", " ", str(v).strip()) if not _blank(v) else v)
    counts = cleaned.dropna().value_counts()
    canon = {}
    for val, _ in counts.items():  # most frequent spelling first
        canon.setdefault(_norm_key(val), val)
    out = cleaned.map(lambda v: canon[_norm_key(v)] if not _blank(v) else v)
    changed = {str(a): str(b) for a, b in zip(series.dropna(), out.dropna()) if str(a) != str(b)}
    return out, changed


# =============================================================================
# 3. CLEANING STEPS ("recipe")  - a fixed list of safe operations, no free-form code
# =============================================================================
OPS_SPEC = {
    "parse_dates": {"column": "text", "dayfirst": "true, false or null (auto)"},
    "parse_number": {"column": "text", "decimal": "auto, dot or comma"},
    "scale": {"column": "text", "factor": "number to multiply by"},
    "map_promo": {"column": "text", "custom": "optional object: value -> discount fraction (0 = just a promo flag)"},
    "standardise_text": {"column": "text"},
    "replace_values": {"column": "text", "mapping": "object: old value -> new value"},
    "set_aside": {"column": "text", "operator": "equals, not_equals, contains, matches, greater_than, less_than, is_empty, not_empty",
                  "value": "text or number", "reason": "text"},
    "set_aside_missing": {"column": "text", "reason": "text"},
    "fill_missing": {"column": "text", "method": "zero, median, mean, ffill or interpolate"},
    "zero_as_missing": {"column": "text"},
    "flag_outliers": {"column": "text", "cap": "true to also cap them"},
    "rename": {"column": "text", "new_name": "text"},
    "ignore_column": {"column": "text"},
    "melt_wide": {"id_cols": "list of column names that stay as they are"},
    "set_aside_repeated_headers": {},
    "set_aside_totals": {},
}


def new_step(op, **params):
    return {"id": uuid.uuid4().hex[:6], "op": op, "enabled": True, **params}


def _e(title, changed=0, examples=None, notes=None):
    return {"title": title, "changed": int(changed), "examples": examples or [], "notes": notes or []}


def _pairs(before, after, mask=None, n=3):
    b, a = (before[mask], after[mask]) if mask is not None else (before, after)
    out = []
    for x, y in list(zip(b.astype(str), a.astype(str)))[:n]:
        out.append((x, y[:10] if re.match(r"^\d{4}-\d{2}-\d{2} 00:00:00$", y) else y))
    return out


def _aside(df, mask, reason):
    moved = df[mask].copy()
    moved["reason"] = reason
    return df[~mask], moved


def _op_parse_dates(df, st):
    c = st["column"]
    before = df[c]
    out, info = parse_dates(before, st.get("dayfirst"))
    df = df.copy()
    df[c] = out
    notes = []
    if info["order"] == "DMY" or st.get("dayfirst") is True:
        notes.append("Read as day first (D/M/Y).")
    elif info["order"] == "MDY" or st.get("dayfirst") is False:
        notes.append("Read as month first (M/D/Y).")
    if info["ambiguous"]:
        notes.append("Ambiguous dates (like 01/02/2024). Assumed month first until you confirm.")
    if info["unparsed"]:
        notes.append(f"{info['unparsed']} value(s) could not be read as dates, e.g. {', '.join(info['unparsed_examples'][:3])}.")
    ok = out.notna() & before.notna() & ~pd.api.types.is_datetime64_any_dtype(before)
    return df, None, _e(f"Read dates in '{c}'", ok.sum(), _pairs(before, out, ok), notes)


def _op_parse_number(df, st):
    c = st["column"]
    before = df[c]
    out, info = parse_number(before, st.get("decimal", "auto"))
    df = df.copy()
    df[c] = out
    notes = []
    if info["currencies"]:
        notes.append("Currency symbols found: " + ", ".join(info["currencies"]) + ". Values are kept as written, with no conversion.")
    if info["multiplied"]:
        notes.append(f"{info['multiplied']} value(s) used K/M/lakh/crore units and were expanded.")
    if info["percent"]:
        notes.append(f"{info['percent']} percentage value(s) were turned into fractions (20% becomes 0.2).")
    if info["unparsed"]:
        notes.append(f"{info['unparsed']} value(s) could not be read (e.g. {', '.join(info['unparsed_examples'][:3])}) and were left empty.")
    changed = (before.astype(str) != out.astype(str)).sum() if not pd.api.types.is_numeric_dtype(before) else 0
    return df, None, _e(f"Read numbers in '{c}'", changed, _pairs(before, out, before.notna()), notes)


def _op_scale(df, st):
    c, f = st["column"], float(st["factor"])
    df = df.copy()
    notes = []
    if not pd.api.types.is_numeric_dtype(df[c]):
        df[c], _ = parse_number(df[c])
        notes.append("Read the text as numbers first.")
    before = df[c].copy()
    df[c] = df[c] * f
    return df, None, _e(f"Multiply '{c}' by {f:g}", before.notna().sum(), _pairs(before, df[c], before.notna()), notes)


def _op_map_promo(df, st):
    c = st["column"]
    flag, disc, info = parse_promo(df[c], st.get("custom"))
    df = df.copy()
    df[f"{c}_promo"], df[f"{c}_discount"] = flag, disc
    notes = [f"Added '{c}_promo' (0/1) and '{c}_discount' (fraction). The original column is kept."]
    if info["unknown"]:
        notes.append("Not sure about: " + ", ".join(info["unknown"][:5]) + ". Counted as a promotion.")
    return df, None, _e(f"Turn '{c}' into a promo flag and discount", int(flag.sum()),
                        _pairs(df[c], flag, df[c].notna()), notes)


def _op_standardise_text(df, st):
    c = st["column"]
    before = df[c]
    out, changed = standardise_text(before)
    df = df.copy()
    df[c] = out
    n = int((before.astype(str) != out.astype(str)).sum())
    return df, None, _e(f"Merge spellings in '{c}'", n, list(changed.items())[:3],
                        [f"{len(changed)} spelling variant(s) merged into the most common one."] if changed else [])


def _op_replace_values(df, st):
    c, mp = st["column"], st["mapping"]
    key = {_norm_key(k): v for k, v in mp.items()}
    before = df[c]
    df = df.copy()
    df[c] = before.map(lambda v: key.get(_norm_key(v), v) if not _blank(v) else v)
    ch = (before.astype(str) != df[c].astype(str)).sum()
    return df, None, _e(f"Replace values in '{c}'", ch, [(str(k), str(v)) for k, v in list(mp.items())[:3]])


def _condition_mask(s, op, value):
    txt = s.astype(str).str.strip()
    if op == "equals":
        return txt.str.lower() == str(value).strip().lower()
    if op == "not_equals":
        return (txt.str.lower() != str(value).strip().lower()) & s.notna()
    if op == "contains":
        return txt.str.contains(str(value), case=False, regex=False) & s.notna()
    if op == "matches":
        return txt.str.contains(str(value), case=False, regex=True) & s.notna()
    if op in ("greater_than", "less_than"):
        num = parse_number(s)[0]
        return (num > float(value)) if op == "greater_than" else (num < float(value))
    if op == "is_empty":
        return s.isna() | (txt == "")
    if op == "not_empty":
        return s.notna() & (txt != "")
    raise ValueError(f"unknown operator {op}")


def _op_set_aside(df, st):
    c = st["column"]
    reason = st.get("reason") or f"{c} {st['operator'].replace('_', ' ')} {st.get('value', '')}".strip()
    mask = _condition_mask(df[c], st["operator"], st.get("value"))
    kept, moved = _aside(df, mask, reason)
    return kept, moved, _e(f"Set aside rows where '{c}' {st['operator'].replace('_', ' ')} {st.get('value', '')}".strip(),
                           mask.sum(), [], [f"{int(mask.sum())} row(s) moved to 'Needs attention'. Nothing is deleted."] if mask.sum() else [])


def _op_set_aside_missing(df, st):
    c = st["column"]
    s = df[c]
    mask = s.isna() | (s.astype(str).str.strip() == "")
    kept, moved = _aside(df, mask, st.get("reason") or f"No usable value in {c}")
    return kept, moved, _e(f"Set aside rows with no usable '{c}'", mask.sum(), [],
                           [f"{int(mask.sum())} row(s) moved to 'Needs attention'."] if mask.sum() else [])


def _op_fill_missing(df, st):
    c, m = st["column"], st.get("method", "zero")
    df = df.copy()
    if not pd.api.types.is_numeric_dtype(df[c]):
        df[c], _ = parse_number(df[c])
    n = int(df[c].isna().sum())
    df[c] = (df[c].fillna(0) if m == "zero" else df[c].fillna(df[c].median()) if m == "median"
             else df[c].fillna(df[c].mean()) if m == "mean" else df[c].ffill() if m == "ffill"
             else df[c].interpolate())
    return df, None, _e(f"Fill missing values in '{c}' ({m})", n)


def _op_zero_as_missing(df, st):
    c = st["column"]
    df = df.copy()
    if not pd.api.types.is_numeric_dtype(df[c]):
        df[c], _ = parse_number(df[c])
    n = int((df[c] == 0).sum())
    df.loc[df[c] == 0, c] = np.nan
    return df, None, _e(f"Treat zeros in '{c}' as missing", n)


def _op_flag_outliers(df, st):
    c = st["column"]
    df = df.copy()
    if not pd.api.types.is_numeric_dtype(df[c]):
        df[c], _ = parse_number(df[c])
    x = df[c].astype(float)
    med = x.median()
    mad = (x - med).abs().median() * 1.4826
    if mad and mad > 0:
        lo, hi = med - 6 * mad, med + 6 * mad
    else:
        q1, q3 = x.quantile(.25), x.quantile(.75)
        lo, hi = q1 - 3 * (q3 - q1), q3 + 3 * (q3 - q1)
    mask = (x < lo) | (x > hi)
    df[f"{c}_outlier"] = mask
    if st.get("cap"):
        df[c] = x.clip(lo, hi)
    return df, None, _e(f"{'Cap' if st.get('cap') else 'Flag'} outliers in '{c}'", mask.sum(),
                        [(f"{v:,.0f}", f"{min(max(v, lo), hi):,.0f}") for v in x[mask].head(3)] if st.get("cap") else [],
                        [f"Normal range about {lo:,.0f} to {hi:,.0f}. " + ("Capped to that range." if st.get("cap") else
                         f"Flagged in '{c}_outlier'; values are unchanged.")])


def _op_rename(df, st):
    return df.rename(columns={st["column"]: st["new_name"]}), None, _e(f"Rename '{st['column']}' to '{st['new_name']}'", 0)


def _op_ignore_column(df, st):
    return df.drop(columns=[st["column"]]), None, _e(f"Leave out column '{st['column']}'", 0,
                                                    notes=["The column is only left out of the model. Your file is unchanged."])


def _op_melt_wide(df, st):
    ids = [c for c in st["id_cols"] if c in df.columns]
    var = "date" if "date" not in df.columns else "period"
    long = df.melt(id_vars=ids, var_name=var, value_name="value")
    return long.reset_index(drop=True), None, _e("Turn the wide table into one row per date", len(long),
                                                notes=[f"{len(df.columns) - len(ids)} date columns became rows. New columns: '{var}' and 'value'."])


def _op_headers(df, st):
    names = [str(c).strip().lower() for c in df.columns]
    hits = np.zeros(len(df), dtype=int)
    for c, nm in zip(df.columns, names):
        hits += (df[c].astype(str).str.strip().str.lower() == nm).to_numpy().astype(int)
    mask = pd.Series(hits >= max(2, int(0.6 * len(df.columns))), index=df.index)
    kept, moved = _aside(df, mask, "Repeated header row")
    return kept, moved, _e("Set aside repeated header rows", mask.sum(), [], [f"{int(mask.sum())} row(s) moved to 'Needs attention'."] if mask.sum() else [])


_TOTAL = re.compile(r"(?i)^\s*(grand\s+|sub\s*)?total\b")


def _op_totals(df, st):
    first = df.iloc[:, :2].astype(str)
    mask = first.apply(lambda col: col.str.match(_TOTAL)).any(axis=1)
    kept, moved = _aside(df, mask, "Total or subtotal row")
    return kept, moved, _e("Set aside total rows", mask.sum(), [], [f"{int(mask.sum())} row(s) moved to 'Needs attention'."] if mask.sum() else [])


OPS = {"parse_dates": _op_parse_dates, "parse_number": _op_parse_number, "scale": _op_scale, "map_promo": _op_map_promo,
       "standardise_text": _op_standardise_text, "replace_values": _op_replace_values, "set_aside": _op_set_aside,
       "set_aside_missing": _op_set_aside_missing, "fill_missing": _op_fill_missing, "zero_as_missing": _op_zero_as_missing,
       "flag_outliers": _op_flag_outliers, "rename": _op_rename, "ignore_column": _op_ignore_column,
       "melt_wide": _op_melt_wide, "set_aside_repeated_headers": _op_headers, "set_aside_totals": _op_totals}


def step_title(st):
    op, c = st["op"], st.get("column", "")
    return {"parse_dates": f"Read dates in '{c}'" + (" (day first)" if st.get("dayfirst") is True else " (month first)" if st.get("dayfirst") is False else ""),
            "parse_number": f"Read numbers in '{c}'", "scale": f"Multiply '{c}' by {st.get('factor')}",
            "map_promo": f"Turn '{c}' into a promo flag and discount", "standardise_text": f"Merge spellings in '{c}'",
            "replace_values": f"Replace values in '{c}'",
            "set_aside": f"Set aside rows where '{c}' {str(st.get('operator', '')).replace('_', ' ')} {st.get('value', '')}",
            "set_aside_missing": f"Set aside rows with no usable '{c}'", "fill_missing": f"Fill missing values in '{c}' ({st.get('method')})",
            "zero_as_missing": f"Treat zeros in '{c}' as missing", "flag_outliers": f"{'Cap' if st.get('cap') else 'Flag'} outliers in '{c}'",
            "rename": f"Rename '{c}' to '{st.get('new_name')}'", "ignore_column": f"Leave out column '{c}'",
            "melt_wide": "Turn the wide table into one row per date", "set_aside_repeated_headers": "Set aside repeated header rows",
            "set_aside_totals": "Set aside total rows"}.get(op, op)


def apply_recipe(raw, recipe):
    """Run the enabled steps on a copy of the raw table. Returns (clean, set_aside, log). Raw is never changed."""
    df, asides, log = raw.copy(), [], []
    for i, st in enumerate(recipe, 1):
        if not st.get("enabled", True):
            continue
        try:
            df, moved, entry = OPS[st["op"]](df, st)
        except Exception as ex:
            moved, entry = None, _e(step_title(st), 0, notes=[f"Skipped because of an error: {type(ex).__name__}: {ex}"])
            entry["error"] = True
        if moved is not None and len(moved):
            asides.append(moved)
        entry["step_id"], entry["n"] = st["id"], i
        log.append(entry)
    return df, (pd.concat(asides) if asides else pd.DataFrame()), log


def _match_col(name, columns):
    low = {str(c).lower(): c for c in columns}
    n = str(name).strip().lower()
    if n in low:
        return low[n]
    close = difflib.get_close_matches(n, list(low), n=1, cutoff=0.85)
    return low[close[0]] if close else None


_REQUIRED = {"scale": ["column", "factor"], "set_aside": ["column", "operator"], "replace_values": ["column", "mapping"],
             "rename": ["column", "new_name"], "melt_wide": ["id_cols"], "fill_missing": ["column", "method"]}


def validate_step(step, columns):
    """Check a step proposed by a person or a language model. Returns (clean_step, message)."""
    if not isinstance(step, dict) or step.get("op") not in OPS_SPEC:
        return None, "unknown step type"
    op, spec = step["op"], OPS_SPEC[step["op"]]
    clean = {"op": op}
    for k in spec:
        if k in step:
            clean[k] = step[k]
    for k in _REQUIRED.get(op, ["column"] if "column" in spec else []):
        if k not in clean:
            return None, f"missing '{k}'"
    if "column" in spec:
        col = _match_col(clean.get("column", ""), columns)
        if col is None:
            return None, f"no column called '{clean.get('column')}'"
        clean["column"] = col
    if op == "melt_wide":
        clean["id_cols"] = [c for c in (_match_col(c, columns) for c in clean.get("id_cols", [])) if c]
    if op == "set_aside" and clean["operator"] not in [o.strip() for o in OPS_SPEC["set_aside"]["operator"].split(",")]:
        return None, "unknown operator"
    if op == "scale":
        try:
            clean["factor"] = float(clean["factor"])
        except Exception:
            return None, "the factor must be a number"
    if op == "fill_missing" and clean["method"] not in ("zero", "median", "mean", "ffill", "interpolate"):
        return None, "unknown fill method"
    if op == "replace_values" and not isinstance(clean["mapping"], dict):
        return None, "mapping must be an object"
    return clean, "ok"


def upsert_step(recipe, step):
    """Add a step. If the same kind of step already exists for the column, update it instead of duplicating."""
    step = {**new_step(step["op"]), **step}
    if step["op"] in ("parse_dates", "parse_number", "map_promo", "standardise_text"):
        for st in recipe:
            if st["op"] == step["op"] and st.get("column") == step.get("column"):
                for k, v in step.items():
                    if k in ("id", "enabled"):
                        continue
                    if k == "custom" and isinstance(st.get("custom"), dict):
                        st["custom"] = {**st["custom"], **v}
                    else:
                        st[k] = v
                st["enabled"] = True
                return recipe, st
    recipe.append(step)
    return recipe, step


# =============================================================================
# 4. ANALYSIS: propose steps, ask when unsure
# =============================================================================
def _opt(label, patch=None, message=None):
    return {"label": label, "patch": patch or {"noop": True}, "message": message}


def _question(qid, text, options, detail="", optional=False):
    return {"id": qid, "text": text, "detail": detail, "optional": optional, "options": options}


_CUR_PATTERN = {"USD": r"\$|usd|us\$", "EUR": "\u20ac|eur", "GBP": "\u00a3|gbp", "INR": "\u20b9|\\brs\\.?|inr"}
_NAME_DATE = re.compile(r"(?i)date|\bday\b|time|period|week|month|\bdt\b")
_NAME_PROMO = re.compile(r"(?i)promo|offer|discount|campaign|scheme|deal|coupon")


def analyze(raw):
    """Inspect a table. Returns {'steps', 'questions', 'clean', 'fixes'}.
    'fix' steps are needed for the data to be usable; optional questions are only suggestions."""
    steps, questions = [], []
    work = raw.copy()

    def add(op, fix=True, **p):
        st = new_step(op, **p)
        st["auto"], st["fix"] = True, fix
        steps.append(st)
        return st

    # --- structure: repeated header rows, total rows, wide tables
    for op in ("set_aside_repeated_headers", "set_aside_totals"):
        st = new_step(op)
        w2, moved, _ = OPS[op](work, st)
        if len(moved):
            st["auto"], st["fix"] = True, True
            steps.append(st)
            work = w2
    date_hdr = [c for c in work.columns if re.search(r"\d", str(c)) and pd.notna(_date_one(str(c), False))]
    if len(date_hdr) >= 4 and len(date_hdr) >= 0.5 * len(work.columns):
        st = add("melt_wide", id_cols=[c for c in work.columns if c not in date_hdr])
        work = OPS["melt_wide"](work, st)[0]

    fuzzy_budget = 3
    for c in list(work.columns):
        s = work[c]
        if pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_bool_dtype(s):
            continue
        nn = s.dropna()
        if nn.empty:
            continue
        name_date = bool(_NAME_DATE.search(str(c)))
        if pd.api.types.is_numeric_dtype(s):
            if name_date and nn.between(20000, 60000).all():
                st = add("parse_dates", column=c, dayfirst=None)
            continue
        uniq = pd.Series(pd.unique(nn.astype(object))[:3000])
        na_share = float(uniq.map(lambda v: str(v).strip().lower() in NA_TOKENS).mean())
        d_rate = float(parse_dates(uniq)[0].notna().mean())
        n_vals, _ = parse_number(uniq)
        n_rate = float(n_vals.notna().mean()) + na_share

        # ---- dates
        if (name_date and d_rate >= 0.6) or (d_rate >= 0.9 and n_rate < 0.5):
            _, info = parse_dates(s)
            st = add("parse_dates", column=c, dayfirst=None)
            if info["ambiguous"]:
                st["confirm"] = f"dates_{c}"
                questions.append(_question(
                    f"dates_{c}", f"The dates in '{c}' (like {', '.join(uniq.astype(str).head(2))}) could be day first or month first. Which is right?",
                    [_opt("Day first (1/2/2024 is 1 Feb)", {"set": {"step": st["id"], "values": {"dayfirst": True}}}, "Read as day first."),
                     _opt("Month first (1/2/2024 is 2 Jan)", {"set": {"step": st["id"], "values": {"dayfirst": False}}}, "Read as month first.")]))
            if info["unparsed"]:
                add("set_aside_missing", column=c, reason=f"No readable date in {c}")
            continue

        # ---- numbers
        if n_rate >= 0.8 and n_vals.notna().any():
            _, info = parse_number(s)
            st = add("parse_number", column=c, decimal="auto")
            if len(info["currencies"]) >= 2:
                main, others = info["currencies"][0], info["currencies"][1:]
                opts = [_opt("Keep the numbers as they are", message="Kept as written, with no conversion.")]
                for o in others:
                    opts.append(_opt(f"Set aside the {o} rows", {"add": {**new_step("set_aside", column=c, operator="matches",
                                                                      value=_CUR_PATTERN.get(o, o), reason=f"{o} amount (main currency is {main})"),
                                                                  "auto": True, "fix": True}, "before": st["id"]},
                                     f"Rows in {o} are set aside."))
                questions.append(_question(f"cur_{c}", f"'{c}' mixes {', '.join(info['currencies'])}. I cannot convert without an exchange rate. What should I do?", opts))
            continue

        # ---- promotions
        keys = {_norm_key(v) for v in uniq}
        boolish = len(keys & (_PROMO_NO | _PROMO_YES)) / max(len(keys), 1)
        if _NAME_PROMO.search(str(c)):
            if boolish >= 0.5 or any("%" in k for k in keys):
                st = add("map_promo", column=c)
                _, _, pinfo = parse_promo(s)
                if pinfo["unknown"]:
                    toks = pinfo["unknown"][:5]
                    questions.append(_question(
                        f"promo_{c}", f"In '{c}', I am not sure what {', '.join(repr(t) for t in toks)} mean. Are they promotions?",
                        [_opt("Yes, count them as promotions", message="Counted as promotions."),
                         _opt("No, treat them as no promotion", {"set": {"step": st["id"], "values": {"custom": {t: 0 for t in toks}}}},
                              "Treated as no promotion.")]))
            elif uniq.size <= 60:
                questions.append(_question(
                    f"promosuggest_{c}", f"'{c}' looks like named offers. Should I also create a simple promo flag (any offer counts as a promotion)?",
                    [_opt("Yes, add a promo flag", {"add": {**new_step("map_promo", column=c), "auto": True, "fix": False}}, "Added a promo flag."),
                     _opt("No, keep it as it is")], optional=True))
            continue

        # ---- text: merge case/space variants, suggest near-duplicates
        if uniq.size <= 0.9 * max(len(nn), 1) or uniq.size < 50:
            groups = {}
            for v in nn.astype(str):
                groups.setdefault(_norm_key(v), set()).add(v)
            if any(len(g) > 1 for g in groups.values()):
                add("standardise_text", column=c)
            counts = nn.astype(str).map(_norm_key).value_counts()
            ks = list(counts.index)
            if 2 <= len(ks) <= 60 and fuzzy_budget > 0:
                for i, a in enumerate(ks):
                    for b in ks[i + 1:]:
                        if fuzzy_budget > 0 and difflib.SequenceMatcher(None, a, b).ratio() >= 0.9 and min(counts[a], counts[b]) <= 0.15 * len(nn):
                            small, big = (a, b) if counts[a] < counts[b] else (b, a)
                            questions.append(_question(
                                f"fuzzy_{c}_{small}", f"In '{c}', is '{small}' a typo of '{big}'?",
                                [_opt(f"Yes, merge it into '{big}'", {"add": {**new_step("replace_values", column=c, mapping={small: big}), "auto": True, "fix": False}}, "Merged."),
                                 _opt("No, they are different")], optional=True))
                            fuzzy_budget -= 1
    fixes = sum(1 for s_ in steps if s_.get("fix"))
    return {"steps": steps, "questions": questions, "fixes": fixes,
            "clean": fixes == 0 and not [q for q in questions if not q.get("optional")]}


def answer_question(recipe, q, idx):
    """Apply a clicked answer to the recipe. Returns (recipe, message)."""
    opt = q["options"][idx]
    patch = opt["patch"]
    if "set" in patch:
        for st in recipe:
            if st["id"] == patch["set"]["step"]:
                for k, v in patch["set"]["values"].items():
                    st[k] = {**st.get(k, {}), **v} if k == "custom" and isinstance(v, dict) else v
                st["confirm"] = None
    if "add" in patch and not any(st["id"] == patch["add"]["id"] for st in recipe):
        new = dict(patch["add"])
        pos = next((i for i, st in enumerate(recipe) if st["id"] == patch.get("before")), None)
        recipe.insert(pos if pos is not None else len(recipe), new)
    return recipe, opt.get("message") or f"Got it: {opt['label']}."


# =============================================================================
# 5. A MESSY SAMPLE FILE (for trying the cleaner)
# =============================================================================
def _indian(n):
    s = f"{int(round(n))}"
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    head = re.sub(r"(\d)(?=(\d{2})+$)", r"\1,", head)
    return f"{head},{tail}"


def messy_demo(seed=3, days=150):
    """Build a deliberately messy sales file. Returns (csv_bytes, truth_dataframe)."""
    import forecaster as fc
    rng = np.random.default_rng(seed)
    base = fc.demo_data(seed)
    d0 = pd.Timestamp("2022-03-01")
    base = base[(base["date"] >= d0) & (base["date"] < d0 + pd.Timedelta(days=days))].reset_index(drop=True)
    n = len(base)
    rev = (base["units_sold"] * base["price"] * 60).round()
    rows, status = [], np.where(rng.random(n) < 0.05, "Cancelled", "Delivered")
    for i, r in base.iterrows():
        f1 = rng.random()
        date = (r["date"].strftime("%d/%m/%Y") if f1 < 0.7 else r["date"].strftime("%d %b %Y") if f1 < 0.85 else r["date"].strftime("%d-%b-%y"))
        f2 = rng.random()
        prod = r["product"] if f2 < 0.7 else r["product"].lower() if f2 < 0.8 else r["product"].upper() + " " if f2 < 0.9 else " " + r["product"]
        f3 = rng.random()
        price = f"\u20b9{r['price']:,.2f}" if f3 < 0.5 else f"Rs. {r['price']:.0f}" if f3 < 0.7 else f"{r['price']:.2f}"
        offer = (rng.choice(["Y", "Yes", "20% off", "BOGO", "Diwali Sale"]) if r["promo"] == 1 else rng.choice(["N", "No", "", "none"]))
        units = "N/A" if rng.random() < 0.02 else f"{int(r['units_sold'])}"
        f4 = rng.random()
        rv = rev[i]
        revenue = f"\u20b9{_indian(rv)}" if f4 < 0.5 else f"{rv / 1e5:.2f} lakh" if f4 < 0.7 else f"{rv:,.0f}"
        rows.append([date, prod, price, offer, units, revenue, status[i]])
    head = ["Order Date", "Product", "Unit Price", "Offer", "Units Sold", "Revenue", "Status"]
    lines = [["Daily Sales Report"], ["Generated on 12 Nov 2026"], [], head] + rows[:100] + [head] + rows[100:]
    lines.append(["Total", "", "", "", str(int(base["units_sold"].sum())), "", ""])
    buf = io.StringIO()
    csv.writer(buf, delimiter=";").writerows(lines)
    truth = pd.DataFrame({"date": base["date"], "product": base["product"], "price": base["price"],
                          "promo": base["promo"], "units": base["units_sold"], "revenue": rev, "status": status})
    return ("\ufeff" + buf.getvalue()).encode("utf-8"), truth
