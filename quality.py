"""
quality.py - can you trust this forecast, and is the data behind it healthy?

  bias_status        direction and size of forecast error (too high / too low)
  segment_products   ABC (volume) and XYZ (variability) classes with a trust rating per product
  data_health        checks on the data itself: dates, gaps, duplicates, outliers, zero runs, stockouts
"""
import re

import numpy as np
import pandas as pd

import forecaster as fc


# ------------------------------------------------------------------ bias
def bias_status(b):
    """Returns (level, text). Level: ok, watch, bad, na. Practitioners aim for roughly +-5%."""
    if b is None or pd.isna(b):
        return "na", "not measured"
    a = abs(b)
    side = "too high (over-forecasts)" if b > 0 else "too low (under-forecasts)"
    if a <= 0.05:
        return "ok", f"{b:+.1%}, balanced"
    if a <= 0.10:
        return "watch", f"{b:+.1%}, tends to be {side}"
    return "bad", f"{b:+.1%}, clearly {side}"


# --------------------------------------------------------- product segments
def segment_products(history, metrics=None):
    """One row per group: ABC (share of volume), XYZ (variability), accuracy and a trust rating."""
    tot = history.groupby("group")["value"].sum().sort_values(ascending=False)
    share = tot / tot.sum() if tot.sum() else tot * 0
    cum = share.cumsum()
    met = metrics.set_index("Group") if metrics is not None and "Group" in metrics else pd.DataFrame()
    rows = []
    for g in tot.index:
        v = history.loc[history["group"] == g, "value"]
        mean = v.mean()
        cv = float(v.std() / mean) if mean and mean > 0 else float("nan")
        abc = "A" if cum[g] - share[g] < 0.80 else ("B" if cum[g] - share[g] < 0.95 else "C")
        xyz = "X" if cv < 0.5 else ("Y" if cv < 1.0 else "Z") if pd.notna(cv) else "?"
        m = met.loc[g] if g in met.index else {}
        wape, base, bias = m.get("Model WAPE", np.nan), m.get("Baseline WAPE", np.nan), m.get("Bias %", np.nan)
        beats = m.get("Beats baseline", np.nan)
        why = []
        if pd.isna(wape):
            trust = "Not tested"
            why.append("not enough history for a back-test")
        else:
            beats = bool(beats)
            why.append("beats the simple baseline" if beats else "does not beat the simple baseline")
            if pd.notna(bias) and abs(bias) > 0.05:
                why.append(f"bias {bias:+.0%}")
            high_err = wape > 0.30
            if high_err:
                why.append(f"typical error is high ({wape:.0%})")
            trust = ("Weak" if (not beats or (pd.notna(bias) and abs(bias) > 0.20))
                     else "Reliable" if ((pd.isna(bias) or abs(bias) <= 0.10) and not high_err) else "Check")
        if xyz == "Z":
            why.append("demand is erratic")
        rows.append({"Group": g, "Class": f"{abc}-{xyz}", "Share of volume": float(share[g]), "Variability (CV)": cv,
                     "Model WAPE": wape, "Baseline WAPE": base, "Bias %": bias, "Trust": trust, "Why": "; ".join(why)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------- data health
def _longest_zero_run(values):
    best = cur = 0
    for v in values:
        cur = cur + 1 if v == 0 else 0
        best = max(best, cur)
    return best


def data_health(df, date_col, target_col, group_col=None):
    """List of {check, status, detail}. status: ok, note or warn."""
    out = []

    def add(check, status, detail):
        out.append({"check": check, "status": status, "detail": detail})

    n = len(df)
    dates = pd.to_datetime(df[date_col], errors="coerce")
    y = pd.to_numeric(df[target_col], errors="coerce")
    grp = df[group_col].astype(str) if group_col and group_col in df else pd.Series("all", index=df.index)

    bad = int(dates.isna().sum())
    add("Dates are readable", "ok" if bad == 0 else "warn",
        "All dates were read." if bad == 0 else f"{bad:,} row(s) have a date that could not be read. They are left out of the forecast.")
    miss = int(y.isna().sum())
    add("Target values present", "ok" if miss == 0 else "warn",
        "Every row has a value." if miss == 0 else f"{miss:,} row(s) have no usable '{target_col}'. They are left out of training.")
    if dates.notna().any():
        add("History length", "ok", f"{dates.min():%d %b %Y} to {dates.max():%d %b %Y}, {n:,} rows.")
        span = (dates.max() - dates.min()).days
        if span < 365:
            add("Enough history", "warn", f"Only {span} days of history. Yearly seasonality cannot be learned from less than a year.")
        elif span < 730:
            add("Enough history", "note", f"{span} days of history. Two full years is better for seasonal patterns.")
        else:
            add("Enough history", "ok", f"{span} days of history, enough to see yearly patterns.")

    dup = int(df.duplicated().sum())
    add("Duplicate rows", "ok" if dup == 0 else "warn",
        "No exact duplicate rows." if dup == 0 else f"{dup:,} row(s) are exact copies of another row. They would be counted twice.")
    keys = pd.DataFrame({"d": dates, "g": grp}).dropna()
    shared = int(keys.duplicated().sum())
    add("One row per date", "ok" if shared == 0 else "note",
        "Each date appears once per product." if shared == 0 else
        f"{shared:,} row(s) share a date (and product) with another row. They are combined by sum or mean when you forecast.")

    # gaps in the timeline
    if dates.notna().sum() > 3:
        gaps_total, worst = 0, 0
        for g, ds in keys.groupby("g")["d"]:
            u = np.sort(ds.dropna().unique())
            if len(u) < 4:
                continue
            diffs = np.diff(u).astype("timedelta64[D]").astype(int)
            md = max(np.median(diffs), 1)
            miss_p = sum(max(int(round(d / md)) - 1, 0) for d in diffs)
            gaps_total += miss_p
            worst = max(worst, max(int(round(d / md)) - 1 for d in diffs))
        add("Gaps in the timeline", "ok" if gaps_total == 0 else "note",
            "No missing periods." if gaps_total == 0 else
            f"{gaps_total:,} period(s) are missing (longest gap {worst}). They are filled by interpolation, which can hide real zeros.")

    neg = int((y < 0).sum())
    add("Negative values", "ok" if neg == 0 else "warn",
        "No negative values." if neg == 0 else f"{neg:,} negative value(s). Returns or credit notes may be mixed with sales.")

    # outliers per group (robust)
    med = y.groupby(grp).transform("median")
    mad = (y - med).abs().groupby(grp).transform("median") * 1.4826
    z = (y - med) / mad.replace(0, np.nan)
    n_out = int((z.abs() > 6).sum())
    if n_out:
        worst_val = y[(z.abs() > 6)].abs().max()
        add("Extreme values", "warn", f"{n_out:,} value(s) are far from the normal range (largest {worst_val:,.0f}). "
                                      "Check for typing errors or units.")
    else:
        add("Extreme values", "ok", "No extreme outliers.")

    # long runs of zeros
    runs = []
    if dates.notna().any():
        tmp = pd.DataFrame({"d": dates, "g": grp, "y": y}).dropna(subset=["d"])
        for g, sub in tmp.groupby("g"):
            r = _longest_zero_run(sub.groupby("d")["y"].sum().sort_index().to_numpy())
            if r >= 7:
                runs.append((g, r))
    add("Long runs of zeros", "ok" if not runs else "warn",
        "No long stretches of zero demand." if not runs else
        "; ".join(f"{g}: {r} periods in a row" for g, r in runs[:4]) + ". These may be stockouts or inactive products, not real zero demand.")

    # stock-like columns
    for c in df.columns:
        if c in (date_col, target_col) or not re.search(r"(?i)stock|inventory|avail", str(c)):
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() == 0:
            continue
        low = int((s <= 0).sum()) if s.max() > 1 else int((s <= 0.05).sum())
        add(f"Stock levels ('{c}')", "ok" if low == 0 else "note",
            "No empty-stock rows." if low == 0 else
            f"{low:,} row(s) look out of stock. Sales on those days were capped, so they understate real demand.")

    # numbers hiding in text columns
    numeric_text = []
    for c in df.columns:
        if c == date_col or df[c].dtype != object:
            continue
        v = df[c].dropna().astype(str)
        if len(v) and v.str.match(r"^[\d.,\s\u20b9$\u20ac\u00a3%()\-]+$").mean() >= 0.8:
            numeric_text.append(str(c))
    if numeric_text:
        add("Numbers stored as text", "warn", "These columns look numeric but are text: " + ", ".join(numeric_text[:5]) +
            ". Use Prepare data to convert them.")
    const = [str(c) for c in df.columns if df[c].nunique(dropna=True) <= 1]
    if const:
        add("Constant columns", "note", "These columns never change and add no information: " + ", ".join(const[:6]) + ".")
    return out
