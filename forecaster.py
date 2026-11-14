"""
forecaster.py - the engine behind the app.

Two ways to forecast, both fully automatic:

  A) Time-series mode  : learns from the history of the target (lags, rolling means,
                         calendar) and forecasts the next N periods.
  B) Driver-based mode : learns how demand depends on columns like price, promo,
                         product, season... and predicts demand for new rows.
                         Several models are trained and the best one on a
                         time-based validation split is kept.
"""
import io
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

try:  # optional: pip install xgboost
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None
try:  # optional: pip install holidays
    import holidays as _holidays
except Exception:
    _holidays = None

warnings.filterwarnings("ignore")

CAL_COLS = ["cal_year", "cal_month", "cal_dayofweek", "cal_dayofyear", "cal_week"]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _log1p(x):
    return np.log1p(np.clip(x, 0, None))


def wape(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    denom = np.abs(y).sum()
    return float(np.abs(y - yhat).sum() / denom) if denom else float("nan")


def bias(y, yhat):
    """Signed error as a share of actual demand. Positive = forecasts too high, negative = too low."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float((yhat - y).sum() / y.sum()) if y.sum() else float("nan")


def load_table(file_or_path):
    name = getattr(file_or_path, "name", str(file_or_path)).lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file_or_path)
    return pd.read_csv(file_or_path)


def guess_date_col(df):
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    for c in df.columns:
        if "date" in str(c).lower():
            return c
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            parsed = pd.to_datetime(df[c].head(50), errors="coerce")
            if parsed.notna().mean() > 0.9:
                return c
    return df.columns[0]


def guess_target_col(df, date_col):
    numeric = [c for c in df.columns
               if c != date_col and pd.api.types.is_numeric_dtype(df[c])
               and not pd.api.types.is_bool_dtype(df[c])]
    for key in ("demand", "sales", "units", "quantity", "qty", "volume", "orders", "revenue"):
        for c in numeric:
            if key in str(c).lower() and "base" not in str(c).lower():
                return c
    return numeric[-1] if numeric else None


def demo_data(seed=0):
    """Synthetic daily sales for 3 products (used when no file is uploaded)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=3 * 365, freq="D")
    frames = []
    for i, p in enumerate(["Product A", "Product B", "Product C"]):
        price = 20 + 5 * i + rng.normal(0, 0.5, len(dates))
        promo = (rng.random(len(dates)) < 0.08).astype(int)
        units = (60 + 25 * i + np.linspace(0, 20, len(dates))
                 + 10 * np.sin(2 * np.pi * dates.dayofweek / 7)
                 + 15 * np.sin(2 * np.pi * dates.dayofyear / 365)
                 + 25 * promo - 1.5 * (price - price.mean())
                 + rng.normal(0, 6, len(dates)))
        frames.append(pd.DataFrame({"date": dates, "product": p, "price": price.round(2),
                                    "promo": promo, "units_sold": np.clip(units, 0, None).round()}))
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------------------
# External signals: public holidays and extra data files (weather, macro, ...)
# ----------------------------------------------------------------------------
HOLIDAY_COUNTRIES = {"India": "IN", "United States": "US", "United Kingdom": "GB", "Canada": "CA",
                     "Australia": "AU", "Germany": "DE", "UAE": "AE", "Singapore": "SG"}
_HCAL = {}


def holiday_flags(dates, country_code):
    """1.0 on public holidays of the country, else 0.0 (needs the 'holidays' package)."""
    idx = pd.DatetimeIndex(dates)
    if _holidays is None or not country_code:
        return np.zeros(len(idx))
    if country_code not in _HCAL:
        _HCAL[country_code] = _holidays.country_holidays(country_code, years=range(2015, 2036))
    cal = _HCAL[country_code]
    return np.array([1.0 if d.date() in cal else 0.0 for d in idx])


def merge_external(df, date_col, ext, ext_date_col):
    """Left-join a second file (date + numeric signals) onto the sales data by date."""
    a, e = df.copy(), ext.copy()
    a["_k"] = pd.to_datetime(a[date_col], errors="coerce").dt.normalize()
    e["_k"] = pd.to_datetime(e[ext_date_col], errors="coerce").dt.normalize()
    e = e.drop(columns=[ext_date_col]).dropna(subset=["_k"]).groupby("_k", as_index=False).mean(numeric_only=True)
    keep = [c for c in e.columns if c == "_k" or c not in a.columns]
    return a.merge(e[keep], on="_k", how="left").drop(columns="_k")


# ----------------------------------------------------------------------------
# A) Driver-based model: put in rows of drivers, get demand predictions
# ----------------------------------------------------------------------------
def _clean(df, date_col, target_col=None):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    if target_col is not None:
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
        df = df.dropna(subset=[target_col])
    return df.sort_values(date_col, kind="stable").reset_index(drop=True)


def _infer_meta(df, date_col, target_col):
    num, cat = [], []
    for c in df.columns:
        if c in (date_col, target_col):
            continue
        s = df[c]
        if s.isna().all() or s.nunique(dropna=True) <= 1:
            continue
        if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
            num.append(c)
        elif s.nunique() <= 50:
            cat.append(c)  # text columns with too many values (IDs, free text) are ignored
    fill, log_cols = {}, []
    for c in num:
        v = pd.to_numeric(df[c].astype(float), errors="coerce")
        fill[c] = float(v.median()) if v.notna().any() else 0.0
        if v.min() >= 0:
            log_cols.append(c)
    return {"date_col": date_col, "target_col": target_col, "num": num, "cat": cat,
            "fill": fill, "log_cols": log_cols}


def _design(df, meta):
    X = pd.DataFrame(index=df.index)
    for c in meta["num"]:
        if c not in df:
            raise ValueError(f"Column '{c}' is missing from the new data.")
        s = df[c].astype(float) if pd.api.types.is_bool_dtype(df[c]) else df[c]
        X[c] = pd.to_numeric(s, errors="coerce").fillna(meta["fill"][c])
    for c in meta["cat"]:
        if c not in df:
            raise ValueError(f"Column '{c}' is missing from the new data.")
        X[c] = df[c].astype(str).where(df[c].notna(), "Missing")
    d = pd.to_datetime(df[meta["date_col"]])
    X["cal_year"], X["cal_month"] = d.dt.year, d.dt.month
    X["cal_dayofweek"], X["cal_dayofyear"] = d.dt.dayofweek, d.dt.dayofyear
    X["cal_week"] = d.dt.isocalendar().week.astype(int)
    return X


def _pipeline(kind, meta):
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    num = meta["num"] + CAL_COLS
    if kind == "linear":
        logc = meta["log_cols"]
        plain = [c for c in num if c not in logc]
        pre = ColumnTransformer([
            ("log", Pipeline([("f", FunctionTransformer(_log1p)), ("s", StandardScaler())]), logc),
            ("num", StandardScaler(), plain),
            ("cat", ohe, meta["cat"])])
        model = Ridge(alpha=1.0)
    else:
        pre = ColumnTransformer([("num", "passthrough", num), ("cat", ohe, meta["cat"])])
        if kind == "rf":
            model = RandomForestRegressor(n_estimators=150, min_samples_leaf=3, n_jobs=-1, random_state=0)
        elif kind == "xgb":
            model = XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8,
                                 colsample_bytree=0.8, n_jobs=-1, random_state=0)
        else:
            model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0)
    return Pipeline([("pre", pre), ("model", model)])


def _inverse(pred, use_log):
    return np.exp(np.clip(pred, -50, 50)) if use_log else np.clip(pred, 0, None)


def train_driver_model(df, date_col, target_col, holiday_country=None):
    """Train several models, keep the best on a time-based validation split."""
    d = _clean(df, date_col, target_col)
    if len(d) < 50:
        raise ValueError("Need at least 50 rows with a valid date and target value.")
    if holiday_country:
        d["is_holiday"] = holiday_flags(d[date_col], holiday_country)
    meta = _infer_meta(d, date_col, target_col)
    meta["holiday_country"] = holiday_country
    X, y = _design(d, meta), d[target_col].to_numpy(float)

    cut = d[date_col].iloc[int(len(d) * 0.8)]
    tr = (d[date_col] < cut).to_numpy()
    if tr.sum() < 20 or (~tr).sum() < 10:
        tr = np.arange(len(d)) < int(len(d) * 0.8)
    va = ~tr

    positive = bool((y > 0).all())
    candidates = {"Linear (log-log)": ("linear", positive),
                  "Random Forest": ("rf", False),
                  "Gradient Boosting": ("gbm", False)}
    if XGBRegressor is not None:
        candidates["XGBoost"] = ("xgb", False)
    if positive:
        candidates["Gradient Boosting (log target)"] = ("gbm", True)
        if XGBRegressor is not None:
            candidates["XGBoost (log target)"] = ("xgb", True)

    rows, fitted = [], {}
    for name, (kind, use_log) in candidates.items():
        m = _pipeline(kind, meta)
        m.fit(X[tr], np.log(y[tr]) if use_log else y[tr])
        pred = _inverse(m.predict(X[va]), use_log)
        rows.append({"Model": name, "MAE": mean_absolute_error(y[va], pred),
                     "WAPE": wape(y[va], pred), "R2": r2_score(y[va], pred), "Bias %": bias(y[va], pred)})
        fitted[name] = (m, use_log, kind, pred)

    # Baseline for context: the training average
    base = np.full(va.sum(), y[tr].mean())
    rows.append({"Model": "Baseline (training average)", "MAE": mean_absolute_error(y[va], base),
                 "WAPE": wape(y[va], base), "R2": r2_score(y[va], base), "Bias %": bias(y[va], base)})
    board = pd.DataFrame(rows).sort_values("WAPE").reset_index(drop=True)
    best = board[board["Model"] != "Baseline (training average)"].iloc[0]["Model"]
    m_tr, use_log, kind, val_pred = fitted[best]

    # Which inputs matter (permutation importance on validation rows)
    sample = np.flatnonzero(va)[:2000]
    pi = permutation_importance(m_tr, X.iloc[sample], np.log(y[sample]) if use_log else y[sample],
                                n_repeats=3, random_state=0, scoring="r2")
    importance = (pd.DataFrame({"Feature": X.columns, "Importance": pi.importances_mean})
                  .sort_values("Importance", ascending=False).reset_index(drop=True))

    final = _pipeline(kind, meta)  # refit best model on ALL data
    final.fit(X, np.log(y) if use_log else y)

    return {"meta": meta, "model": final, "log_target": use_log, "best": best,
            "leaderboard": board, "importance": importance,
            "validation": pd.DataFrame({"date": d.loc[va, date_col].values,
                                        "actual": y[va], "predicted": val_pred}),
            "train_rows": int(tr.sum()), "valid_rows": int(va.sum())}


def predict_drivers(bundle, new_df):
    meta = bundle["meta"]
    d = _clean(new_df, meta["date_col"])
    if meta.get("holiday_country"):
        d["is_holiday"] = holiday_flags(d[meta["date_col"]], meta["holiday_country"])
    pred = _inverse(bundle["model"].predict(_design(d, meta)), bundle["log_target"])
    out = d.copy()
    out["Forecast"] = pred.round(2)
    return out


def bundle_to_bytes(bundle):
    buf = io.BytesIO()
    joblib.dump(bundle, buf)
    return buf.getvalue()


# ----------------------------------------------------------------------------
# B) Time-series mode: learn from history, forecast the next N periods
# ----------------------------------------------------------------------------
_FREQ = {
    "D": dict(rule="D", lags=[1, 2, 3, 7, 14, 28], wins=[7, 28], season=7, unit="days"),
    "W": dict(rule="W", lags=[1, 2, 3, 4, 8, 13, 52], wins=[4, 13], season=52, unit="weeks"),
    "M": dict(rule="MS", lags=[1, 2, 3, 6, 12], wins=[3, 6], season=12, unit="months"),
}


def infer_freq(dates):
    gaps = pd.Series(sorted(pd.to_datetime(dates).unique())).diff().dropna().dt.days
    med = gaps.median() if len(gaps) else 1
    return "D" if med <= 1.5 else "W" if med <= 10 else "M"


def _ts_row(vals, date, lags, wins, ex=None):
    n = len(vals)
    row = [vals[n - l] for l in lags] + [float(np.mean(vals[n - w:])) for w in wins]
    row += [date.month, date.dayofyear, date.dayofweek]
    if ex is not None:
        row += [float(v) for v in ex]
    return row


def _fit_series(vals, dates, lags, wins, exog=None):
    t0 = max(lags + wins)
    X = [_ts_row(vals[:t], dates[t], lags, wins, None if exog is None else exog[t])
         for t in range(t0, len(vals))]
    model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=0)
    return model.fit(X, vals[t0:])


def _recursive(model, vals, future_dates, lags, wins, fut_exog=None):
    hist, out = list(vals), []
    for i, dt in enumerate(future_dates):
        ex = None if fut_exog is None else fut_exog[i]
        p = float(model.predict([_ts_row(hist, dt, lags, wins, ex)])[0])
        out.append(p)
        hist.append(p)
    return np.array(out)


def _period_holiday_share(index, rule, country):
    """Share of days in each period that are public holidays (0/1 for daily data)."""
    if rule == "D":
        return holiday_flags(index, country)
    out = []
    for end in index:
        days = (pd.date_range(end - pd.Timedelta(days=6), end) if rule == "W"
                else pd.date_range(end, end + pd.offsets.MonthEnd(0)))
        out.append(holiday_flags(days, country).mean())
    return np.array(out)


def forecast_timeseries(df, date_col, target_col, horizon=30, group_col=None, agg="sum",
                        exog_cols=None, future_exog=None, holiday_country=None):
    """Learn from the history (plus optional driver columns) and forecast the next `horizon` periods.

    exog_cols      numeric driver columns (price, promo, weather, ...) used as model inputs.
    future_exog    optional table (date [+ group] + driver columns) with the planned values for the
                   forecast period. Where it has gaps, the last known value is carried forward.
    holiday_country  adds a public-holiday signal that is known for the future automatically.
    """
    d = _clean(df, date_col, target_col)
    if d.empty:
        raise ValueError("No valid rows after reading the date and target columns.")
    freq = infer_freq(d[date_col])
    cfg = _FREQ[freq]
    exog_cols = [c for c in (exog_cols or []) if c in d.columns and c not in (date_col, target_col, group_col)]
    for c in exog_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    fx = None
    if future_exog is not None and exog_cols:
        fx = future_exog.copy()
        fx[date_col] = pd.to_datetime(fx[date_col], errors="coerce")
        fx = fx.dropna(subset=[date_col])
        if group_col and group_col in fx:
            fx[group_col] = fx[group_col].astype(str)
    if group_col:
        d[group_col] = d[group_col].astype(str)
        groups = list(d.groupby(group_col)[target_col].sum().sort_values(ascending=False).index[:12])
    else:
        groups = [None]
    driver_names = exog_cols + (["is_holiday"] if holiday_country else [])

    fc_rows, hist_rows, met_rows, notes = [], [], [], []
    for g in groups:
        sub = d if g is None else d[d[group_col] == g]
        s = sub.set_index(date_col)[target_col].resample(cfg["rule"]).agg(agg)
        s = s.interpolate().bfill()
        vals, dates = s.to_numpy(float), s.index
        n = len(vals)
        lags = [l for l in cfg["lags"] if l <= max(n // 4, 1)]
        wins = [w for w in cfg["wins"] if w <= max(n // 4, 1)] or [1]
        label = g or "All"
        if n < max(lags) + max(wins) + 15:
            met_rows.append({"Group": label, "Note": "Not enough history", "Periods": n})
            continue
        future = pd.date_range(dates[-1], periods=horizon + 1, freq=cfg["rule"])[1:]

        # Driver matrices: history (n x k) and future (horizon x k)
        ex_h = ex_f = None
        if driver_names:
            parts_h, parts_f = [], []
            if exog_cols:
                e = (sub.set_index(date_col)[exog_cols].resample(cfg["rule"]).mean()
                     .reindex(s.index).interpolate().bfill().ffill())
                parts_h.append(e.to_numpy(float))
                fe = pd.DataFrame(index=future, columns=exog_cols, dtype=float)
                if fx is not None:
                    fsub = fx if (g is None or group_col not in fx) else fx[fx[group_col] == g]
                    have = [c for c in exog_cols if c in fsub.columns]
                    if len(fsub) and have:
                        got = (fsub.set_index(date_col)[have].apply(pd.to_numeric, errors="coerce")
                               .resample(cfg["rule"]).mean())
                        fe = got.reindex(columns=exog_cols).reindex(future)
                missing = int(fe.isna().any(axis=1).sum())
                if missing:
                    notes.append(f"{label}: no driver values provided for {missing} of {horizon} future "
                                 f"periods, so the last known values were carried forward.")
                parts_f.append(fe.fillna(e.iloc[-1]).to_numpy(float))
            if holiday_country:
                parts_h.append(_period_holiday_share(s.index, cfg["rule"], holiday_country)[:, None])
                parts_f.append(_period_holiday_share(future, cfg["rule"], holiday_country)[:, None])
            ex_h, ex_f = np.hstack(parts_h), np.hstack(parts_f)

        # Back-test: hold out the last few periods, compare with a seasonal-naive baseline
        hold = int(min(horizon, max(n // 5, 1)))
        resid_sd = float(np.std(np.diff(vals))) * 0.7
        m_row = {"Group": label, "Periods": n, "Back-test periods": hold,
                 "Drivers": ", ".join(driver_names) or "none"}
        if n - hold >= max(lags + wins) + 15:
            tr_vals, te_vals = vals[:-hold], vals[-hold:]
            m = _fit_series(tr_vals, dates[:-hold], lags, wins, None if ex_h is None else ex_h[:-hold])
            pred = _recursive(m, tr_vals, dates[-hold:], lags, wins, None if ex_h is None else ex_h[-hold:])
            season = cfg["season"]
            base = (np.array([tr_vals[-season + (i % season)] for i in range(hold)])
                    if len(tr_vals) >= season else np.full(hold, tr_vals[-1]))
            resid_sd = float(np.std(te_vals - pred)) or resid_sd
            m_row.update({"Model WAPE": wape(te_vals, pred), "Baseline WAPE": wape(te_vals, base),
                          "MAE": float(np.abs(te_vals - pred).mean()), "Bias %": bias(te_vals, pred),
                          "Baseline bias %": bias(te_vals, base)})
            m_row["Beats baseline"] = bool(m_row["Model WAPE"] < m_row["Baseline WAPE"])
        m_row["Error SD"] = resid_sd
        met_rows.append(m_row)

        final = _fit_series(vals, dates, lags, wins, ex_h)
        pred_f = _recursive(final, vals, future, lags, wins, ex_f)
        lo, hi = np.clip(pred_f - 1.28 * resid_sd, 0, None), pred_f + 1.28 * resid_sd
        fc_rows.append(pd.DataFrame({"group": label, "date": future, "forecast": pred_f.round(2),
                                     "lower": lo.round(2), "upper": hi.round(2)}))
        hist_rows.append(pd.DataFrame({"group": label, "date": dates, "value": vals}))

    if not fc_rows:
        raise ValueError("Not enough history to forecast. Provide more past data.")
    return {"forecast": pd.concat(fc_rows, ignore_index=True),
            "history": pd.concat(hist_rows, ignore_index=True),
            "metrics": pd.DataFrame(met_rows), "freq": cfg["unit"],
            "drivers": driver_names, "notes": notes}


# ----------------------------------------------------------------------------
# Monitoring: compare forecasts with what actually happened, flag drift
# ----------------------------------------------------------------------------
def drift_verdict(wape_now, wape_ref, bias):
    """Simple rule: error well above the back-test level, or a consistent bias, means retrain."""
    reasons = []
    if wape_ref is not None and wape_ref == wape_ref and wape_now > max(1.5 * wape_ref, wape_ref + 0.03):
        reasons.append(f"error is {wape_now:.1%} vs {wape_ref:.1%} when the model was tested")
    if abs(bias) > 0.10:
        reasons.append(f"forecasts run {'too high' if bias > 0 else 'too low'} by {abs(bias):.1%} on average")
    if reasons:
        return True, "Drift detected: " + "; ".join(reasons) + ". Retraining on the newest data is recommended."
    return False, "No significant drift: accuracy is in line with the original test. Keep monitoring."


def compare_to_actuals(forecast_df, actuals_df, date_col, target_col, group_col=None,
                       agg="sum", ref_wape=None):
    """Match a saved forecast (group, date, forecast, lower, upper) to the actuals that arrived."""
    f = forecast_df.copy()
    f["date"] = pd.to_datetime(f["date"])
    f["group"] = f["group"].astype(str)
    rule = _FREQ[infer_freq(f["date"]) if f["date"].nunique() > 2 else "D"]["rule"]

    a = _clean(actuals_df, date_col, target_col)
    parts = []
    for g, sub in (a.groupby(group_col) if group_col else [("All", a)]):
        r = sub.set_index(date_col)[target_col].resample(rule)
        val, cnt = r.agg(agg), r.count()
        parts.append(pd.DataFrame({"group": str(g), "date": val[cnt > 0].index, "actual": val[cnt > 0].values}))
    if not parts:
        raise ValueError("No usable actuals found.")
    m = f.merge(pd.concat(parts), on=["group", "date"], how="inner")
    if m.empty:
        raise ValueError("The actuals don't overlap with the forecast dates yet. "
                         "Upload data that covers the forecasted period.")
    m["error"] = m["forecast"] - m["actual"]
    m["within_range"] = (m["actual"] >= m["lower"]) & (m["actual"] <= m["upper"])

    rows = []
    for g, s in list(m.groupby("group")) + [("ALL", m)]:
        rows.append({"Group": g, "Periods compared": len(s), "MAE": s["error"].abs().mean(),
                     "WAPE": wape(s["actual"], s["forecast"]),
                     "Bias %": s["error"].sum() / s["actual"].sum() if s["actual"].sum() else np.nan,
                     "Actual inside range %": s["within_range"].mean()})
    summary = pd.DataFrame(rows)
    tot = summary.iloc[-1]
    flag, msg = drift_verdict(tot["WAPE"], ref_wape, tot["Bias %"])
    return {"matched": m, "summary": summary, "drift": flag, "message": msg}


def compare_driver_to_actuals(bundle, actuals_df):
    """Re-predict rows from a saved driver model and compare with the actual target values."""
    meta = bundle["meta"]
    if meta["target_col"] not in actuals_df:
        raise ValueError(f"The file needs the actual '{meta['target_col']}' column to compare against.")
    d = _clean(actuals_df, meta["date_col"], meta["target_col"])
    if d.empty:
        raise ValueError("No usable rows with both a date and an actual value.")
    if meta.get("holiday_country"):
        d["is_holiday"] = holiday_flags(d[meta["date_col"]], meta["holiday_country"])
    out = d.copy()
    out["Forecast"] = _inverse(bundle["model"].predict(_design(d, meta)), bundle["log_target"])
    y, p = out[meta["target_col"]].to_numpy(float), out["Forecast"].to_numpy(float)
    bias = float((p - y).sum() / y.sum()) if y.sum() else 0.0
    now = wape(y, p)
    ref = float(bundle["leaderboard"].set_index("Model").loc[bundle["best"], "WAPE"])
    flag, msg = drift_verdict(now, ref, bias)
    summary = pd.DataFrame([{"Rows compared": len(out), "MAE": float(np.abs(p - y).mean()),
                             "WAPE": now, "Bias %": bias, "Original test WAPE": ref}])
    return {"matched": out, "summary": summary, "drift": flag, "message": msg}
