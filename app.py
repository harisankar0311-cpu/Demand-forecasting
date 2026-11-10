"""
Demand Forecaster  -  run with:  python -m streamlit run app.py
Upload sales history -> forecast -> inventory plan -> monitoring.
"""
import html
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import forecaster as fc
import inventory as inv

# ---------------------------------------------------------------- THEME FILE
THEME_TOML = """[theme]
base = "dark"
primaryColor = "#8B5CF6"
backgroundColor = "#000000"
secondaryBackgroundColor = "#130F22"
textColor = "#F7F5FF"
"""


def ensure_theme():
    """Create .streamlit/config.toml once so tables and widgets also use the purple/black theme."""
    try:
        path = os.path.join(os.getcwd(), ".streamlit", "config.toml")
        if os.path.exists(path):
            return False
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            fh.write(THEME_TOML)
        return True
    except Exception:
        return False


THEME_JUST_INSTALLED = ensure_theme()

st.set_page_config(page_title="Demand Forecaster", page_icon="🟣", layout="wide",
                   initial_sidebar_state="expanded")

# ----------------------------------------------------------------------- CSS
CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Sora:wght@500;600;700&display=swap');
:root{--violet:#8B5CF6;--deep:#6D28D9;--lilac:#C4B5FD;--paper:#F7F5FF;--muted:#A89FC9;--line:rgba(196,181,253,.22);--plum:#130F22;}
html,body,.stApp{font-family:'DM Sans','Segoe UI',sans-serif;}
.stApp{background:radial-gradient(1100px 560px at 12% -12%,rgba(139,92,246,.30),transparent 62%),
       radial-gradient(700px 420px at 100% 4%,rgba(109,40,217,.20),transparent 60%),#000;color:var(--paper);}
header[data-testid="stHeader"]{background:transparent;}
.block-container{padding-top:2rem;max-width:1380px;}
h1,h2,h3,h4{font-family:'Sora','DM Sans',sans-serif!important;color:#fff!important;letter-spacing:-.015em;}
p,li,label,.stMarkdown,[data-testid="stWidgetLabel"]{color:var(--paper);}
[data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] *{color:var(--muted)!important;}

/* sidebar */
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0D0919,#000);border-right:1px solid var(--line);}
.brand{font-family:'Sora',sans-serif;font-weight:700;font-size:1.25rem;color:#fff;margin:.2rem 0 1rem;display:flex;align-items:center;gap:.55rem;}
.brand i{width:14px;height:14px;border-radius:50%;background:radial-gradient(circle at 30% 30%,#fff,var(--violet) 55%,var(--deep));box-shadow:0 0 14px var(--violet);display:inline-block;}
.side-h{font-family:'Sora',sans-serif;font-weight:600;font-size:.95rem;color:var(--lilac);margin:1.1rem 0 .3rem;}

/* inputs */
[data-baseweb="select"]>div,[data-baseweb="input"]>div,.stNumberInput input,.stTextInput input{
  background:#120E20!important;border:1px solid var(--line)!important;border-radius:10px!important;color:#fff!important;}
[data-baseweb="select"]>div:hover,[data-baseweb="input"]>div:hover,[data-baseweb="select"]>div:focus-within{border-color:var(--violet)!important;}
[data-testid="stFileUploaderDropzone"]{background:rgba(139,92,246,.07);border:1.5px dashed rgba(196,181,253,.5);border-radius:14px;}
[data-testid="stFileUploaderDropzone"]:hover{background:rgba(139,92,246,.15);border-color:var(--lilac);}
[data-testid="stExpander"]{background:rgba(255,255,255,.035);border:1px solid var(--line);border-radius:12px;}

/* buttons: react to the person's action */
.stButton>button,.stDownloadButton>button{border-radius:10px;border:1px solid rgba(255,255,255,.55);background:transparent;color:#fff;font-weight:700;transition:background .15s,box-shadow .15s,transform .1s;}
.stButton>button:hover,.stDownloadButton>button:hover{background:rgba(255,255,255,.1);border-color:#fff;color:#fff;}
.stButton>button:active,.stDownloadButton>button:active{transform:scale(.98);}
.stButton>button[kind="primary"],button[data-testid="stBaseButton-primary"]{background:linear-gradient(135deg,#8B5CF6,#6D28D9);border:none;color:#fff;box-shadow:0 6px 22px rgba(139,92,246,.45);}
.stButton>button[kind="primary"]:hover,button[data-testid="stBaseButton-primary"]:hover{box-shadow:0 8px 30px rgba(139,92,246,.7);background:linear-gradient(135deg,#9D77FF,#7C3AED);}

/* tabs */
.stTabs [data-baseweb="tab-list"]{gap:4px;border-bottom:1px solid var(--line);}
.stTabs [data-baseweb="tab"]{height:auto;padding:.6rem 1.1rem;color:var(--muted);font-family:'Sora',sans-serif;font-weight:600;background:transparent;}
.stTabs [aria-selected="true"]{color:#fff!important;}
.stTabs [data-baseweb="tab-highlight"]{background:var(--violet);height:3px;border-radius:3px;}
.stTabs [data-baseweb="tab-border"]{display:none;}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:12px;overflow:hidden;}
[data-testid="stAlert"]{border-radius:12px;border:1px solid var(--line);background:rgba(139,92,246,.10);}

/* hero and headings */
.hero{padding:3.2rem 0 1.4rem;max-width:880px;}
.hero h1{font-size:3.4rem;line-height:1.08;margin:0 0 1rem;font-weight:700;}
.hero p{font-size:1.15rem;color:var(--lilac);max-width:620px;line-height:1.55;}
.hero-sm h1{font-size:2rem;margin:0;font-weight:700;}
.hero-sm p{color:var(--muted);margin:.25rem 0 .7rem;}
.facts{display:flex;gap:.5rem;flex-wrap:wrap;margin-bottom:1.1rem;}
.fact{border:1px solid var(--line);color:var(--lilac);padding:.2rem .8rem;border-radius:999px;font-size:.85rem;}
.how{display:grid;grid-template-columns:repeat(3,1fr);gap:2rem;margin-top:2.6rem;max-width:1000px;}
.how div{border-top:2px solid var(--violet);padding-top:.8rem;}
.how b{font-family:'Sora',sans-serif;font-size:1.05rem;color:#fff;display:block;margin-bottom:.35rem;}
.how span{color:var(--muted);line-height:1.5;}
.sec h3{font-size:1.25rem;margin:1.6rem 0 .15rem;}
.sec p{color:var(--muted);margin:0 0 .8rem;}

/* key numbers: plain text separated by lines */
.kstrip{display:flex;flex-wrap:wrap;margin:.6rem 0 1.2rem;}
.kstrip>div{padding:.1rem 2rem .1rem 0;margin-right:2rem;border-right:1px solid var(--line);min-width:150px;}
.kstrip>div:last-child{border-right:none;}
.kv{display:block;font-family:'Sora',sans-serif;font-weight:700;font-size:1.9rem;color:#fff;line-height:1.15;}
.kl{display:block;color:var(--lilac);font-size:.88rem;margin-top:.15rem;}
.ks{display:block;color:var(--muted);font-size:.8rem;}

/* the forecast stage: the one pale panel on the black page */
[class*="st-key-stage"]{background:linear-gradient(165deg,#FFFFFF 0%,#ECE6FF 100%);border-radius:22px;padding:1.1rem 1.2rem .4rem;
  box-shadow:0 0 0 1px rgba(255,255,255,.7),0 24px 80px rgba(139,92,246,.35);animation:reveal .7s ease both;}
@keyframes reveal{from{opacity:0;transform:translateY(14px) scale(.985)}to{opacity:1;transform:none}}

/* stock status rows: the coloured edge encodes risk */
.prow{display:grid;grid-template-columns:1.3fr 1fr 1fr 1fr 1fr auto;gap:1rem;align-items:center;padding:.8rem 1rem;margin-bottom:.45rem;
  background:rgba(255,255,255,.04);border-radius:8px;border-left:4px solid var(--muted);}
.prow.hi{border-left-color:#FB7185;}.prow.md{border-left-color:#FBBF24;}.prow.lo{border-left-color:#34D399;}
.prow .pn{font-family:'Sora',sans-serif;font-weight:600;color:#fff;}
.prow .pa{font-weight:700;color:#fff;}.prow .pm{color:var(--muted);font-size:.9rem;}
.chip{padding:.15rem .7rem;border-radius:999px;font-size:.8rem;font-weight:700;}
.chip.hi{background:rgba(251,113,133,.16);color:#FDA4AF;}.chip.md{background:rgba(251,191,36,.16);color:#FCD34D;}.chip.lo{background:rgba(52,211,153,.16);color:#6EE7B7;}
[class*="st-key-po_"]{background:rgba(255,255,255,.04);border:1px solid var(--line);border-radius:12px;padding:1rem 1.2rem;margin-bottom:.8rem;}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important;}}
@media (max-width:900px){.how{grid-template-columns:1fr;}.hero h1{font-size:2.4rem;}.prow{grid-template-columns:1fr 1fr;}}
"""
st.markdown("<style>" + CSS + "</style>", unsafe_allow_html=True)


# ------------------------------------------------------------------- HELPERS
def style_fig(fig, h=380, light=False):
    ink = "#2A1B52" if light else "#E9E3FF"
    grid = "rgba(42,27,82,.10)" if light else "rgba(255,255,255,.07)"
    fig.update_layout(
        template="plotly_white" if light else "plotly_dark", height=h,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=ink, family="DM Sans, Segoe UI, sans-serif"), margin=dict(l=6, r=6, t=30, b=6),
        hovermode="x unified", legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#1B1530", bordercolor="#8B5CF6", font=dict(color="#FFFFFF")))
    fig.update_xaxes(gridcolor=grid, zeroline=False)
    fig.update_yaxes(gridcolor=grid, zeroline=False)
    return fig


def show(fig, h=380, light=False):
    st.plotly_chart(style_fig(fig, h, light), theme=None, width="stretch")


def section(title, sub=""):
    st.markdown(f'<div class="sec"><h3>{html.escape(title)}</h3><p>{html.escape(sub)}</p></div>',
                unsafe_allow_html=True)


def kstrip(items):
    cells = "".join(
        f'<div><span class="kv">{html.escape(str(v))}</span><span class="kl">{html.escape(l)}</span>'
        + (f'<span class="ks">{html.escape(s)}</span>' if s else "") + "</div>" for l, v, s in items)
    st.markdown(f'<div class="kstrip">{cells}</div>', unsafe_allow_html=True)


def pick_group(names, label, key):
    if len(names) == 1:
        return names[0]
    k = f"{key}_{abs(hash(tuple(names))) % 10**6}"
    if len(names) <= 6:
        return st.segmented_control(label, names, default=names[0], key=k, label_visibility="collapsed") or names[0]
    return st.selectbox(label, names, key=k)


def peek():
    d = pd.DataFrame({"d": pd.to_datetime(df[date_col], errors="coerce"),
                      "v": pd.to_numeric(df[target_col], errors="coerce")}).dropna()
    s = d.groupby("d")["v"].sum().reset_index()
    fig = go.Figure(go.Scatter(x=s["d"], y=s["v"], mode="lines", name=target_col, line=dict(color="#8B5CF6", width=2),
                               fill="tozeroy", fillcolor="rgba(139,92,246,.16)"))
    show(fig, 300)


def upload_box(label, key):
    return st.file_uploader(label, type=["csv", "xlsx", "xls"], key=key)


# ------------------------------------------------------------- 1. LANDING
def landing():
    st.markdown("""
    <div class="hero">
      <h1>Know what to order before you run out.</h1>
      <p>Upload your sales history. Get a demand forecast, the stock you should hold, and purchase orders
      ready for your approval.</p>
    </div>""", unsafe_allow_html=True)
    c = st.columns([1.1, 3])
    c[0].button("Try with demo data", type="primary", width="stretch",
                on_click=lambda: st.session_state.update(demo=True))
    c[1].caption("Or upload your own CSV or Excel file in the sidebar. You need a date column and a number to forecast.")
    st.markdown("""
    <div class="how">
      <div><b>Forecast demand</b><span>Learns from past sales, price, promotions and holidays, then projects the weeks ahead.</span></div>
      <div><b>Plan your stock</b><span>Turns the forecast into safety stock, reorder points and order quantities.</span></div>
      <div><b>Stay accurate</b><span>Compares forecasts with real sales and tells you when the model needs retraining.</span></div>
    </div>""", unsafe_allow_html=True)


# --------------------------------------------------------- 2. SIDEBAR (data)
with st.sidebar:
    st.markdown('<div class="brand"><i></i>Demand Forecaster</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-h">Your data</div>', unsafe_allow_html=True)
    up = st.file_uploader("Sales history", type=["csv", "xlsx", "xls"], label_visibility="collapsed")
    st.toggle("Use demo data", key="demo")

df = fc.load_table(up) if up else (fc.demo_data() if st.session_state.get("demo") else None)
if df is None:
    landing()
    st.stop()

# --------------------------------------------------------- 3. SIDEBAR (setup)
cols = list(df.columns)
numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
g_date, = [fc.guess_date_col(df)]
g_target = fc.guess_target_col(df, g_date)
with st.sidebar:
    st.markdown('<div class="side-h">Setup</div>', unsafe_allow_html=True)
    date_col = st.selectbox("Date column", cols, index=cols.index(g_date))
    target_col = st.selectbox("What to forecast", numeric,
                              index=numeric.index(g_target) if g_target in numeric else 0)
    with st.expander("External signals and holidays"):
        ext_file = upload_box("Extra file with a date and numeric signals (weather, inflation, ad spend)", "ext")
        if ext_file:
            try:
                ext = fc.load_table(ext_file)
                ext_date = st.selectbox("Date column in that file", list(ext.columns),
                                        index=list(ext.columns).index(fc.guess_date_col(ext)))
                before = set(df.columns)
                df = fc.merge_external(df, date_col, ext, ext_date)
                added = [c for c in df.columns if c not in before]
                st.success(f"Added: {', '.join(added) or 'nothing new (names already exist)'}")
            except Exception as e:
                st.error(f"Could not merge that file: {e}")
        country = st.selectbox("Public holidays for", ["(none)"] + list(fc.HOLIDAY_COUNTRIES))
        holiday_country = None if country.startswith("(") else fc.HOLIDAY_COUNTRIES[country]
        if holiday_country and fc._holidays is None:
            st.warning("Run: pip install holidays")
            holiday_country = None
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    st.markdown('<div class="side-h">Mode</div>', unsafe_allow_html=True)
    mode = st.radio("Mode", ["Forecast the future", "Predict from drivers"], label_visibility="collapsed",
                    help="Future: learn from history and forecast the next periods. "
                         "Drivers: predict demand for rows with given price, promo, product and so on.")
    group_col, agg, horizon, drivers, future_df = None, "sum", 30, [], None
    if mode == "Forecast the future":
        group_opts = ["(total)"] + [c for c in cols if c not in (date_col, target_col)
                                    and not pd.api.types.is_float_dtype(df[c]) and df[c].nunique() <= 30]
        gp = st.selectbox("Forecast separately for each", group_opts)
        group_col = None if gp == "(total)" else gp
        agg = st.selectbox("Rows sharing a date are combined by", ["sum", "mean"])
        horizon = st.slider("Periods ahead", 7, 180, 30)
        drivers = st.multiselect("Drivers", [c for c in numeric if c not in (date_col, target_col, group_col)],
                                 help="Use only what you will know in advance, such as planned price or promotions.")
        if drivers:
            ff = upload_box("Planned driver values for the forecast period (optional)", "future_drivers")
            future_df = fc.load_table(ff) if ff else None
            st.caption("Without this file, the last known values are carried forward.")
    run = st.button("Train and forecast" if mode == "Forecast the future" else "Train models",
                    type="primary", width="stretch")

# ----------------------------------------------------------------- 4. HEADER
dts = pd.to_datetime(df[date_col], errors="coerce")
span = f"{dts.min():%d %b %Y} to {dts.max():%d %b %Y}" if dts.notna().any() else "dates not recognised"
st.markdown(f"""<div class="hero-sm"><h1>Demand Forecaster</h1>
<p>Forecast, plan stock, and check accuracy.</p></div>
<div class="facts"><span class="fact">{len(df):,} rows</span><span class="fact">{len(df.columns)} columns</span>
<span class="fact">{html.escape(span)}</span></div>""", unsafe_allow_html=True)
if THEME_JUST_INSTALLED:
    st.info("Theme file created. Stop the app (Ctrl+C in the terminal) and start it again to apply the full "
            "purple and black theme to tables and widgets.")


# ------------------------------------------------------- 5. FUTURE FORECAST
def render_future():
    if run:
        with st.status("Training on your history", expanded=True) as stt:
            st.write("Reading and aligning your data")
            st.write("Learning patterns, back-testing and building the forecast")
            try:
                st.session_state["ts"] = fc.forecast_timeseries(
                    df, date_col, target_col, horizon, group_col, agg,
                    exog_cols=drivers, future_exog=future_df, holiday_country=holiday_country)
                st.session_state["ts_params"] = dict(
                    date_col=date_col, target_col=target_col, group_col=group_col, agg=agg, horizon=horizon,
                    exog_cols=drivers, future_exog=future_df, holiday_country=holiday_country)
                stt.update(label="Forecast ready", state="complete", expanded=False)
                st.toast("Forecast ready")
            except Exception as e:
                st.session_state.pop("ts", None)
                stt.update(label="Training failed", state="error")
                st.error(str(e))

    res = st.session_state.get("ts")
    if not res:
        section("Your data at a glance", "Choose your settings in the sidebar, then select Train and forecast.")
        peek()
        return

    p = st.session_state["ts_params"]
    fdf, hdf, met, unit = res["forecast"], res["history"], res["metrics"], res["freq"]
    names = list(fdf["group"].unique())
    group_label = (p["group_col"] or "Group").replace("_", " ")
    if res["drivers"]:
        st.caption("Drivers used: " + ", ".join(res["drivers"]))
    for note in res["notes"][:3]:
        st.warning(note)

    t_fc, t_acc, t_inv, t_mon, t_data = st.tabs(["Forecast", "Accuracy", "Inventory", "Monitor", "Data"])

    # ---- Forecast
    with t_fc:
        pick = pick_group(names, group_label, "grp")
        f, h = fdf[fdf["group"] == pick], hdf[hdf["group"] == pick]
        mrow = met[met["Group"] == pick].iloc[0]
        H = len(f)
        total_f, prev = float(f["forecast"].sum()), float(h["value"].tail(H).sum())
        err = mrow.get("Model WAPE", np.nan)
        beats = mrow.get("Beats baseline", np.nan)
        kstrip([(f"Forecast for the next {H} {unit}", f"{total_f:,.0f}",
                 f"{total_f / prev - 1:+.1%} vs the previous {H}" if prev > 0 else ""),
                ("Typical error in testing", f"{err:.1%}" if pd.notna(err) else "n/a", "lower is better"),
                ("Beats a simple baseline", "n/a" if pd.isna(beats) else ("Yes" if beats else "No"),
                 "repeat last season")])
        show_band = st.toggle("Show likely range", value=True)
        with st.container(key="stage"):
            fig = go.Figure()
            fig.add_scatter(x=h["date"], y=h["value"], name="History", line=dict(color="#1A1033", width=1.8))
            if show_band:
                fig.add_scatter(x=f["date"], y=f["upper"], line=dict(width=0), showlegend=False, hoverinfo="skip")
                fig.add_scatter(x=f["date"], y=f["lower"], fill="tonexty", line=dict(width=0),
                                fillcolor="rgba(124,58,237,.20)", name="Likely range (about 80%)")
            fig.add_scatter(x=f["date"], y=f["forecast"], name="Forecast", line=dict(color="#7C3AED", width=3.2))
            x0 = f["date"].iloc[0]
            fig.add_shape(type="line", x0=x0, x1=x0, y0=0, y1=1, yref="paper",
                          line=dict(color="rgba(42,27,82,.45)", dash="dot", width=1))
            fig.add_annotation(x=x0, y=1.0, yref="paper", text="Forecast starts", showarrow=False,
                               xanchor="left", yanchor="bottom", font=dict(size=11, color="#6D28D9"))
            start = h["date"].iloc[-min(len(h), max(H * 4, 60))]
            fig.update_xaxes(range=[start, f["date"].iloc[-1]], rangeslider=dict(visible=False),
                             rangeselector=dict(bgcolor="#ECE6FF", activecolor="#7C3AED", font=dict(color="#2A1B52"),
                                                buttons=[dict(count=1, label="1m", step="month", stepmode="backward"),
                                                         dict(count=3, label="3m", step="month", stepmode="backward"),
                                                         dict(count=6, label="6m", step="month", stepmode="backward"),
                                                         dict(step="all", label="All")]))
            show(fig, 440, light=True)
        st.download_button("Download forecast", fdf.to_csv(index=False), "forecast.csv", "text/csv")

    # ---- Accuracy
    with t_acc:
        done = met.dropna(subset=["Model WAPE"]) if "Model WAPE" in met else met.iloc[0:0]
        if len(done):
            wins = int(done["Beats baseline"].astype(bool).sum())
            section("How reliable is it?",
                    f"The model beat the simple baseline for {wins} of {len(done)} groups in a back-test on the "
                    f"latest periods it had not seen. Lower error is better.")
            fig = go.Figure()
            fig.add_bar(x=done["Group"], y=done["Model WAPE"], name="Model", marker_color="#8B5CF6")
            fig.add_bar(x=done["Group"], y=done["Baseline WAPE"], name="Baseline", marker_color="#F7F5FF")
            fig.update_layout(barmode="group", yaxis_tickformat=".0%")
            show(fig, 320)
        st.dataframe(met.round(3), width="stretch", hide_index=True)

    # ---- Inventory
    with t_inv:
        section("Inventory planner", f"Turns the forecast into safety stock, reorder points and orders. "
                                     f"Planning periods are {unit}. Replace the sample numbers with your own.")
        k1, k2, k3, k4 = st.columns(4)
        service = k1.slider("Service level (%)", 80, 99, 95) / 100
        order_cost = k2.number_input("Cost per order", min_value=0.0, value=50.0)
        hold_rate = k3.slider("Holding cost per year (% of unit cost)", 5, 50, 20) / 100
        review = k4.number_input(f"Review period ({unit})", min_value=1, value=1)

        avg_f = fdf.groupby("group")["forecast"].mean()
        base = pd.DataFrame({"group": names})
        base["on_hand"] = (avg_f[base["group"]].to_numpy() * 5).round()
        base["on_order"] = 0.0
        base["lead_time"] = 7 if unit == "days" else 2
        base["unit_cost"] = 100.0
        base["moq"] = 1
        params = st.data_editor(
            base, hide_index=True, width="stretch", key="inv_" + "_".join(names)[:60],
            column_config={"group": st.column_config.TextColumn(group_label.title(), disabled=True),
                           "on_hand": "Stock on hand", "on_order": "On order",
                           "lead_time": f"Lead time ({unit})", "unit_cost": "Unit cost", "moq": "Min order qty"})
        plan = inv.plan_inventory(fdf, met, hdf, params, unit, service, order_cost, hold_rate, review)
        kstrip([("Products to reorder now", f"{int(plan['Order now?'].sum())}", f"of {len(plan)}"),
                ("Total order value", f"{plan['Order value'].sum():,.0f}", ""),
                ("Cash tied up in excess stock", f"{plan['Excess value'].sum():,.0f}", "")])

        rows = ""
        for _, r in plan.sort_values("Order now?", ascending=False).iterrows():
            c = {"High": "hi", "Medium": "md", "Low": "lo"}[r["Stockout risk"]]
            act = f"Order {r['Order qty']:,.0f}" if r["Order now?"] else "No order needed"
            rows += (f'<div class="prow {c}"><span class="pn">{html.escape(r["Group"])}</span>'
                     f'<span class="pa">{act}</span><span class="pm">Cover {r["Cover (periods)"]:.1f} {unit}</span>'
                     f'<span class="pm">Reorder point {r["Reorder point"]:,.0f}</span>'
                     f'<span class="pm">Position {r["Inventory position"]:,.0f}</span>'
                     f'<span class="chip {c}">{r["Stockout risk"]} risk</span></div>')
        st.markdown(rows, unsafe_allow_html=True)

        fig = go.Figure()
        fig.add_bar(x=plan["Group"], y=plan["Inventory position"], name="Inventory position", marker_color="#F7F5FF")
        fig.add_bar(x=plan["Group"], y=plan["Reorder point"], name="Reorder point", marker_color="#8B5CF6")
        fig.add_bar(x=plan["Group"], y=plan["Order-up-to level"], name="Order-up-to level", marker_color="#6D28D9")
        fig.update_layout(barmode="group")
        show(fig, 320)
        with st.expander("Full planning table"):
            st.dataframe(plan.round(1), width="stretch", hide_index=True)

        spikes = inv.detect_spikes(hdf)
        section("Unusual demand", "Spikes and drops in the most recent periods.")
        if len(spikes):
            st.dataframe(spikes.round(1), width="stretch", hide_index=True)
        else:
            st.caption("Nothing unusual in the recent history.")

        section("Draft orders for your approval", "The planner drafts each order and explains why. Nothing is sent.")
        drafts = inv.draft_purchase_orders(plan, spikes, unit, service)
        if not drafts:
            st.caption("Nothing to order or flag right now.")
        with st.expander("Reword the drafts with Claude (needs an API key)"):
            key = st.text_input("Anthropic API key", type="password")
            mdl = st.text_input("Model", value="claude-sonnet-5-5")
            if st.button("Reword drafts") and key:
                try:
                    st.session_state["polished"] = {
                        d["group"]: inv.llm_polish(d["text"], key, mdl) for d in drafts if d["kind"] == "PO"}
                except Exception as e:
                    st.error(f"Could not reach the model: {e}")
        polished = st.session_state.get("polished", {})
        approved = []
        for d in drafts:
            text = polished.get(d["group"], d["text"]) if d["kind"] == "PO" else d["text"]
            with st.container(key="po_" + "".join(ch for ch in d["group"] if ch.isalnum())):
                st.text(text)
                if d["kind"] == "PO":
                    choice = st.segmented_control("Decision", ["Pending", "Approved", "Rejected"],
                                                  default="Pending", key=f"dec_{d['group']}") or "Pending"
                    if choice == "Approved":
                        approved.append(text)
        if approved:
            st.download_button("Download approved orders", "\n\n".join(approved), "purchase_orders.txt")

    # ---- Monitor
    with t_mon:
        section("Compare the forecast with what happened",
                "Upload the real numbers when they arrive. The app checks accuracy and says if the model has drifted.")
        n = st.session_state.get("n_ts", 0)
        act_file = upload_box("Actual results (same columns as your original file)", f"act_ts_{n}")
        if act_file:
            try:
                actuals = fc.load_table(act_file)
                ref = met["Model WAPE"].mean() if "Model WAPE" in met else None
                cmp = fc.compare_to_actuals(fdf, actuals, p["date_col"], p["target_col"], p["group_col"], p["agg"], ref)
                (st.error if cmp["drift"] else st.success)(cmp["message"])
                st.dataframe(cmp["summary"].round(3), width="stretch", hide_index=True)
                mm = cmp["matched"]
                gsel = pick_group(sorted(mm["group"].unique()), group_label, "mon")
                mm = mm[mm["group"] == gsel]
                fig = go.Figure()
                fig.add_scatter(x=mm["date"], y=mm["actual"], name="Actual", line=dict(color="#F7F5FF", width=2))
                fig.add_scatter(x=mm["date"], y=mm["forecast"], name="Forecast", line=dict(color="#8B5CF6", width=3))
                show(fig, 320)
                if st.button("Retrain with these actuals included"):
                    merged = pd.concat([df, actuals], ignore_index=True).drop_duplicates()
                    st.session_state["ts"] = fc.forecast_timeseries(
                        merged, p["date_col"], p["target_col"], p["horizon"], p["group_col"], p["agg"],
                        exog_cols=p.get("exog_cols"), future_exog=p.get("future_exog"),
                        holiday_country=p.get("holiday_country"))
                    st.session_state["n_ts"] = n + 1
                    st.rerun()
            except Exception as e:
                st.error(str(e))

    with t_data:
        st.dataframe(df.head(500), width="stretch", hide_index=True)


# ----------------------------------------------------------- 6. DRIVER MODE
def render_drivers():
    if run:
        with st.status("Training and comparing models", expanded=True) as stt:
            st.write("Preparing features and splitting by time")
            st.write("Fitting each model and scoring it on the latest rows")
            try:
                st.session_state["drv"] = fc.train_driver_model(df, date_col, target_col, holiday_country)
                stt.update(label="Models ready", state="complete", expanded=False)
                st.toast("Models ready")
            except Exception as e:
                st.session_state.pop("drv", None)
                stt.update(label="Training failed", state="error")
                st.error(str(e))

    bundle = st.session_state.get("drv")
    if not bundle:
        section("Your data at a glance", "Choose your settings in the sidebar, then select Train models.")
        peek()
        return

    board = bundle["leaderboard"]
    best = board[board["Model"] == bundle["best"]].iloc[0]
    kstrip([("Best model", bundle["best"], ""), ("Typical error", f"{best['WAPE']:.1%}", "on the latest rows"),
            ("R squared", f"{best['R2']:.3f}", "1.0 is a perfect fit"),
            ("Trained on", f"{bundle['train_rows']:,} rows", f"tested on {bundle['valid_rows']:,}")])
    t_mod, t_pred, t_mon, t_data = st.tabs(["Models", "Predict", "Monitor", "Data"])

    with t_mod:
        left, right = st.columns(2)
        with left:
            section("Model comparison", "Scored on the most recent rows, which were held out of training.")
            st.dataframe(board.round(3), width="stretch", hide_index=True)
        with right:
            section("What drives demand", "How much accuracy drops when each input is scrambled.")
            imp = bundle["importance"].head(10).iloc[::-1]
            fig = go.Figure(go.Bar(x=imp["Importance"], y=imp["Feature"], orientation="h", marker_color="#8B5CF6"))
            show(fig, 320)
        section("Actual and predicted on the held-out period")
        v = bundle["validation"].groupby("date")[["actual", "predicted"]].sum().reset_index()
        with st.container(key="stage"):
            fig = go.Figure()
            fig.add_scatter(x=v["date"], y=v["actual"], name="Actual", line=dict(color="#1A1033", width=1.8))
            fig.add_scatter(x=v["date"], y=v["predicted"], name="Predicted", line=dict(color="#7C3AED", width=3))
            show(fig, 380, light=True)

    with t_pred:
        section("Predict for new rows", "Upload a file with the same columns as before. The target can be missing. "
                                         "Include future dates and your planned price, promotions and so on.")
        new = upload_box("New data", "new")
        if new:
            try:
                out = fc.predict_drivers(bundle, fc.load_table(new))
                st.dataframe(out.head(300), width="stretch", hide_index=True)
                st.download_button("Download predictions", out.to_csv(index=False), "predictions.csv", "text/csv")
            except Exception as e:
                st.error(str(e))
        st.download_button("Download trained model", fc.bundle_to_bytes(bundle), "demand_model.joblib")

    with t_mon:
        section("Compare predictions with actuals", "Upload rows where the real value has arrived, with the target "
                                                    "column included. The app checks for drift.")
        n = st.session_state.get("n_drv", 0)
        act_file = upload_box("Actual results", f"act_drv_{n}")
        if act_file:
            try:
                actuals = fc.load_table(act_file)
                cmp = fc.compare_driver_to_actuals(bundle, actuals)
                (st.error if cmp["drift"] else st.success)(cmp["message"])
                st.dataframe(cmp["summary"].round(3), width="stretch", hide_index=True)
                mt, tc, dc = cmp["matched"], bundle["meta"]["target_col"], bundle["meta"]["date_col"]
                daily = mt.groupby(dc)[[tc, "Forecast"]].sum().reset_index()
                fig = go.Figure()
                fig.add_scatter(x=daily[dc], y=daily[tc], name="Actual", line=dict(color="#F7F5FF", width=2))
                fig.add_scatter(x=daily[dc], y=daily["Forecast"], name="Predicted", line=dict(color="#8B5CF6", width=3))
                show(fig, 320)
                if st.button("Retrain with these actuals included"):
                    merged = pd.concat([df, actuals], ignore_index=True).drop_duplicates()
                    st.session_state["drv"] = fc.train_driver_model(merged, date_col, target_col, holiday_country)
                    st.session_state["n_drv"] = n + 1
                    st.rerun()
            except Exception as e:
                st.error(str(e))

    with t_data:
        st.dataframe(df.head(500), width="stretch", hide_index=True)


if mode == "Forecast the future":
    render_future()
else:
    render_drivers()
