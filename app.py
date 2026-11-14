"""
Demand Forecaster  -  run with:  python -m streamlit run app.py
Upload sales history -> forecast -> inventory plan -> monitoring.
"""
import copy
import html
import io
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import assistant as asst
import forecaster as fc
import intake
import inventory as inv
import news
import quality

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
.chip.nu{background:rgba(255,255,255,.08);color:var(--muted);}
.hrow{display:grid;grid-template-columns:210px 1fr auto;gap:1rem;align-items:center;padding:.65rem 1rem;margin-bottom:.35rem;
  background:rgba(255,255,255,.04);border-radius:8px;border-left:4px solid var(--muted);}
.hrow.ok{border-left-color:#34D399;}.hrow.warn{border-left-color:#FBBF24;}.hrow.note{border-left-color:#A89FC9;}
.hrow .hn{font-family:'Sora',sans-serif;font-weight:600;color:#fff;font-size:.92rem;}.hrow .hd{color:var(--muted);font-size:.88rem;line-height:1.4;}
.srow{padding:.15rem 0 .45rem;border-bottom:1px solid var(--line);margin-bottom:.4rem;}
.srow small{color:var(--muted);}
.nrow{padding:.75rem 1rem;margin-bottom:.4rem;background:rgba(255,255,255,.04);border-radius:8px;border-left:4px solid var(--muted);}
.nrow.hi{border-left-color:#FB7185;}.nrow.md{border-left-color:#FBBF24;}.nrow.lo{border-left-color:#34D399;}
.nrow a{color:#fff;font-weight:600;text-decoration:none;line-height:1.4;}
.nrow a:hover{color:var(--lilac);text-decoration:underline;}
.nrow .nm{display:flex;gap:.6rem;align-items:center;flex-wrap:wrap;margin-top:.35rem;color:var(--muted);font-size:.84rem;}
.tag{border:1px solid var(--line);color:var(--lilac);padding:.05rem .6rem;border-radius:999px;font-size:.76rem;}
[class*="st-key-po_"]{background:rgba(255,255,255,.04);border:1px solid var(--line);border-radius:12px;padding:1rem 1.2rem;margin-bottom:.8rem;}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important;}}
@media (max-width:900px){.how{grid-template-columns:1fr;}.hero h1{font-size:2.4rem;}.prow,.hrow{grid-template-columns:1fr 1fr;}}
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
    c = st.columns([1.1, 1.1, 2.6])
    c[0].button("Try with demo data", type="primary", width="stretch",
                on_click=lambda: st.session_state.update(demo=True))
    c[1].button("Try a messy file", width="stretch", on_click=lambda: st.session_state.update(messy=True, page="Prepare data"))
    c[2].caption("Or upload your own CSV or Excel file in the sidebar. It can be messy: odd date formats, currency "
                 "symbols, repeated headers. The assistant helps you clean it.")
    st.markdown("""
    <div class="how">
      <div><b>Forecast demand</b><span>Learns from past sales, price, promotions and holidays, then projects the weeks ahead.</span></div>
      <div><b>Plan your stock</b><span>Turns the forecast into safety stock, reorder points and order quantities.</span></div>
      <div><b>Stay accurate</b><span>Compares forecasts with real sales and tells you when the model needs retraining.</span></div>
    </div>""", unsafe_allow_html=True)


# ----------------------------------------------------------------- NEWS PAGE
@st.cache_data(ttl=900, show_spinner=False)
def load_news(region, days, themes, keywords, publishers):
    return news.collect(region, days, list(themes), list(keywords), publishers)


def ago(ts):
    secs = (pd.Timestamp.now("UTC").tz_localize(None) - pd.Timestamp(ts)).total_seconds()
    return f"{max(int(secs // 60), 1)}m ago" if secs < 3600 else (f"{int(secs // 3600)}h ago" if secs < 86400
                                                                  else f"{int(secs // 86400)}d ago")


def seg(label, options, key):
    k = f"{key}_{abs(hash(tuple(options))) % 10**6}"
    return st.segmented_control(label, options, default=options[0], key=k, label_visibility="collapsed") or options[0]


def news_page():
    with st.sidebar:
        st.markdown('<div class="side-h">News settings</div>', unsafe_allow_html=True)
        region = st.selectbox("Region", list(news.REGIONS))
        days = st.select_slider("Look back", options=[1, 3, 7, 14, 30], value=7,
                                format_func=lambda d: f"{d} day" + ("s" if d > 1 else ""))
        topics = st.multiselect("Topics", list(news.THEMES), default=list(news.THEMES))
        kw = st.text_input("Your own keywords", placeholder="rice exports, steel prices, Chennai port",
                           help="Separate with commas. Up to 5. Each one is searched on its own.")
        publishers = st.toggle("Also scan business news feeds", value=True)
        if st.button("Refresh now", width="stretch"):
            load_news.clear()
    keywords = tuple(k.strip() for k in kw.split(",") if k.strip())[:5]

    st.markdown('<div class="hero-sm"><h1>Market news</h1><p>Headlines from open sources that can move '
                'demand, imports and exports.</p></div>', unsafe_allow_html=True)
    if not topics and not keywords:
        st.info("Pick at least one topic or add a keyword in the sidebar.")
        return
    with st.spinner("Reading the news"):
        nd, errors = load_news(region, days, tuple(topics), keywords, publishers)
    if errors:
        st.caption("Could not reach: " + ", ".join(errors))
    if nd.empty:
        st.info("No headlines found. Try a longer look-back, more topics, or check your internet connection.")
        return

    top_topic = nd["topic"].value_counts().idxmax()
    kstrip([("Headlines", f"{len(nd):,}", f"last {days} day" + ("s" if days > 1 else "")),
            ("Risk signals", f"{(nd['signal'] == 'Risk').sum():,}", "delays, bans, shortages, falls"),
            ("Supportive signals", f"{(nd['signal'] == 'Supportive').sum():,}", "growth, easing, deals"),
            ("Most covered topic", top_topic, f"{(nd['topic'] == top_topic).sum()} headlines")])

    daily = nd.assign(day=nd["published"].dt.floor("D")).groupby(["day", "signal"]).size().unstack(fill_value=0)
    fig = go.Figure()
    for sig, colour in [("Risk", "#FB7185"), ("Mixed", "#FBBF24"), ("Neutral", "rgba(247,245,255,.35)"),
                        ("Supportive", "#34D399")]:
        if sig in daily:
            fig.add_bar(x=daily.index, y=daily[sig], name=sig, marker_color=colour)
    fig.update_layout(barmode="stack")
    show(fig, 260)

    c1, c2 = st.columns([3, 2])
    with c1:
        topic_pick = seg("Topic", ["All"] + sorted(nd["topic"].unique()), "ntopic")
    with c2:
        sig_pick = seg("Signal", ["All", "Risk", "Supportive", "Mixed", "Neutral"], "nsig")
    q = st.text_input("Search these headlines", placeholder="Type a word, for example rice or port")
    view = nd
    if topic_pick != "All":
        view = view[view["topic"] == topic_pick]
    if sig_pick != "All":
        view = view[view["signal"] == sig_pick]
    if q:
        view = view[view["title"].str.contains(q, case=False, regex=False)]

    cls = {"Risk": "hi", "Mixed": "md", "Supportive": "lo", "Neutral": "nu"}
    rows = ""
    for _, r in view.head(60).iterrows():
        link = r["link"] if str(r["link"]).startswith("http") else "#"
        rows += (f'<div class="nrow {cls[r["signal"]] if r["signal"] != "Neutral" else ""}">'
                 f'<a href="{html.escape(link)}" target="_blank" rel="noopener noreferrer">{html.escape(r["title"])}</a>'
                 f'<div class="nm"><span class="chip {cls[r["signal"]]}">{r["signal"]}</span>'
                 f'<span class="tag">{html.escape(r["topic"])}</span>'
                 f'<span>{html.escape(r["source"])}</span><span>{ago(r["published"])}</span></div></div>')
    st.markdown(rows or "<p>No headlines match these filters.</p>", unsafe_allow_html=True)
    if len(view) > 60:
        st.caption(f"Showing the newest 60 of {len(view)}. Use the filters to narrow down.")
    st.download_button("Download headlines", view.to_csv(index=False), "market_news.csv", "text/csv")
    st.caption("Sources: Google News search feeds and public business RSS feeds. Signals come from simple keyword "
               "rules on the headline, so treat them as a quick guide and open the article before acting on it.")


# ------------------------------------------------------------- DATA HELPERS
def get_raw(up):
    """Read the current data source. Returns (table, notes, signature) or (None, None, None)."""
    ss = st.session_state
    try:
        if up is not None:
            sheet = ss.get(f"sheet_{up.name}")
            key = ("file", up.name, up.size, sheet)
            if ss.get("raw_key") != key:
                ss["raw"] = intake.load_smart(up, sheet)
                ss["raw_key"] = key
            raw, notes = ss["raw"]
            if len(notes.get("sheets", [])) > 1:
                with st.sidebar:
                    st.selectbox("Sheet", notes["sheets"], index=notes["sheets"].index(notes["sheet"]), key=f"sheet_{up.name}")
            return raw, notes, f"{up.name}|{up.size}|{notes.get('sheet')}"
        if ss.get("messy"):
            if ss.get("raw_key") != ("messy",):
                data, _ = intake.messy_demo()
                buf = io.BytesIO(data)
                buf.name = "messy_sales_sample.csv"
                ss["raw"], ss["raw_key"] = intake.load_smart(buf), ("messy",)
            return ss["raw"][0], ss["raw"][1], "messy"
        if ss.get("demo"):
            return fc.demo_data(), {"name": "demo data", "kind": "demo", "messages": [], "sheets": []}, "demo"
    except Exception as e:
        st.error(f"Could not read this file: {e}")
    return None, None, None


def triage(raw, sig):
    ss = st.session_state
    if ss.get("triage_sig") != sig:
        ss["triage"], ss["triage_sig"] = intake.analyze(raw), sig
    return ss["triage"]


def health_panel(frame, dcol, tcol, gcol=None):
    try:
        items = quality.data_health(frame, dcol, tcol, gcol)
    except Exception as e:
        st.warning(f"Could not run the health check: {e}")
        return
    n_ok = sum(i["status"] == "ok" for i in items)
    kstrip([("Checks passed", f"{n_ok} of {len(items)}", ""), ("Need a look", str(sum(i["status"] == "warn" for i in items)), "warnings"),
            ("Good to know", str(sum(i["status"] == "note" for i in items)), "notes")])
    chip, cls = {"ok": "Fine", "note": "Note", "warn": "Check"}, {"ok": "lo", "note": "nu", "warn": "md"}
    st.markdown("".join(
        f'<div class="hrow {i["status"]}"><span class="hn">{html.escape(i["check"])}</span>'
        f'<span class="hd">{html.escape(i["detail"])}</span><span class="chip {cls[i["status"]]}">{chip[i["status"]]}</span></div>'
        for i in items), unsafe_allow_html=True)


def prepare_empty():
    st.markdown('<div class="hero-sm"><h1>Prepare data</h1><p>Upload a file in the sidebar, or try the sample.</p></div>',
                unsafe_allow_html=True)
    c = st.columns([1.2, 1.2, 2.6])
    c[0].button("Load a messy sample", type="primary", width="stretch", on_click=lambda: st.session_state.update(messy=True))
    c[1].button("Use demo data", width="stretch", on_click=lambda: st.session_state.update(demo=True))
    st.caption("The messy sample has odd date formats, rupee symbols, lakh amounts, mixed promo labels, repeated headers "
               "and a total row, so you can see the cleaner and the assistant at work.")


# ------------------------------------------------------------ PREPARE PAGE
def _clean_recipe(recipe):
    keep = ("id", "op", "enabled", "column", "dayfirst", "decimal", "factor", "custom", "operator", "value", "reason",
            "method", "cap", "new_name", "mapping", "id_cols")
    return [{k: v for k, v in s.items() if k in keep} for s in recipe]


def _use_data(sig, clean, aside, log, key):
    st.session_state["prepared"] = {"sig": sig, "df": clean, "aside": aside, "log": log, "key": key}
    st.session_state["page"] = "Forecasting"


def _answer(qid, idx):
    S = st.session_state["intake"]
    q = next(q for q in S["questions"] if q["id"] == qid)
    S["recipe"], msg = intake.answer_question(S["recipe"], q, idx)
    S["answered"].append(qid)
    S["chat"].append({"role": "assistant", "content": msg})


def prepare_page(raw, notes, sig):
    ss = st.session_state
    S = ss.get("intake")
    if not S or S["sig"] != sig:
        an = copy.deepcopy(triage(raw, sig))
        if an["clean"]:
            hello = ("Your file looks clean, so nothing needs fixing. You can go straight to Forecasting. "
                     "If something looks wrong to you, tell me and I will adjust it.")
        else:
            hello = (f"I read {len(raw):,} rows and {len(raw.columns)} columns. I added {an['fixes']} cleaning step(s) "
                     f"on my own and have {len([q for q in an['questions'] if not q['optional']])} question(s) for you. "
                     "You can also just tell me about your data, for example *dates are day first* or *ignore Cancelled orders*. "
                     "Nothing is deleted: rows I cannot use are set aside and listed.")
        S = ss["intake"] = {"sig": sig, "recipe": an["steps"], "questions": an["questions"], "answered": [],
                            "chat": [{"role": "assistant", "content": hello}], "roles": {}, "applied_key": None}

    # run the recipe on a copy of the raw table
    key = json.dumps(S["recipe"], sort_keys=True, default=str)
    if S["applied_key"] != key:
        S["applied"], S["applied_key"] = intake.apply_recipe(raw, S["recipe"]), key
    clean, aside, log = S["applied"]

    st.markdown('<div class="hero-sm"><h1>Prepare data</h1><p>Fix messy files without losing rows. Explain your data to the '
                'assistant and it follows your instructions.</p></div>', unsafe_allow_html=True)
    read_bits = [f"{notes.get('kind', 'file')}"]
    if notes.get("delimiter"):
        read_bits.append("separator '" + ("tab" if notes["delimiter"] == "\t" else notes["delimiter"]) + "'")
    if notes.get("encoding"):
        read_bits.append(notes["encoding"])
    if notes.get("sheet"):
        read_bits.append(f"sheet '{notes['sheet']}'")
    st.caption("Read as " + ", ".join(read_bits) + ". " + " ".join(notes.get("messages", [])))
    kstrip([("Rows read", f"{len(raw):,}", f"{len(raw.columns)} columns"), ("Rows after cleaning", f"{len(clean):,}", f"{len(S['recipe'])} steps"),
            ("Set aside", f"{len(aside):,}", "listed under Needs attention")])

    left, right = st.columns([5, 6], gap="large")
    pending = [q for q in S["questions"] if q["id"] not in S["answered"] and not q["optional"]]
    optional = [q for q in S["questions"] if q["id"] not in S["answered"] and q["optional"]]

    # ---------------- chat
    with left:
        section("Assistant", "Ask, explain or correct. I will ask when I am unsure.")
        with st.expander("Understand any wording with a language model (optional)"):
            api_key = st.text_input("Anthropic API key", type="password", key="prep_key")
            model = st.text_input("Model", value="claude-sonnet-5-5", key="prep_model")
            st.caption("Without a key, I follow the common phrasings listed when you type *help*. With a key, a language model "
                       "turns your own words into the same safe cleaning steps. Only column names and a few sample values are sent.")
            if st.checkbox("Show what would be sent", key="prep_show"):
                st.code(asst.build_prompt("<your message>", clean, S["recipe"])[:2500], language=None)
        llm = (api_key, model) if api_key else None
        box = st.container(height=440)
        with box:
            for m in S["chat"]:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])
            for q in pending + optional[:2]:
                with st.chat_message("assistant"):
                    st.markdown(("**Suggestion:** " if q["optional"] else "**Question:** ") + q["text"])
                    bc = st.columns(len(q["options"]))
                    for i, o in enumerate(q["options"]):
                        bc[i].button(o["label"], key=f"ans_{q['id']}_{i}", on_click=_answer, args=(q["id"], i), width="stretch")
        text = st.chat_input("Tell me about your data, or type help")
        if text:
            S["chat"].append({"role": "user", "content": text})
            r = asst.respond(text, clean, S["recipe"], llm)
            for stp in r["steps"]:
                S["recipe"], _ = intake.upsert_step(S["recipe"], stp)
            if r["undo"] == "last" and S["recipe"]:
                S["recipe"].pop()
            elif isinstance(r["undo"], int) and 0 < r["undo"] <= len(S["recipe"]):
                S["recipe"].pop(r["undo"] - 1)
            S["roles"].update(r["roles"])
            if r["question"]:
                S["questions"].append(r["question"])
            S["chat"].append({"role": "assistant", "content": r["reply"] or "I did not catch that. Type *help* to see what I can do."})
            st.rerun()

    # ---------------- recipe, preview, set aside, health
    with right:
        t_rec, t_prev, t_aside, t_health = st.tabs(["Cleaning recipe", "Preview", "Needs attention", "Data health"])
        with t_rec:
            if not S["recipe"]:
                st.caption("No steps. The file is used as it is.")
            by_id = {e["step_id"]: e for e in log}
            for i, stp in enumerate(S["recipe"]):
                c1, c2, c3 = st.columns([0.6, 7, 1.4])
                stp["enabled"] = c1.checkbox("on", value=stp.get("enabled", True), key=f"en_{stp['id']}", label_visibility="collapsed")
                e = by_id.get(stp["id"])
                line = f"**{i + 1}. {intake.step_title(stp)}**"
                if stp.get("auto"):
                    line += "  <small>(added automatically)</small>"
                c2.markdown(line, unsafe_allow_html=True)
                if e:
                    detail = ([f"{e['changed']:,} value(s) changed"] if e["changed"] else []) + e["notes"][:3]
                    if detail:
                        c2.caption(" ".join(detail))
                    if e["examples"]:
                        c2.caption("Example: " + "; ".join(f"{a} → {b}" for a, b in e["examples"][:3]))
                if c3.button("Remove", key=f"rm_{stp['id']}"):
                    S["recipe"].remove(stp)
                    st.rerun()
            with st.expander("Add a step yourself"):
                labels = {"Read dates": "parse_dates", "Read numbers or money": "parse_number", "Multiply by a factor": "scale",
                          "Merge spellings": "standardise_text", "Set rows aside": "set_aside", "Fill missing values": "fill_missing",
                          "Treat zeros as missing": "zero_as_missing", "Flag or cap outliers": "flag_outliers",
                          "Rename a column": "rename", "Leave out a column": "ignore_column"}
                kind = labels[st.selectbox("What do you want to do?", list(labels), key="man_op")]
                col = st.selectbox("Column", list(clean.columns) + [c for c in raw.columns if c not in clean.columns], key="man_col")
                step = {"op": kind, "column": col}
                if kind == "parse_dates":
                    step["dayfirst"] = {"Detect automatically": None, "Day first": True, "Month first": False}[
                        st.selectbox("Date order", ["Detect automatically", "Day first", "Month first"], key="man_df")]
                elif kind == "parse_number":
                    step["decimal"] = st.selectbox("Decimal mark", ["auto", "dot", "comma"], key="man_dec")
                elif kind == "scale":
                    step["factor"] = st.number_input("Multiply by", value=1.0, key="man_fac")
                elif kind == "set_aside":
                    step["operator"] = st.selectbox("When the value", ["equals", "not_equals", "contains", "greater_than", "less_than", "is_empty"], key="man_opr")
                    step["value"] = st.text_input("This value", key="man_val")
                elif kind == "fill_missing":
                    step["method"] = st.selectbox("Fill with", ["zero", "median", "mean", "ffill", "interpolate"], key="man_met")
                elif kind == "flag_outliers":
                    step["cap"] = st.checkbox("Also cap them", key="man_cap")
                elif kind == "rename":
                    step["new_name"] = st.text_input("New name", key="man_new")
                if st.button("Add this step", key="man_add"):
                    good, msg = intake.validate_step(step, list(clean.columns) + list(raw.columns))
                    if good:
                        S["recipe"], _ = intake.upsert_step(S["recipe"], good)
                        S["chat"].append({"role": "assistant", "content": f"Added: {intake.step_title(good)}."})
                        st.rerun()
                    else:
                        st.warning(f"Could not add that step: {msg}.")
            c1, c2 = st.columns(2)
            c1.download_button("Save this recipe", json.dumps(_clean_recipe(S["recipe"]), indent=1, default=str),
                               "cleaning_recipe.json", "application/json", width="stretch")
            saved = c2.file_uploader("Load a saved recipe", type=["json"], key="recipe_up", label_visibility="collapsed")
            if saved:
                try:
                    steps_in = json.load(saved)
                    good = [g for g in (intake.validate_step(x, list(raw.columns))[0] for x in steps_in) if g]
                    if good and json.dumps(good, sort_keys=True) != ss.get("recipe_loaded"):
                        ss["recipe_loaded"] = json.dumps(good, sort_keys=True)
                        S["recipe"] = [{**intake.new_step(g["op"]), **g, "auto": False} for g in good]
                        S["chat"].append({"role": "assistant", "content": f"Loaded a saved recipe with {len(good)} step(s)."})
                        st.rerun()
                except Exception as e:
                    st.warning(f"Could not read that recipe: {e}")

        with t_prev:
            section("Cleaned table", "This is what the forecast will use.")
            st.dataframe(clean.head(60), width="stretch")
            common = [c for c in raw.columns if c in clean.columns]
            if common:
                pick = st.selectbox("Compare one column before and after", common, key="cmp_col")
                head = raw[pick].head(12)
                st.dataframe(pd.DataFrame({"Before": head.astype(str), "After": clean[pick].reindex(head.index).astype(str)}),
                             width="stretch")

        with t_aside:
            if len(aside):
                section("Rows set aside", "Not deleted. They are left out of the forecast because they cannot be used as they are.")
                st.dataframe(aside.groupby("reason").size().rename("Rows").reset_index().rename(columns={"reason": "Reason"}),
                             width="stretch", hide_index=True)
                st.dataframe(aside.astype(str), width="stretch")
                st.download_button("Download these rows", aside.to_csv(), "rows_needing_attention.csv", "text/csv")
            else:
                st.caption("Nothing has been set aside. Every row is used.")

        with t_health:
            cc = list(clean.columns)
            d_guess = S["roles"].get("date") or fc.guess_date_col(clean)
            numc = [c for c in cc if pd.api.types.is_numeric_dtype(clean[c])]
            t_guess = S["roles"].get("target") or fc.guess_target_col(clean, d_guess)
            a, b = st.columns(2)
            dcol = a.selectbox("Date column", cc, index=cc.index(d_guess) if d_guess in cc else 0, key="h_date")
            tcol = b.selectbox("Target column", numc or cc, index=(numc or cc).index(t_guess) if t_guess in (numc or cc) else 0, key="h_tgt")
            health_panel(clean, dcol, tcol)

    st.divider()
    a, b = st.columns([2, 3])
    a.button("Use this data and go to Forecasting", type="primary", width="stretch",
             on_click=_use_data, args=(sig, clean, aside, log, key))
    prep = ss.get("prepared")
    if prep and prep["sig"] == sig:
        b.caption("This data is in use for forecasting." if prep["key"] == key else
                  "You changed the recipe after applying it. Select the button again to update the forecast data.")
    else:
        b.caption("Skip this page if your file already looks clean. The forecast then uses it as it is.")


# --------------------------------------------------------- 2. SIDEBAR (data)
st.session_state.setdefault("page", "Forecasting")
with st.sidebar:
    st.markdown('<div class="brand"><i></i>Demand Forecaster</div>', unsafe_allow_html=True)
    page = st.segmented_control("View", ["Prepare data", "Forecasting", "Market news"], key="page",
                                label_visibility="collapsed") or "Forecasting"
if page == "Market news":
    news_page()
    st.stop()
with st.sidebar:
    st.markdown('<div class="side-h">Your data</div>', unsafe_allow_html=True)
    up = st.file_uploader("Sales history", type=["csv", "xlsx", "xlsm", "xls"], label_visibility="collapsed")
    st.toggle("Use demo data", key="demo")
    st.toggle("Use messy sample file", key="messy")

raw, load_notes, sig = get_raw(up)
if raw is None:
    prepare_empty() if page == "Prepare data" else landing()
    st.stop()
if page == "Prepare data":
    prepare_page(raw, load_notes, sig)
    st.stop()

# Lane 1 (clean data) uses the file as it is. Lane 2 uses the cleaned table from Prepare data.
prep = st.session_state.get("prepared")
using_prepared = bool(prep and prep["sig"] == sig)
df = prep["df"] if using_prepared else raw
triage_result = triage(raw, sig)

# --------------------------------------------------------- 3. SIDEBAR (setup)
cols = list(df.columns)
numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
_S = st.session_state.get("intake") or {}
roles = _S.get("roles", {}) if _S.get("sig") == sig else {}
g_date = roles["date"] if roles.get("date") in cols else fc.guess_date_col(df)
g_target = roles["target"] if roles.get("target") in numeric else fc.guess_target_col(df, g_date)
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
data_state = ("Cleaned data in use" if using_prepared else "Data looks clean" if triage_result["clean"] else "Needs cleaning")
st.markdown(f"""<div class="hero-sm"><h1>Demand Forecaster</h1>
<p>Forecast, plan stock, and check accuracy.</p></div>
<div class="facts"><span class="fact">{len(df):,} rows</span><span class="fact">{len(df.columns)} columns</span>
<span class="fact">{html.escape(span)}</span><span class="fact">{data_state}</span></div>""", unsafe_allow_html=True)
if not using_prepared and not triage_result["clean"]:
    _nq = len([q for q in triage_result["questions"] if not q["optional"]])
    st.warning(f"This file needs cleaning before it can be forecast reliably: {triage_result['fixes']} fix(es) and {_nq} "
               "question(s) found. Forecasting it as it is may misread or skip rows.")
    st.button("Open Prepare data", on_click=lambda: st.session_state.update(page="Prepare data"))
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

    t_fc, t_acc, t_inv, t_mon, t_health, t_data = st.tabs(["Forecast", "Accuracy", "Inventory", "Monitor", "Data health", "Data"])

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
                 "repeat last season"),
                ("Bias in testing", f"{mrow.get('Bias %', np.nan):+.1%}" if pd.notna(mrow.get("Bias %", np.nan)) else "n/a",
                 quality.bias_status(mrow.get("Bias %", np.nan))[1].split(", ", 1)[-1])])
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
        seg = quality.segment_products(hdf, met)
        section("Which products can you trust?",
                "Class: A = products making up the first 80% of volume, B = the next 15%, C = the rest. X = steady demand, "
                "Y = moderate swings, Z = erratic. Reliable means it beats the baseline, bias is within 10% and the typical "
                "error is under 30%.")
        tint = {"Reliable": "rgba(52,211,153,.22)", "Check": "rgba(251,191,36,.22)", "Weak": "rgba(251,113,133,.28)"}
        sty = seg.style.format({"Share of volume": "{:.0%}", "Variability (CV)": "{:.2f}", "Model WAPE": "{:.1%}",
                                "Baseline WAPE": "{:.1%}", "Bias %": "{:+.1%}"}, na_rep="n/a")
        sty = getattr(sty, "map", None) and sty.map(lambda v: f"background-color: {tint.get(v, 'transparent')}", subset=["Trust"]) or \
            sty.applymap(lambda v: f"background-color: {tint.get(v, 'transparent')}", subset=["Trust"])
        st.dataframe(sty, width="stretch", hide_index=True)
        bias_df = seg.dropna(subset=["Bias %"])
        if len(bias_df):
            fig = go.Figure(go.Bar(x=bias_df["Group"], y=bias_df["Bias %"],
                                   marker_color=["#FB7185" if v > 0.10 else "#FBBF24" if abs(v) > 0.05 else "#34D399" if v >= -0.05
                                                 else "#FBBF24" if v >= -0.10 else "#FB7185" for v in bias_df["Bias %"]]))
            fig.update_layout(yaxis_tickformat=".0%", title="Bias in testing (above zero = forecasts too high)")
            show(fig, 260)
        with st.expander("Full test results"):
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

    with t_health:
        section("Data health", "Checks on the data behind this forecast.")
        health_panel(df, p["date_col"], p["target_col"], p["group_col"])

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
            ("Bias", f"{best['Bias %']:+.1%}", quality.bias_status(best["Bias %"])[1].split(", ", 1)[-1]),
            ("Trained on", f"{bundle['train_rows']:,} rows", f"tested on {bundle['valid_rows']:,}")])
    t_mod, t_pred, t_mon, t_health, t_data = st.tabs(["Models", "Predict", "Monitor", "Data health", "Data"])

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

    with t_health:
        section("Data health", "Checks on the data behind these models.")
        health_panel(df, date_col, target_col)

    with t_data:
        st.dataframe(df.head(500), width="stretch", hide_index=True)


if mode == "Forecast the future":
    render_future()
else:
    render_drivers()
