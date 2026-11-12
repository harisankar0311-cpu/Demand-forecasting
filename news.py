"""
news.py - headlines from open sources that can affect demand, imports and exports.

Sources (free, no API key):
  * Google News search feeds (RSS), one search per topic and per keyword of yours
  * A few public business RSS feeds from publishers

Every headline gets
  * a topic       (trade policy, logistics, economy and demand, ...)
  * a signal      (risk, supportive, mixed, neutral) from simple keyword rules on the headline

The signal is a rough aid for skimming, not a prediction. Always open the article.
"""
import html
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote_plus

import pandas as pd
import requests

REGIONS = {  # name: (gl, hl, ceid)
    "India": ("IN", "en-IN", "IN:en"), "United States": ("US", "en-US", "US:en"),
    "United Kingdom": ("GB", "en-GB", "GB:en"), "Canada": ("CA", "en-CA", "CA:en"),
    "Australia": ("AU", "en-AU", "AU:en"), "Singapore": ("SG", "en-SG", "SG:en"),
}

THEMES = {  # topic: Google News search
    "Trade policy": '"tariffs" OR "export ban" OR "import duty" OR "trade deal" OR sanctions OR "export restrictions"',
    "Imports and exports": '"trade deficit" OR "export orders" OR "import volumes" OR "export growth" OR "trade data"',
    "Logistics": '"supply chain" OR "port congestion" OR "freight rates" OR "container shipping" OR "Red Sea" OR "shipping delays"',
    "Economy and demand": '"consumer spending" OR "retail sales" OR "consumer demand" OR "consumer confidence" OR inflation OR "interest rates"',
    "Costs and commodities": '"oil prices" OR "commodity prices" OR "raw material prices" OR "fuel prices" OR "exchange rate"',
    "Weather and disruption": '"extreme weather" OR monsoon OR flood OR heatwave OR "labour strike" OR "factory shutdown"',
    "Seasons and events": '"festive season" OR "holiday shopping" OR "Black Friday" OR "Singles Day"',
}

PUBLISHER_FEEDS = {
    "BBC Business": "https://feeds.bbci.co.uk/news/business/rss.xml",
    "The Guardian Business": "https://www.theguardian.com/uk/business/rss",
    "Al Jazeera": "https://www.aljazeera.com/xml/rss/all.xml",
    "Economic Times": "https://economictimes.indiatimes.com/rssfeedstopstories.cms",
}

_THEME_WORDS = {  # used to sort headlines from the publisher feeds into topics
    "Trade policy": r"tariffs?|duty|duties|export bans?|import bans?|sanctions?|embargo|trade deals?|trade war|quotas?|anti-dumping|customs",
    "Imports and exports": r"imports?|exports?|exporters?|importers?|trade deficit|trade surplus|trade data",
    "Logistics": r"supply chains?|ports?|freight|containers?|shipping|logistics|red sea|suez|panama canal|warehous\w+",
    "Economy and demand": r"consumer spending|retail sales|demand|inflation|interest rates?|rate cuts?|rate hikes?|gdp|recession|consumer confidence|unemployment",
    "Costs and commodities": r"oil|crude|fuel|diesel|commodit\w+|raw materials?|steel|fertili[sz]er|currency|rupee|exchange rate",
    "Weather and disruption": r"weather|monsoon|floods?|cyclone|heatwave|drought|hurricane|storm|strikes?|shutdown|earthquake|wildfire",
    "Seasons and events": r"festivals?|festive|holiday season|black friday|diwali|christmas|eid|singles day",
}
_THEME_RE = {k: re.compile(r"\b(?:" + v + r")\b", re.I) for k, v in _THEME_WORDS.items()}

_RISK = re.compile(r"\b(?:shortages?|delays?|delayed|disruptions?|disrupt\w*|bans?|banned|strikes?|slump\w*|slowdown|"
                   r"declines?|declined|falls?|fell|drops?|dropped|plunge\w*|crisis|hikes?|soar\w*|recession|tariffs?|"
                   r"sanctions?|embargo|congestion|blockade|war|shutdown|halts?|inflation|warns?|worst|fears?)\b", re.I)
_SUPPORT = re.compile(r"\b(?:growth|grows?|grew|rises?|rose|boosts?|record|recovery|recovers?|rebounds?|eases?|easing|"
                      r"rate cuts?|strong demand|expands?|expansion|upbeat|optimism|improves?|improved|gains?|"
                      r"jumps?|deals?|agreement|relief|lifts?|lifted)\b", re.I)


# ------------------------------------------------------------------ fetching
def http_get(url, timeout=8):
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 (DemandForecaster)"})
    resp.raise_for_status()
    return resp.content


def google_news_url(query, region, days):
    gl, hl, ceid = REGIONS.get(region, REGIONS["India"])
    q = quote_plus(f"{query} when:{int(days)}d")
    return f"https://news.google.com/rss/search?q={q}&hl={hl}&gl={gl}&ceid={ceid}"


# ------------------------------------------------------------------- parsing
def _text(el):
    return (el.text or "").strip() if el is not None else ""


def _strip_tags(s):
    return html.unescape(re.sub(r"<[^>]+>", " ", s or "")).strip()


def _date(s):
    ts = pd.to_datetime(s, utc=True, errors="coerce") if s else pd.NaT
    return None if pd.isna(ts) else ts.tz_localize(None).to_pydatetime()


def parse_feed(content):
    """Parse RSS 2.0 or Atom bytes into a list of dicts: title, link, published, source, summary."""
    root = ET.fromstring(content)
    out = []
    for it in root.iter("item"):
        out.append({"title": html.unescape(_text(it.find("title"))), "link": _text(it.find("link")),
                    "published": _date(_text(it.find("pubDate")) or _text(it.find("{http://purl.org/dc/elements/1.1/}date"))),
                    "source": _text(it.find("source")), "summary": _strip_tags(_text(it.find("description")))})
    atom = "{http://www.w3.org/2005/Atom}"
    for en in root.iter(atom + "entry"):
        link = en.find(atom + "link")
        out.append({"title": html.unescape(_text(en.find(atom + "title"))),
                    "link": link.get("href", "") if link is not None else "",
                    "published": _date(_text(en.find(atom + "updated")) or _text(en.find(atom + "published"))),
                    "source": "", "summary": _strip_tags(_text(en.find(atom + "summary")))})
    return out


def signal_for(text):
    risk, support = bool(_RISK.search(text)), bool(_SUPPORT.search(text))
    return "Mixed" if risk and support else "Risk" if risk else "Supportive" if support else "Neutral"


def theme_for(text):
    for name, rx in _THEME_RE.items():
        if rx.search(text):
            return name
    return None


# ---------------------------------------------------------------- collecting
def collect(region="India", days=7, themes=None, keywords=(), publishers=True, per_feed=25):
    """Fetch everything in parallel. Returns (DataFrame, list of sources that failed)."""
    themes = list(THEMES) if themes is None else [t for t in themes if t in THEMES]
    jobs = [(t, google_news_url(THEMES[t], region, days), t, "Google News") for t in themes]
    jobs += [(f'"{k}"', google_news_url(f'"{k}"', region, days), "Your keywords", "Google News") for k in keywords]
    if publishers:
        jobs += [(name, url, None, name) for name, url in PUBLISHER_FEEDS.items()]

    def run(job):
        label, url, hint, origin = job
        try:
            return label, hint, origin, parse_feed(http_get(url))[:per_feed], None
        except Exception as e:  # one broken source must not break the page
            return label, hint, origin, [], f"{label} ({type(e).__name__})"

    rows, errors = [], []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for label, hint, origin, items, err in ex.map(run, jobs):
            if err:
                errors.append(err)
            for it in items:
                title, src = it["title"], it["source"] or origin
                if it["source"] and title.endswith(" - " + it["source"]):
                    title = title[: -(len(it["source"]) + 3)]
                topic = hint or theme_for(title + " " + it["summary"])
                if not title or not topic or it["published"] is None:
                    continue  # publisher items that match no topic are dropped
                rows.append({"published": it["published"], "title": title, "source": src, "link": it["link"],
                             "topic": topic, "signal": signal_for(title)})
    cols = ["published", "title", "source", "link", "topic", "signal"]
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df, errors
    cutoff = pd.Timestamp.now("UTC").tz_localize(None) - pd.Timedelta(days=days)
    df = df[df["published"] >= cutoff]
    df["_k"] = df["title"].str.lower().str.replace(r"\W+", " ", regex=True).str.strip().str[:90]
    df = (df.sort_values("published", ascending=False).drop_duplicates("_k").drop(columns="_k")
          .reset_index(drop=True))
    return df, errors
