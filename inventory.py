"""
inventory.py - turns forecasts into inventory decisions.

  plan_inventory          safety stock, reorder point, order-up-to level, order quantity
  detect_spikes           flags unusual jumps/drops in recent history
  draft_purchase_orders   rule-based planner agent: drafts justified purchase orders and alerts
  llm_polish              OPTIONAL: rewrites a draft with Claude (needs an API key)

Method (periodic-review, order-up-to policy):
  lead-time demand  = forecast demand summed over the supplier lead time
  safety stock      = z(service level) x forecast error SD x sqrt(lead time + review period)
  reorder point     = lead-time demand + safety stock
  order-up-to level = forecast demand over (lead time + review period) + safety stock
  If inventory position (on hand + on order) <= reorder point, order up to the target level.
"""
from statistics import NormalDist

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = {"days": 365, "weeks": 52, "months": 12}
SPIKE_COLS = ["Group", "Date", "Value", "Expected", "Z-score", "Type"]


def z_for(service_level):
    return NormalDist().inv_cdf(min(max(float(service_level), 0.5), 0.9999))


def _cum(vals, n):
    """Sum of the first n forecast periods (extends with the recent average if n > len)."""
    vals = list(vals)
    if n <= len(vals):
        return float(sum(vals[:n]))
    tail = float(np.mean(vals[-min(7, len(vals)):]))
    return float(sum(vals) + tail * (n - len(vals)))


def plan_inventory(forecast, metrics, history, params, unit, service_level=0.95,
                   order_cost=50.0, holding_rate=0.20, review=1):
    """One row per group. `params` needs columns: group, on_hand, on_order, lead_time, unit_cost, moq."""
    z, ppy, R = z_for(service_level), PERIODS_PER_YEAR.get(unit, 365), max(int(review), 1)
    rows = []
    for _, p in params.iterrows():
        g = str(p["group"])
        f = forecast.loc[forecast["group"] == g].sort_values("date")["forecast"].to_numpy(float)
        if len(f) == 0:
            continue
        h = history.loc[history["group"] == g].sort_values("date")["value"].to_numpy(float)
        m = metrics.loc[metrics["Group"] == g]
        sigma = (float(m["Error SD"].iloc[0]) if "Error SD" in m and len(m) and pd.notna(m["Error SD"].iloc[0])
                 else (float(np.std(np.diff(h))) * 0.7 if len(h) > 2 else 0.0))

        L = max(int(p["lead_time"]), 1)
        on_hand, on_order = float(p["on_hand"]), float(p["on_order"])
        unit_cost, moq = float(p["unit_cost"]), max(float(p["moq"]), 1.0)
        mean_f = float(f.mean())

        lead_demand = _cum(f, L)
        safety = z * sigma * np.sqrt(L + R)
        rop = lead_demand + safety
        target = _cum(f, L + R) + safety
        position = on_hand + on_order
        order_now = position <= rop
        qty = max(target - position, 0.0) if order_now else 0.0
        qty = float(np.ceil(qty / moq) * moq) if qty > 0 else 0.0

        hold_cost = unit_cost * holding_rate
        eoq = (float(np.sqrt(2 * mean_f * ppy * order_cost / hold_cost))
               if hold_cost > 0 and mean_f > 0 else float("nan"))
        recent = float(np.mean(h[-L:])) if len(h) else float("nan")
        outlook = (float(f[:L].mean()) / recent - 1) if recent and recent > 0 else float("nan")
        excess = max(position - target, 0.0)

        rows.append({
            "Group": g, "Avg forecast / period": mean_f, "Lead-time demand": lead_demand,
            "Safety stock": safety, "Reorder point": rop, "Order-up-to level": target,
            "On hand": on_hand, "Inventory position": position,
            "Order now?": bool(order_now), "Order qty": qty, "Order value": qty * unit_cost,
            "Cover (periods)": on_hand / mean_f if mean_f > 0 else float("inf"),
            "Stockout risk": "High" if on_hand < lead_demand else ("Medium" if order_now else "Low"),
            "Excess units": excess, "Excess value": excess * unit_cost,
            "EOQ (reference)": eoq, "Demand outlook %": outlook,
        })
    return pd.DataFrame(rows)


def detect_spikes(history, window=28, z=3.0, last=14):
    """Flag unusual values among the last `last` periods vs the trailing window."""
    out = []
    for g, h in history.groupby("group"):
        s = h.sort_values("date").set_index("date")["value"]
        w = int(max(min(window, len(s) // 3), 5))
        mu = s.shift(1).rolling(w, min_periods=w // 2).mean()
        sd = s.shift(1).rolling(w, min_periods=w // 2).std().replace(0, np.nan)
        zs = (s - mu) / sd
        for dt in s.index[-last:]:
            zv = zs.get(dt, np.nan)
            if pd.notna(zv) and abs(zv) > z:
                out.append({"Group": g, "Date": dt, "Value": float(s[dt]), "Expected": float(mu[dt]),
                            "Z-score": float(zv), "Type": "Spike" if zv > 0 else "Drop"})
    return pd.DataFrame(out, columns=SPIKE_COLS)


def draft_purchase_orders(plan, spikes, unit, service_level, surge=0.25):
    """Rule-based planner agent. Returns a list of {group, kind, text}. A human approves each one."""
    short = unit.rstrip("s")
    drafts = []
    for _, r in plan.iterrows():
        g = r["Group"]
        flags = []
        sp = spikes[spikes["Group"] == g] if len(spikes) else spikes
        if len(sp):
            last = sp.sort_values("Date").iloc[-1]
            flags.append(f"{last['Type'].lower()} on {pd.Timestamp(last['Date']).date()}: "
                         f"{last['Value']:,.0f} vs about {last['Expected']:,.0f} expected")
        o = r["Demand outlook %"]
        if pd.notna(o) and abs(o) >= surge:
            flags.append(f"forecast demand over the lead time is {o:+.0%} vs the recent {short} average")
        if r["Order now?"]:
            lines = [
                f"PURCHASE ORDER DRAFT - {g}",
                f"Recommended order: {r['Order qty']:,.0f} units (estimated value {r['Order value']:,.0f})",
                "Why:",
                f"  - Inventory position ({r['Inventory position']:,.0f}) is at or below the reorder point "
                f"({r['Reorder point']:,.0f}).",
                f"  - Forecast demand over the lead time is {r['Lead-time demand']:,.0f} units; safety stock of "
                f"{r['Safety stock']:,.0f} covers a {service_level:.0%} service level.",
                f"  - Ordering up to {r['Order-up-to level']:,.0f} units covers lead time plus the review period.",
                f"  - Stockout risk without ordering: {r['Stockout risk']}. Current cover: "
                f"{r['Cover (periods)']:.1f} {unit}.",
            ]
            if flags:
                lines.append("Planner attention: " + "; ".join(flags) + ".")
            lines.append("Status: awaiting human approval.")
            drafts.append({"group": g, "kind": "PO", "text": "\n".join(lines)})
        elif flags or r["Excess units"] > 0:
            lines = [f"PLANNER ALERT - {g}"]
            if flags:
                lines.append("Attention: " + "; ".join(flags) + ".")
            if r["Excess units"] > 0:
                lines.append(f"Overstock: about {r['Excess units']:,.0f} units above the target level "
                             f"(value {r['Excess value']:,.0f}) is tying up working capital. "
                             f"Consider delaying the next order.")
            drafts.append({"group": g, "kind": "ALERT", "text": "\n".join(lines)})
    return drafts


def llm_polish(text, api_key, model="claude-sonnet-5-5"):
    """OPTIONAL: have Claude reword a draft. Needs `pip install anthropic` and an API key."""
    import anthropic  # imported here so the rest of the app works without it
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=model, max_tokens=600,
        messages=[{"role": "user", "content":
                   "Rewrite this purchase order justification for a procurement manager. Keep every number "
                   "exactly as given, add no new facts, and keep it concise.\n\n" + text}])
    return msg.content[0].text
