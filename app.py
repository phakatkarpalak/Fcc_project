"""Streamlit dashboard. Run with:  streamlit run app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from greenwarm.carbon import REGIONS, ci_series
from greenwarm.experiment import run_all, run_seeds, tradeoff_sweep
from greenwarm.traffic import synthetic_dataset

st.set_page_config(page_title="GreenWarm", layout="wide")
st.title("GreenWarm")
st.caption("Carbon-aware cold-start prediction and scheduling for serverless functions. "
           "Traffic is simulated; carbon curves are illustrative.")

with st.sidebar:
    st.header("Settings")
    region = st.selectbox("Region", list(REGIONS), format_func=lambda k: REGIONS[k]["name"])
    predictor = st.radio("Predictor", ["mean", "gbm"], format_func=lambda k: {"mean": "Average of past days", "gbm": "Gradient boosting"}[k])
    weight = st.slider("Carbon weight", 0.0, 3.0, 1.0, 0.1,
                       help="0 ignores carbon. Higher keeps containers warm longer on clean hours.")
    keep = st.slider("Baseline keep-alive (min)", 1, 30, 10)
    others = ["(stay)"] + [k for k in REGIONS if k != region]
    shift = st.selectbox("Move flexible functions to", others,
                         format_func=lambda k: k if k == "(stay)" else REGIONS[k]["name"])
    defer = st.slider("Max delay for flexible calls (min)", 0, 240, 120, 15,
                      help="Flexible (tolerant) calls may wait this long for a cleaner minute. 0 turns delaying off.")
    ci_error = st.slider("Carbon forecast error", 0.0, 0.4, 0.0, 0.05,
                         help="The scheduler decides from a noisy forecast; emissions are charged at the true intensity.")
    variability = st.slider("Day-to-day traffic variation", 0.0, 0.5, 0.25, 0.05,
                            help="0 = every day identical (easy for the predictor).")
    seed = st.number_input("Traffic seed", 1, 999, 7)

shift_to = None if shift == "(stay)" else shift


@st.cache_resource(show_spinner="Simulating...")
def compute(region, predictor, weight, keep, shift_to, seed, defer, ci_error, variability):
    ds = synthetic_dataset(int(seed), variability=variability)
    out = run_all(region, predictor, int(seed), keep, weight, shift_to, ds, defer, ci_error)
    fixed, green = tradeoff_sweep(out["functions"], out["test"], out["est"], region, int(seed), None, ci_error)
    return out, fixed, green


out, fixed_sweep, green_sweep = compute(region, predictor, weight, keep, shift_to, seed, defer, ci_error, variability)
res = out["results"]
FULL = "GreenWarm full" if "GreenWarm full" in res else "GreenWarm keep-alive only"
base, gw = res[f"Fixed keep-alive {keep} min"], res[FULL]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Cold-start rate (per function)", f"{gw.cold_rate_per_function*100:.1f}%", f"{(gw.cold_rate_per_function-base.cold_rate_per_function)*100:+.1f} pts vs fixed", delta_color="inverse")
c2.metric("p95 latency", f"{gw.p95_ms:.0f} ms", f"{(gw.p95_ms/base.p95_ms-1)*100:+.0f}% vs fixed", delta_color="inverse")
c3.metric("Warm memory", f"{gw.gb_min:,.0f} GB-min", f"{(gw.gb_min/base.gb_min-1)*100:+.0f}% vs fixed", delta_color="inverse")
c4.metric("Emissions", f"{gw.co2_g:.1f} g", f"{(gw.co2_g/base.co2_g-1)*100:+.0f}% vs fixed", delta_color="inverse")

st.subheader("Comparison")
st.dataframe(out["table"], hide_index=True, width='stretch')
q = out["predictor_quality"]
st.caption(f"Predictor on the test day: F1 {q['f1']:.2f}, precision {q['precision']:.2f}, recall {q['recall']:.2f} "
           "for 'a call arrives in the next 2 minutes'.")

left, right = st.columns(2)
with left:
    st.subheader("One day, hour by hour")
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    hours = list(range(24))
    ax.bar([h - 0.2 for h in hours], base.hour_gb_min, 0.4, color="#4f6d8f", label="Fixed")
    ax.bar([h + 0.2 for h in hours], gw.hour_gb_min, 0.4, color="#14805a", label="GreenWarm")
    ax.set_xlabel("Hour of day"); ax.set_ylabel("Warm GB-min")
    ax2 = ax.twinx(); ax2.plot(hours, ci_series(region).reshape(24, 60).mean(axis=1), color="#c9741a", lw=2)
    ax2.set_ylabel("gCO2/kWh", color="#c9741a"); ax.legend(loc="upper left")
    fig.tight_layout(); st.pyplot(fig)
with right:
    st.subheader("Latency against carbon")
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(fixed_sweep["cold_%"], fixed_sweep["co2_g"], "o-", color="#4f6d8f", label="Fixed keep-alive")
    ax.plot(green_sweep["cold_%"], green_sweep["co2_g"], "o-", color="#14805a", label="GreenWarm keep-alive only")
    ax.scatter([gw.cold_rate_per_function * 100], [gw.co2_g], s=140, facecolors="none", edgecolors="black", linewidths=2, label=f"{FULL} (your settings)")
    ax.set_xlabel("Cold-start rate (%)"); ax.set_ylabel("Emissions (g)"); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); st.pyplot(fig)

st.subheader("Mean ± std over 10 seeds")
st.caption("Fresh traffic for each seed; `vs_fixed_%` is computed per seed against that seed's fixed baseline.")
@st.cache_resource(show_spinner="Running 10 seeds...")
def multiseed(region, predictor, weight, keep, shift_to, defer, ci_error, variability):
    return run_seeds(range(1, 11), region=region, predictor=predictor, keep=keep, carbon_weight=weight,
                     shift_to=shift_to, defer_max=defer, ci_error=ci_error, variability=variability)
st.dataframe(multiseed(region, predictor, weight, keep, shift_to, defer, ci_error, variability), hide_index=True, width='stretch')

st.subheader("Per function")
rows = []
for i, f in enumerate(out["functions"]):
    a, b, c = res["No pre-warming"].per_function[i], res[f"Fixed keep-alive {keep} min"].per_function[i], gw.per_function[i]
    rate = lambda x: round(x["cold"] / x["inv"] * 100, 1) if x["inv"] else None
    rows.append({"function": f.name, "pattern": f.pattern, "SLA": f.sla, "calls": a["inv"],
                 "cold % none": rate(a), "cold % fixed": rate(b), "cold % GreenWarm": rate(c),
                 "GB-min fixed": round(b["gb_min"]), "GB-min GreenWarm": round(c["gb_min"])})
st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
