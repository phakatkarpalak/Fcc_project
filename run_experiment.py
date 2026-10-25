"""Command-line run: prints the comparison tables and saves CSVs and charts to results/.

Synthetic traffic (default):
    python run_experiment.py --shift-to sweden

Real Azure traffic (train on days 1-5, test on days 8, 9, 10; folder is searched recursively):
    python run_experiment.py --azure-dir C:\\Users\\you\\Downloads\\azure_trace --train-days 1 2 3 4 5 --test-days 8 9 10 --shift-to sweden

Real carbon curves (24-row CSV with columns hour, gco2_per_kwh; repeat the flag per region):
    python run_experiment.py --carbon-csv india=in_we_july.csv --carbon-csv sweden=se_july.csv --shift-to sweden
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from greenwarm.carbon import REGIONS, ci_series, register_region_from_csv
from greenwarm.experiment import (paired_gain, per_function_table, region_gap_table, robustness, run_all, run_datasets,
                                  run_seeds, sensitivity, tradeoff_sweep)
from greenwarm.traffic import _scan_totals, azure_day_files, load_azure_days, load_azure_trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="india")
    ap.add_argument("--predictor", default="mean", choices=["mean", "gbm"])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--seeds", type=int, default=10, help="number of synthetic seeds for the mean ± std table (0 = skip)")
    ap.add_argument("--keep", type=int, default=10, help="fixed keep-alive baseline in minutes")
    ap.add_argument("--carbon-weight", type=float, default=1.0)
    ap.add_argument("--shift-to", default=None)
    ap.add_argument("--defer-max", type=int, default=120, help="max minutes a tolerant call may wait for cleaner power (0 = off)")
    ap.add_argument("--ci-error", type=float, default=0.0, help="std of the carbon forecast error the scheduler sees")
    ap.add_argument("--variability", type=float, default=0.25, help="day-to-day variation in the synthetic traffic (0 = identical days)")
    ap.add_argument("--carbon-csv", action="append", default=[], metavar="REGION=FILE",
                    help="load a real 24-row carbon curve for REGION (hour, gco2_per_kwh columns); repeatable")
    ap.add_argument("--azure-dir", help="folder holding the extracted Azure trace (searched recursively)")
    ap.add_argument("--train-days", type=int, nargs="*", default=[1, 2, 3, 4, 5])
    ap.add_argument("--test-days", type=int, nargs="*", default=[8])
    ap.add_argument("--azure-functions", type=int, default=12, help="how many functions to sample from the trace")
    ap.add_argument("--azure-pick", default="spread", choices=["spread", "busiest"])
    ap.add_argument("--azure", nargs="*", help="explicit Azure day files; the last one is the test day")
    ap.add_argument("--skip-sensitivity", action="store_true")
    ap.add_argument("--day-totals", action="store_true",
                    help="with --azure-dir: print total calls per day file (to spot weekends) and exit")
    args = ap.parse_args()

    if args.day_totals:
        if not args.azure_dir:
            ap.error("--day-totals needs --azure-dir")
        print("Total invocations per day file (weekends usually show clearly lower totals):")
        for d in range(1, 15):
            try:
                f = azure_day_files(args.azure_dir, [d])[0]
            except FileNotFoundError:
                continue
            print(f"  day {d:02d}: {int(_scan_totals(f).sum()):>14,}", flush=True)
        return

    for spec in args.carbon_csv:
        key, _, path = spec.partition("=")
        register_region_from_csv(key.strip(), path.strip())
        print(f"Loaded real carbon curve for '{key.strip()}' from {path.strip()}")
    for r in (args.region, args.shift_to):
        if r and r not in REGIONS:
            ap.error(f"unknown region '{r}'. Built-in: {', '.join(REGIONS)}; add more with --carbon-csv REGION=FILE")

    datasets = None
    if args.azure_dir:
        files = azure_day_files(args.azure_dir, list(args.train_days) + list(args.test_days))
        datasets = load_azure_days(files, len(args.train_days), n_functions=args.azure_functions, pick=args.azure_pick,
                                   folder=args.azure_dir, memory_day=args.train_days[0], verbose=True)
        fns = datasets[0][0]
        print(f"\nAzure trace: {len(fns)} functions, trained on days {args.train_days}, tested on days {args.test_days}")
        from collections import Counter
        trig = Counter(f.pattern.rsplit("trigger: ", 1)[-1] for f in fns)
        print("  triggers in the sample:", dict(trig))
        print("  treated as delay-tolerant (assumption from trigger):", sum(f.sla == "tolerant" for f in fns), "of", len(fns))
    elif args.azure:
        datasets = [load_azure_trace(args.azure, n_functions=args.azure_functions, pick=args.azure_pick)]

    common = dict(region=args.region, predictor=args.predictor, keep=args.keep, carbon_weight=args.carbon_weight,
                  shift_to=args.shift_to, defer_max=args.defer_max)
    if datasets is None:
        common["variability"] = args.variability

    out = run_all(seed=args.seed, ci_error=args.ci_error, dataset=datasets[0] if datasets else None, **common)
    print("\nPredictor quality on the test day:", {k: round(v, 3) for k, v in out["predictor_quality"].items()})
    print(f"\nSingle run:\n", out["table"].to_string(index=False), "\n")

    res = Path(__file__).parent / "results"
    res.mkdir(exist_ok=True)
    out["table"].to_csv(res / "summary.csv", index=False)
    pf = per_function_table(out, args.keep)
    print("Per function (busiest first). cold_start_% in the table above counts every call equally, so the busiest\n"
          "function can hide the quiet ones; cold_per_fn_% and this table show each function separately:\n", pf.to_string(index=False), "\n")
    pf.to_csv(res / "per_function.csv", index=False)

    if datasets is not None and len(datasets) > 1:
        ms = run_datasets(datasets, ci_error=args.ci_error, **common)
        print(f"Mean ± std over {len(datasets)} test days:\n", ms.to_string(index=False), "\n")
        ms.to_csv(res / "summary_multiseed.csv", index=False)
    elif datasets is None and args.seeds > 0:
        ms = run_seeds(range(1, args.seeds + 1), ci_error=args.ci_error, **common)
        print(f"Mean ± std over {args.seeds} seeds:\n", ms.to_string(index=False), "\n")
        ms.to_csv(res / "summary_multiseed.csv", index=False)

    seeds = range(1, max(args.seeds, 2) + 1)
    rb = robustness(seeds, datasets=datasets, **common)
    print("Carbon forecast error:\n", rb.to_string(index=False), "\n")
    rb.to_csv(res / "robustness_ci_error.csv", index=False)
    m, s, n = paired_gain(seeds, "Predictive, carbon off", "GreenWarm keep-alive only", datasets=datasets, ci_error=args.ci_error, **common)
    spread = f" ± {s:.2f}%" if n > 1 else " (one run only, so no spread; use 3+ test days)"
    print(f"Carbon-aware keep-alive vs carbon-off, paired over {n} run(s): {m:.2f}%{spread} less emissions\n")

    if not args.skip_sensitivity:
        sens_kw = {k: v for k, v in common.items() if k not in ("shift_to",)}
        sens = sensitivity(seeds=range(1, 6), datasets=datasets, ci_error=args.ci_error, **sens_kw)
        print("Sensitivity to the grid's daily swing (0 = flat grid, 1 = curves as given):\n", sens.to_string(index=False), "\n")
        sens.to_csv(res / "sensitivity.csv", index=False)
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.errorbar(sens["swing"], sens["keep_alive_gain_%"], yerr=sens["keep_alive_gain_std"], marker="o", color="#14805a", label="Carbon-aware keep-alive")
        ax.errorbar(sens["swing"], sens["delay_gain_%"], yerr=sens["delay_gain_std"], marker="s", color="#c9741a", label="Delaying flexible calls")
        ax.set_xlabel("Daily swing of the carbon curve (0 = flat grid, 1 = as given)")
        ax.set_ylabel("Extra emissions saved vs carbon-off (%)")
        ax.set_title("When does carbon-awareness pay off?"); ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout(); fig.savefig(res / "sensitivity.png", dpi=150)
        if datasets is None:
            targets = [t for t in REGIONS if t != args.region]
            gap = region_gap_table(args.region, targets, seeds=range(1, 6),
                                   **{k: v for k, v in common.items() if k not in ("shift_to", "region")})
            print(f"Emissions saved by moving flexible functions from {args.region}:\n", gap.to_string(index=False), "\n")
            gap.to_csv(res / "region_gap.csv", index=False)

    fixed, green = tradeoff_sweep(out["functions"], out["test"], out["est"], args.region, args.seed, None, args.ci_error)
    fixed.to_csv(res / "sweep_fixed.csv", index=False)
    green.to_csv(res / "sweep_green.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fixed["cold_%"], fixed["co2_g"], "o-", color="#4f6d8f", label="Fixed keep-alive (sweep)")
    ax.plot(green["cold_%"], green["co2_g"], "o-", color="#14805a", label="GreenWarm keep-alive only (carbon-weight sweep)")
    markers = ["s", "D", "^", "v", "P", "X", "*", "h"]
    for i, (_, row) in enumerate(out["table"].iterrows()):
        if row["policy"] == "No pre-warming":
            continue
        ax.scatter(row["cold_per_fn_%"], row["emissions_g"], s=90, marker=markers[i % len(markers)],
                   edgecolors="black", linewidths=0.6, zorder=3, label=row["policy"])
    ax.set_xlim(left=0)
    ax.set_xlabel("Cold-start rate, averaged over functions (%)"); ax.set_ylabel("Emissions: warm containers + boots (g)")
    ax.set_title(f"Latency vs carbon ({args.region}, {args.predictor} predictor)")
    ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout(); fig.savefig(res / "tradeoff.png", dpi=150)

    fig, ax = plt.subplots(figsize=(8, 4))
    hours = range(24)
    full = "GreenWarm full" if "GreenWarm full" in out["results"] else "GreenWarm keep-alive only"
    ax.bar([h - 0.2 for h in hours], out["results"][f"Fixed keep-alive {args.keep} min"].hour_gb_min, 0.4, color="#4f6d8f", label="Fixed")
    ax.bar([h + 0.2 for h in hours], out["results"][full].hour_gb_min, 0.4, color="#14805a", label=full)
    ax.set_xlabel("Hour of day"); ax.set_ylabel("Warm GB-min per hour")
    ax2 = ax.twinx()
    ci = ci_series(args.region).reshape(24, 60).mean(axis=1)
    ax2.plot(hours, ci, color="#c9741a", lw=2, label="Carbon intensity"); ax2.set_ylabel("gCO2/kWh")
    ax.legend(loc="upper left"); fig.tight_layout(); fig.savefig(res / "day.png", dpi=150)
    print("Saved CSVs and charts to results/")


if __name__ == "__main__":
    main()
