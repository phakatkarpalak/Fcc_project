"""Turn a raw hourly carbon-intensity CSV (for example from Electricity Maps) into the
24-row file GreenWarm reads (columns: hour, gco2_per_kwh).

Usage (from the project folder):
    python prepare_carbon_csv.py raw_in_we_2024.csv in_we_july.csv --month 7
    python prepare_carbon_csv.py raw_se_2024.csv  se_july.csv     --month 7

It finds the date column and the carbon-intensity column by name and PRINTS which ones it used,
so check that line. If it picks the wrong column, name it yourself with --column "exact name".
By default it prefers the "direct" intensity over life-cycle (LCA); say which you used in your report.
Hours are UTC, as in the source. Use --utc-offset-hours 5.5 to shift to India time (see README).
"""
import argparse
import sys

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("out")
    ap.add_argument("--month", type=int, help="keep only this month (1-12), e.g. 7 to match the July 2019 trace")
    ap.add_argument("--column", help="exact name of the carbon intensity column")
    ap.add_argument("--datetime-column", help="exact name of the date/time column")
    ap.add_argument("--utc-offset-hours", type=float, default=0.0)
    args = ap.parse_args()

    df = pd.read_csv(args.raw, encoding="utf-8-sig")
    cols = list(df.columns)

    tcol = args.datetime_column or next((c for c in cols if "datetime" in c.lower()), None) \
        or next((c for c in cols if "date" in c.lower() or "time" in c.lower()), None)
    if tcol is None:
        sys.exit(f"Could not find a date/time column. Columns are: {cols}. Use --datetime-column.")
    if args.column:
        vcol = args.column
    else:
        cands = [c for c in cols if "carbon intensity" in c.lower() or "carbon_intensity" in c.lower()]
        if not cands:
            sys.exit(f"Could not find a carbon intensity column. Columns are: {cols}. Use --column.")
        vcol = next((c for c in cands if "direct" in c.lower()), cands[0])
    print(f"Using date column:   {tcol!r}")
    print(f"Using value column:  {vcol!r}   (other candidates: "
          f"{[c for c in cols if 'carbon' in c.lower() and c != vcol]})")

    t = pd.to_datetime(df[tcol], utc=True, errors="coerce") + pd.Timedelta(hours=args.utc_offset_hours)
    v = pd.to_numeric(df[vcol], errors="coerce")
    ok = t.notna() & v.notna()
    print(f"Rows: {len(df)} total, {int(ok.sum())} usable ({len(df) - int(ok.sum())} dropped for missing values)")
    d = pd.DataFrame({"t": t[ok], "v": v[ok]})
    if args.month:
        d = d[d["t"].dt.month == args.month]
        print(f"Rows in month {args.month}: {len(d)}")
    if d.empty:
        sys.exit("No rows left. Check --month and the file's date range.")
    curve = d.groupby(d["t"].dt.hour)["v"].mean()
    missing = sorted(set(range(24)) - set(curve.index))
    if missing:
        sys.exit(f"Hours with no data: {missing}. Need all 24; use more months or a different file.")
    out = curve.rename_axis("hour").reset_index(name="gco2_per_kwh").round(1)
    out.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}: mean {out['gco2_per_kwh'].mean():.0f}, min {out['gco2_per_kwh'].min():.0f} "
          f"(hour {int(out.loc[out['gco2_per_kwh'].idxmin(), 'hour'])}), max {out['gco2_per_kwh'].max():.0f} "
          f"(hour {int(out.loc[out['gco2_per_kwh'].idxmax(), 'hour'])})")


if __name__ == "__main__":
    main()
