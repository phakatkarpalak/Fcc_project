# GreenWarm

Carbon-aware cold-start prediction and scheduling for serverless functions.
Project for Fundamentals of Cloud Computing (B.Tech AI & DS, second year).

## Idea
1. A predictor learns when each serverless function is called (average of past days, or gradient boosting).
2. A scheduler pre-warms the container just before a predicted call and decides how long to keep it warm.
3. The keep-alive window and the thresholds move with the carbon intensity of the grid and the function's latency tolerance: longer on clean hours, shorter on dirty hours.
4. Latency-tolerant functions can also (a) run in a cleaner region and (b) have their calls delayed a little (default up to 2 hours) to a cleaner minute.
5. A minute-by-minute simulator compares all of this with "no pre-warming", a fixed 10-minute keep-alive and a simplified Azure-style histogram policy. Each idea gets its own row so their effects are never mixed.

Metrics: cold-start rate, p95 latency, warm GB-minutes, emissions (gCO2, warm holding plus container boot energy), mean delay of deferred calls.

## Set up in VS Code
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```
Then in VS Code: Ctrl+Shift+P, "Python: Select Interpreter", pick `.venv`.

## Run
```bash
python run_experiment.py --shift-to sweden       # single run + 10-seed table + forecast-error table + charts in results/
python run_experiment.py --predictor gbm --region california --carbon-weight 1.5
python run_experiment.py --ci-error 0.2 --variability 0.4 --defer-max 0
streamlit run app.py                             # dashboard in the browser
pytest                                           # tests
```
The Run and Debug panel (F5) has ready launch entries for both.

## Use real data
* **Traffic:** download the Azure Functions 2019 trace (invocations_per_function_md.anon.dNN.csv) from the AzurePublicDataset repository on GitHub, then
  `python run_experiment.py --azure data/d01.csv data/d02.csv data/d03.csv data/d04.csv` (last file is the test day, the others train the predictor).
  Memory size and cold-start time are not in that file, so defaults are used. Edit them in `traffic.py` if you merge the memory trace.
* **Carbon:** replace the hourly lists in `carbon.py` with values from Electricity Maps, or use `load_carbon_csv()` on a 24-row CSV.

## Layout
```
src/greenwarm/carbon.py      carbon curves per region, noisy forecast
src/greenwarm/traffic.py     synthetic traffic and Azure trace loader
src/greenwarm/predictor.py   mean-rate and gradient boosting predictors, accuracy check
src/greenwarm/simulator.py   policies (fixed, histogram, green) and the minute-by-minute simulator
src/greenwarm/experiment.py  runs all policies, call delaying, multi-seed, forecast-error study, trade-off sweep
run_experiment.py            command-line run
app.py                       Streamlit dashboard
tests/                       pytest tests
```

## Running on real data
**Real traffic (Azure Functions 2019 trace).** Extract the archive, then point `--azure-dir` at the folder (subfolders are searched):

```
python run_experiment.py --azure-dir C:\Users\you\Downloads\azure_trace --train-days 1 2 3 4 5 --test-days 8 9 10 --shift-to sweden
```

Trains on the listed days and reports mean ± std over the test days (use 3 or more test days so the spread means something). Day files are read in chunks, so a full-size file (about 46,000 functions) needs well under 1 GB of memory; expect roughly 10 seconds of reading per day file, with progress printed. Functions are chosen using the training days only (never the test day). By default the 12 functions are spread evenly down the volume ranking (`--azure-pick busiest` takes the top 12, which flatters the cold-start problem less: busy functions rarely go cold). Run `python run_experiment.py --azure-dir <folder> --day-totals` first to see total calls per day and pick weekdays (the dataset description does not say which weekday day 1 is, nor the time zone). If an `app_memory_percentiles*dNN*.csv` file is in the folder, real memory sizes are used; otherwise 0.5 GB.

**Real carbon curves.** Any region can be replaced by a 24-row CSV (`hour`, `gco2_per_kwh`; hourly means for one month, UTC):

```
python run_experiment.py --carbon-csv india=in_we_july.csv --carbon-csv sweden=se_july.csv --shift-to sweden
```

Keep traffic and carbon on the same clock. The trace's time zone is not documented, so run with the curve shifted by a few hours as a check.

## Results (synthetic traffic, India grid, flexible functions shift to Sweden, 10 seeds, mean ± std)
Emissions are charged on the true carbon curve and include container boot energy. `vs fixed` is paired per seed against fixed 10-minute keep-alive.

| Policy | Cold starts % | Emissions vs fixed % |
|---|---|---|
| Fixed keep-alive 10 min | 7.3 ± 1.4 | 0 |
| Histogram (Azure-style) | 1.3 ± 0.3 | +79 ± 11 |
| Predictive, carbon off | 5.4 ± 0.7 | -16.6 ± 3.0 |
| GreenWarm keep-alive only | 5.6 ± 0.7 | -18.5 ± 3.2 |
| Predictive + region shift | 5.4 ± 0.7 | -64.0 ± 1.4 |
| Predictive + delay calls (14 min avg wait) | 5.3 ± 0.6 | -21.5 ± 2.9 |
| GreenWarm full (keep-alive + shift + delay) | 5.0 ± 0.7 | -64.4 ± 1.4 |

What this says, honestly:
* **Prediction is the main win over fixed keep-alive**: fewer cold starts and about 17% less carbon, simply by not holding idle containers.
* **Carbon-aware keep-alive is a small extra**: 2.3% ± 1.2% less carbon than the same predictor with carbon off (paired over 10 seeds), with a slightly higher cold-start rate. With a noisier carbon forecast the carbon number holds but cold starts rise (5.6% at no error to 6.8% at 30% error), so the saving is partly bought with latency.
* **Delaying flexible calls adds about 5 points** for roughly 14 minutes of average wait on India's curve.
* **Moving flexible work to a much cleaner grid dominates everything else** (about 64%). That is a placement result, not a scheduling one.
* **Histogram policy** reaches the fewest cold starts but uses about twice the warm memory, so it sits at a different point on the trade-off, not a better one.

### How much does carbon-awareness help as the grid gets more variable? (5 synthetic seeds)
`python run_experiment.py` also scales each carbon curve's daily swing around its mean (0 = flat grid, 1 = curves as given) and measures the extra saving over the same predictor with carbon off:

| Daily swing | Carbon-aware keep-alive saves | Delaying flexible calls saves |
|---|---|---|
| 0 (flat) | 0.0% | 0.0% |
| 0.5 | 1.0% ± 0.7 | 4.0% ± 0.6 |
| 1.0 (as given) | 2.8% ± 1.2 | 5.8% ± 0.4 |
| 2.0 | 6.4% ± 1.3 | 11.0% ± 0.6 |

A flat grid gives exactly zero, which is a useful sanity check. The gain grows with how much the grid varies, and delaying calls is consistently worth more than the keep-alive rule. Moving flexible functions out of the India curve saves about 28% (to Germany), 44% (to California) or 57% (to Sweden) of emissions, because the gap between grids is much larger than the swing within one.

## Assumptions and limits
One container per function, 3 W per GB held warm, 20 W per GB while booting, 80 ms extra per request in another region, Poisson arrivals when turning predicted rates into probabilities. Delayed calls are not counted in p95 latency (their wait is reported as `mean_delay_min`); wrap-around at midnight assumes a daily cycle. The histogram baseline is a simplified version of the paper's policy (no ARIMA fallback).
Not yet done: **no results on the real Azure trace or real carbon data are included here**; every number above comes from synthetic traffic and illustrative carbon curves. The real-data path is implemented and tested on files in the trace's format, but it has not been run on the real trace. No concurrency (one container per function), and the carbon forecast error is modelled as simple hourly noise. The delay-tolerant label for Azure functions is inferred from the trigger group (queue, timer, storage, event and orchestration may wait; http and others may not) and is an assumption.

## Patent angle
Be cautious. Carbon-aware keep-alive and placement for serverless is an active research area, and the measured gain from the keep-alive rule alone is small. Run a prior-art search before claiming novelty; the honest framing is a course project that quantifies where the carbon savings really come from.


## Dataset notes (Azure Functions 2019 trace)
Cite: Shahrad et al., "Serverless in the Wild", USENIX ATC 2020 (data is CC-BY). Only the invocation files are required. The memory files (`app_memory_percentiles.anon.d01..d12`, column `AverageAllocatedMb`) are used if present; they are per application and measure committed (allocated) memory. The duration files are not used: their times exclude cold starts, and execution energy is the same for every policy. Minute counts are recorded after execution, so they are completion times, and the trace samples applications randomly at an unspecified rate. Azure makes warm-up decisions per application, while this simulator keeps one container per function.
#   F c c _ p r o j e c t  
 