#!/usr/bin/env python3
"""GTA VI trophy earn-rate forecaster (make.fun gta6-trophy-rarity-forecast-01).

Reads each case file /input/<id>.in (every Exophase read of the game's trophy
page before the case checkpoint) and writes /output/<id>.out: for every trophy,
five quantiles (5/25/50/75/95%) of its earn rate at the case target, in percent.

Model (per trophy, on the logit scale):
- Earn rates saturate: a trophy's rate grows fast while players are new to it,
  then flattens toward a ceiling, so growth is modelled as an exponential
  approach to a tier-based ceiling, not a straight line.
- The page refreshes its counts hours apart, not continuously. A run of reads
  that never moves is a stale snapshot, and the next refresh dilutes every
  rate: the tracked-player count jumps and the new players hold fewer
  trophies. When the history is flat, the forecast expects that refresh and
  dilutes the rates with a power-law popularity model calibrated on the one
  refresh the kit documents (launch night: 638 -> 3868 players tracked,
  943 -> 4276 trophies earned, every rate but the 100% one fell).
- When the history moves, the recent trend is fit on log-time (growth slows
  as the game ages), damped over the horizon, capped at the ceiling, and
  reduced by the dilution the refreshes still to come are expected to cause.
- The quantile spread comes from the fit's residual, the horizon, and the
  refresh uncertainty, so an honest wide band is written when the forecast
  is unsure. Deterministic: no randomness, no clock, no network.

Stdlib only. Runs in well under 8 seconds per case.
"""

import glob
import json
import math
import os
import sys
from datetime import datetime, timezone

# Standard normal quantiles for the levels this competition uses.
Z = {0.05: -1.6448536269514722, 0.25: -0.6744897501960817, 0.5: 0.0,
     0.75: 0.6744897501960817, 0.95: 1.6448536269514722}

# Ceiling earn rate (percent) a trophy can approach, by tier. The platinum
# needs 100% completion, so almost nobody reaches it in a week; story bronze
# trophies are the chapters most players finish.
CEILING = {"platinum": 2.0, "gold": 12.0, "silver": 25.0, "bronze": 45.0}

# Refresh priors, calibrated on the kit's documented launch-night refresh
# (638 -> 3868 players tracked, 943 -> 4276 trophies earned): a refresh
# multiplies the tracked players by about this much, and the newly tracked
# players hold about this fraction as many trophies each as the old ones.
REFRESH_GROWTH = 6.0
REFRESH_HOLD = 0.70


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit_pct(p):
    """Percent (0..100) -> logit, clamped where the logit does not exist."""
    p = clamp(p, 0.01, 99.99) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(y):
    """Logit -> percent."""
    if y >= 30.0:
        return 99.99
    if y <= -30.0:
        return 0.01
    return 100.0 / (1.0 + math.exp(-y))


def parse_ts(s):
    """ISO timestamp -> hours since the Unix epoch (fractional part dropped)."""
    s = s.strip()
    if "." in s:
        s = s.split(".")[0] + "Z"
    dt = datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return dt.timestamp() / 3600.0


def z_for(tau):
    """Standard normal quantile for a level (Acklam's approximation)."""
    if tau in Z:
        return Z[tau]
    p = tau
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0
    # Acklam's rational approximation, good to ~1e-9.
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285984469534e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def solve_power(a, target_sum):
    """Find p in [1, 4] with sum(a_i ** p) = target_sum, by bisection."""
    lo, hi = 1.0, 4.0
    if sum(a) <= target_sum:
        return 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if sum(x ** mid for x in a) > target_sum:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def forecast_flat(last_rates):
    """History never moved: the page is stale, a refresh is due.

    The refresh multiplies the tracked players by REFRESH_GROWTH and the new
    players hold trophies at a lower rate than the old ones, concentrated on
    the popular trophies: a new player holds trophy i with probability
    a_i ** p, where a_i is the old fraction holding it and p is solved so
    the new players hold REFRESH_HOLD as many trophies each as the old ones.
    Returns (median_rate_per_trophy, spread_logit).
    """
    a = [clamp(r, 0.01, 99.99) / 100.0 for r in last_rates]
    total_old = sum(a)
    if total_old <= 0.0:
        return [0.01] * len(a), 0.6
    p = solve_power(a, REFRESH_HOLD * total_old)
    g = REFRESH_GROWTH
    med = []
    for ai in a:
        bi = ai ** p
        # Old players keep theirs; new players hold bi each; all pooled.
        med.append(clamp((ai + (g - 1.0) * bi) / g * 100.0, 0.01, 99.99))
    # The refresh itself is uncertain (its size, its timing), so the band is
    # wide: about 0.6 logit of spread.
    return med, 0.6


def forecast_case(case):
    """Return the list of {"id": n, "q": [q05, q25, q50, q75, q95]} rows."""
    trophies = case["trophies"]
    hist = case["history"]
    levels = case.get("levels") or [0.05, 0.25, 0.5, 0.75, 0.95]
    at = parse_ts(case["at"])
    target = parse_ts(case["target"])
    horizon_h = max(0.0, target - at)
    horizon_d = horizon_h / 24.0

    t0 = parse_ts(hist[0]["slot"])
    times = [parse_ts(r["slot"]) - t0 for r in hist]
    players = [float(r["players_tracked"]) for r in hist]
    n = len(hist)
    last_rates = hist[-1]["rates"]

    # Is the whole history flat? (max logit move over all trophies and reads)
    max_move = 0.0
    for i in range(len(trophies)):
        ys = [logit_pct(r["rates"][i]) for r in hist]
        max_move = max(max_move, max(ys) - min(ys))
    flat = (n < 3) or (max_move < 0.02)

    if flat:
        med_rates, spread = forecast_flat(last_rates)
        rows = []
        for t, m in zip(trophies, med_rates):
            if m >= 99.99:
                rows.append({"id": t["id"], "q": [99.99] * 5})
                continue
            y = logit_pct(m)
            q = [round(clamp(inv_logit_pct(y + z_for(tau) * spread), 0.01, 99.99), 4)
                 for tau in levels]
            for k in range(1, 5):
                q[k] = max(q[k], q[k - 1])
            rows.append({"id": t["id"], "q": q})
        return rows

    # --- Moving history: trend on log-time since release, damped, capped.
    # Fit on the whole history with recent reads weighted heaviest: the
    # fitted line already averages over the refresh sawtooth, so the net
    # trend (growth minus dilution) is what is extrapolated. A separate
    # dilution term would double-count it: refresh drops recover as the new
    # players earn, so they do not accumulate over the horizon.
    release = parse_ts("2026-11-19T00:00:00Z")
    t_last = times[-1]
    # Log-time since release (growth slows as the game ages).
    lt_all = [math.log1p(max(0.0, parse_ts(r["slot"]) - release)) for r in hist]
    lt_last = lt_all[-1]
    lt_target = math.log1p(max(0.0, parse_ts(case["target"]) - release))

    # Refresh volatility seen in the history (for the spread only): the
    # median logit drop at a refresh, and the mean time between refreshes.
    drops = []
    for j in range(1, n):
        if players[j] > players[j - 1] * 1.05:
            d = []
            for i in range(len(trophies)):
                y0 = logit_pct(hist[j - 1]["rates"][i])
                y1 = logit_pct(hist[j]["rates"][i])
                if y1 < y0:
                    d.append(y0 - y1)
            if d:
                d.sort()
                drops.append(d[len(d) // 2])
    drop_per = sorted(drops)[len(drops) // 2] if drops else 0.0

    rows = []
    for i, t in enumerate(trophies):
        ys = [logit_pct(r["rates"][i]) for r in hist]
        y_last = logit_pct(last_rates[i])
        if y_last >= 9.0:
            rows.append({"id": t["id"], "q": [99.99] * 5})
            continue
        # Weighted least squares of logit on log-time, recent reads heavier
        # (half-life 12 hours, so the last day dominates and the stale
        # launch-night reads barely count).
        w = [math.exp(-(t_last - times[j]) / 12.0) for j in range(n)]
        sw = sum(w)
        mx = sum(wi * x for wi, x in zip(w, lt_all)) / sw
        my = sum(wi * y for wi, y in zip(w, ys)) / sw
        sxx = sum(wi * (x - mx) ** 2 for wi, x in zip(w, lt_all))
        slope = 0.0
        if sxx > 1e-9:
            slope = sum(wi * (x - mx) * (y - my) for wi, x, y in zip(w, lt_all, ys)) / sxx
        inter = my - slope * mx
        # Residual spread of the fit (the refresh sawtooth lives here).
        resid = [y - (inter + slope * x) for x, y in zip(lt_all, ys)]
        var = sum(wi * r * r for wi, r in zip(w, resid)) / sw if sw else 0.0
        resid_sd = math.sqrt(max(var, 0.0))
        # Trend read at the target, damped over the horizon, capped at the
        # tier ceiling. A falling trend is not extrapolated: declines come
        # from refreshes, and the net trend already carries them.
        s = max(0.0, slope)
        y_trend = inter + s * lt_target
        damp = 1.0 / (1.0 + horizon_d / 4.0)
        y_med = y_last + (y_trend - y_last) * damp
        ceil_rate = max(CEILING.get(t.get("tier", "bronze"), 45.0),
                        last_rates[i] * 2.0)
        ceil_rate = min(ceil_rate, 95.0)
        y_med = min(y_med, logit_pct(ceil_rate))
        # Spread: the fit's own noise, plus a random-walk term that grows
        # with the horizon, plus the refresh volatility when the history
        # shows refreshes (each refresh is a jump the trend cannot time).
        var_h = 0.25 * 0.25 * horizon_d
        var_r = drop_per * drop_per * 0.25
        spread = math.sqrt(resid_sd * resid_sd + var_h + var_r)
        spread = clamp(spread, 0.25, 1.5)
        y = y_med
        q = [round(clamp(inv_logit_pct(y + z_for(tau) * spread), 0.01, 99.99), 4)
             for tau in levels]
        for k in range(1, 5):
            q[k] = max(q[k], q[k - 1])
        rows.append({"id": t["id"], "q": q})
    return rows


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(in_dir, "*.in"))):
        with open(path, "r", encoding="utf-8") as fh:
            case = json.load(fh)
        rows = forecast_case(case)
        out_path = os.path.join(out_dir, case["id"] + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump({"trophies": rows}, fh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
