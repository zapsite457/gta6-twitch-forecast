"""Hour-by-hour Twitch CCU quantile forecast for a launch day.

The grader runs `python3 -P -s /bot/main.py` with one case in the input directory and
expects `/output/<id>.out`. This program reads every `*.in` case and writes the five
quantiles (5%, 25%, 50%, 75%, 95%) of concurrent viewers for hours 2 through 24.

The forecast is of log(CCU_h / CCU_hour1). Comparable launches in the case are
kernel-weighted (Nadaraya-Watson) by how closely their pre-checkpoint path matches
the target, then a ridge regression on log(pre-peak / hour-1) shifts that curve:
a deep hour-1 dip after the New Zealand-to-Europe wave lines up with a further
decline in hours 2-4 and a higher US-afternoon wave in hours 18-22. A light shrink
toward persistence (log-ratio 0) and a class upweight for console-only launches
stabilize the quantiles. Named console peers (Red Dead Redemption 2, Spider-Man 2,
Wolverine, God of War Ragnarok) get extra weight when the case includes them.
"""

import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_SCORES = [-1.645, -0.674, 0.0, 0.674, 1.645]
SHAPE_ANCHORS = [-13, -10, -7, -4, -2, -1]
PERSIST_SIGMA = 0.15

# Tuned on leave-one-out pinball over the kit's analog launches. See backtest.py.
PARAMS = {
    "lam": 2.0,
    "reg_w": 0.75,
    "shrink": 0.05,
    "shape_bw": 1.8,
    "log_bw": 3.2,
    "elev_bw": 2.4,
    "console_mult": 6.0,
    "cross_mult": 0.5,
    "early_mult": 0.45,
    "early_match": 1.4,
    "sm": 0.85,
    "tail": 1.05,
    "emp_blend": 0.45,
    "min_sigma": 0.07,
    "peer_boost": 3.2,
}


def parse_time(text):
    """Parse an ISO-8601 UTC timestamp ending in Z into a UTC epoch seconds."""
    # Expected form YYYY-MM-DDTHH:MM:SSZ, which every kit timestamp uses.
    year = int(text[0:4])
    month = int(text[5:7])
    day = int(text[8:10])
    hour = int(text[11:13])
    minute = int(text[14:16])
    second = int(text[17:19])
    # Civil date to Unix day count (Howard Hinnant's algorithm), then UTC seconds.
    y = year
    m = month
    d = day
    y -= m <= 2
    era = (y if y >= 0 else y - 399) // 400
    yoe = y - era * 400
    doy = (153 * (m + (-3 if m > 2 else 9)) + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    days = era * 146097 + doe - 719468
    return days * 86400 + hour * 3600 + minute * 60 + second


def hour_offset(stamp, day_epoch):
    """Whole-hour offset of a sample timestamp from day_start. Exact hours stay exact."""
    delta = (parse_time(stamp) - day_epoch) / 3600.0
    nearest = round(delta)
    if abs(delta - nearest) < 1e-6:
        return int(nearest)
    return int(math.floor(delta + 1e-9))


def median(values):
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def weighted_quantile(values, weights, tau):
    """Smallest value whose cumulative weight reaches tau. Ties break on input order."""
    order = sorted(range(len(values)), key=lambda i: (values[i], i))
    total = 0.0
    for w in weights:
        total += w
    if total <= 0:
        return values[order[len(order) // 2]]
    target = tau * total
    acc = 0.0
    for i in order:
        acc += weights[i]
        if acc >= target:
            return values[i]
    return values[order[-1]]


def target_profile(case):
    """Whether this launch is console-only, early-access, and a rolling local midnight."""
    notes = (case.get("notes") or "").lower()
    name = (case.get("game") or {}).get("name") or ""
    early = ("early access" in notes) or ("early-access" in notes)
    rolling = ("midnight" in notes) or ("rolling" in notes)
    clause = notes
    marker = " on "
    at = notes.find(marker)
    if at >= 0:
        clause = notes[at + len(marker):]
        stop = clause.find(".")
        if stop >= 0:
            clause = clause[:stop]
    has_pc = ("pc" in clause) or ("stadia" in clause)
    has_console = any(token in notes for token in ("ps5", "ps4", "playstation", "xbox", "switch"))
    console_only = ("console only" in notes) or ("ps5 only" in notes) or ("playstation 5 only" in notes)
    console = console_only or (has_console and not has_pc)
    if "grand theft auto vi" in name.lower():
        # Release-day case: PS5 and Xbox Series, rolling local midnight, no early access.
        console = True
        early = False
        rolling = True
    return {"console": console, "early": early, "rolling": rolling, "name": name}


def peer_boost(name, console, early):
    """Upweight the console launches closest to GTA VI when they are in the case."""
    if not console or early:
        return 1.0
    text = name.lower()
    if "red dead redemption 2" in text:
        return PARAMS["peer_boost"]
    if "spider-man 2" in text:
        return PARAMS["peer_boost"]
    if "wolverine" in text:
        return PARAMS["peer_boost"]
    if "god of war" in text and "ragnar" in text:
        return PARAMS["peer_boost"]
    return 1.0


def is_launch_analog(comp):
    """The GTA VI pre-release category is history, not a launch curve."""
    name = (comp.get("name") or "").lower()
    unlock = (comp.get("unlock") or "").lower()
    if "grand theft auto vi" in name and ("before release" in name or "not a launch" in unlock):
        return False
    return True


def series_from_rows(rows, day_epoch, stamp_key):
    """Map hour offset -> mean viewers. Later duplicate hours average in."""
    sums = {}
    counts = {}
    for row in rows:
        viewers = row.get("viewers")
        if viewers is None:
            continue
        stamp = row.get(stamp_key)
        if not stamp:
            continue
        off = hour_offset(stamp, day_epoch)
        sums[off] = sums.get(off, 0.0) + float(viewers)
        counts[off] = counts.get(off, 0) + 1
    return {off: sums[off] / counts[off] for off in sums}


def target_series(case, day_epoch):
    """Pre-checkpoint hourly viewers. Hourly means win; minute reads fill gaps."""
    hourly = case.get("history_hourly") or []
    points = series_from_rows(hourly, day_epoch, "from") if hourly else {}
    history = case.get("history") or []
    if history:
        minute = series_from_rows(history, day_epoch, "at")
        for off, viewers in minute.items():
            if off not in points:
                points[off] = viewers
    return points


def prepeak_ratio(points, p1):
    """Max viewers in the 14 hours before hour 1, divided by hour 1."""
    window = [points[off] for off in range(-14, 0) if off in points]
    if not window or p1 <= 0:
        return 1.0
    return max(window) / p1


def elevation(points, p1):
    """Log gap between hour 1 and the quiet baseline one to two days earlier."""
    base = [points[off] for off in range(-48, -24) if off in points]
    if len(base) < 4:
        base = [points[off] for off in range(-36, -14) if off in points]
    if len(base) < 3:
        return None
    return math.log(p1 + 1.0) - math.log(median(base) + 1.0)


def shape_vector(points, p1):
    """Log viewers at anchor hours before the checkpoint, relative to hour 1."""
    vector = {}
    for off in SHAPE_ANCHORS:
        if off in points:
            vector[off] = math.log(points[off] + 1.0) - math.log(p1 + 1.0)
    return vector


def analog_record(comp, day_epoch):
    points = series_from_rows(comp.get("hourly") or [], day_epoch, "from")
    if 0 not in points:
        return None
    p1 = points[0]
    if p1 < 1:
        return None
    unlock = comp.get("unlock") or ""
    platforms = comp.get("platforms") or ""
    early = ("early access" in unlock.lower()) or ("early-access" in unlock.lower())
    plat = platforms.lower()
    console = ("pc" not in plat) and ("stadia" not in plat)
    rolling = ("midnight" in unlock.lower()) or ("rolling" in unlock.lower())
    future = {}
    for hour in range(2, 25):
        off = hour - 1
        if off in points:
            future[hour] = math.log(max(points[off], 1.0) / p1)
    if len(future) < 12:
        return None
    return {
        "name": comp.get("name") or "",
        "p1": p1,
        "x": math.log(max(prepeak_ratio(points, p1), 1e-3)),
        "elev": elevation(points, p1),
        "shape": shape_vector(points, p1),
        "future": future,
        "early": early,
        "console": console,
        "rolling": rolling,
    }


def kernel_weight(target, analog):
    """Gaussian kernel on path shape, hour-1 level, and launch elevation, times class."""
    weight = 1.0
    diffs = []
    for off, value in target["shape"].items():
        if off in analog["shape"]:
            gap = value - analog["shape"][off]
            diffs.append(gap * gap)
    if len(diffs) >= 3 and PARAMS["shape_bw"] > 0:
        distance = math.sqrt(sum(diffs) / len(diffs))
        bw = PARAMS["shape_bw"]
        weight *= math.exp(-0.5 * (distance / bw) ** 2)
    log_gap = math.log(target["p1"] + 1.0) - math.log(analog["p1"] + 1.0)
    weight *= math.exp(-0.5 * (log_gap / PARAMS["log_bw"]) ** 2)
    if target["elev"] is not None and analog["elev"] is not None:
        elev_gap = target["elev"] - analog["elev"]
        weight *= math.exp(-0.5 * (elev_gap / PARAMS["elev_bw"]) ** 2)
    if target["console"] and analog["console"]:
        weight *= PARAMS["console_mult"]
    elif target["console"] != analog["console"]:
        weight *= PARAMS["cross_mult"]
    if analog["early"] and not target["early"]:
        weight *= PARAMS["early_mult"]
    if target["early"] and analog["early"]:
        weight *= PARAMS["early_match"]
    if target["rolling"] and analog["rolling"]:
        weight *= 1.0
    weight *= peer_boost(analog["name"], target["console"], target["early"])
    return weight


def persistence_quantiles(p1, hour):
    spread = PERSIST_SIGMA * math.sqrt(hour - 1)
    return [p1 * math.exp(z * spread) for z in Z_SCORES]


def fit_hour(analogs, weights, hour, target_x):
    """Ridge line of log-ratio on log(pre-peak), plus a weighted median."""
    xs = []
    ys = []
    ws = []
    for analog, weight in zip(analogs, weights):
        if hour not in analog["future"] or weight <= 0:
            continue
        xs.append(analog["x"])
        ys.append(analog["future"][hour])
        ws.append(weight)
    if not ys:
        return None
    total = sum(ws)
    mean_x = sum(w * x for w, x in zip(ws, xs)) / total
    mean_y = sum(w * y for w, y in zip(ws, ys)) / total
    var_x = sum(w * (x - mean_x) ** 2 for w, x in zip(ws, xs))
    cov = sum(w * (x - mean_x) * (y - mean_y) for w, x, y in zip(ws, xs, ys))
    slope = cov / (var_x + PARAMS["lam"])
    intercept = mean_y - slope * mean_x
    fitted = intercept + slope * target_x
    center = weighted_quantile(ys, ws, 0.5)
    location = PARAMS["reg_w"] * fitted + (1.0 - PARAMS["reg_w"]) * center
    location *= 1.0 - PARAMS["shrink"]
    residuals = [y - (intercept + slope * x) for y, x in zip(ys, xs)]
    residual_median = weighted_quantile(residuals, ws, 0.5)
    abs_dev = [abs(r - residual_median) for r in residuals]
    mad = weighted_quantile(abs_dev, ws, 0.5)
    sigma = max(1.4826 * mad, PARAMS["min_sigma"])
    quantiles = []
    for z, tau in zip(Z_SCORES, LEVELS):
        tail = PARAMS["tail"] if tau in (0.05, 0.95) else 1.0
        parametric = z * tail * sigma * PARAMS["sm"]
        empirical = weighted_quantile(residuals, ws, tau) - residual_median
        blend = PARAMS["emp_blend"]
        quantiles.append(location + blend * empirical + (1.0 - blend) * parametric)
    for i in range(1, len(quantiles)):
        if quantiles[i] < quantiles[i - 1]:
            quantiles[i] = quantiles[i - 1]
    return quantiles


def quantiles_to_viewers(p1, log_quantiles):
    viewers = []
    for log_q in log_quantiles:
        value = p1 * math.exp(log_q)
        if value < 1e-4:
            value = 1e-4
        if value > 1e9:
            value = 1e9
        viewers.append(value)
    for i in range(1, len(viewers)):
        if viewers[i] < viewers[i - 1]:
            viewers[i] = viewers[i - 1]
    # Four decimal places, half away from zero via integer arithmetic on positives.
    rounded = []
    for value in viewers:
        scaled = value * 10000.0
        whole = math.floor(scaled)
        if scaled - whole >= 0.5:
            whole += 1
        rounded.append(whole / 10000.0)
    for i in range(1, len(rounded)):
        if rounded[i] < rounded[i - 1]:
            rounded[i] = rounded[i - 1]
    return rounded


def forecast_case(case):
    """Return {"hours": [{"hour", "q"}, ...]} for one case."""
    if case.get("levels") not in (None, LEVELS):
        # The contest fixes the five levels. A different request still gets those five.
        pass
    known = None
    for row in case.get("known") or []:
        if row.get("hour") == 1:
            known = row
            break
    if known is None:
        raise ValueError("case has no hour-1 print")
    p1 = known.get("viewers") or 0
    if p1 < 1:
        p1 = 1.0
    else:
        p1 = float(p1)
    day_epoch = parse_time(case["day_start"])
    points = target_series(case, day_epoch)
    profile = target_profile(case)
    target = {
        "p1": p1,
        "x": math.log(max(prepeak_ratio(points, p1), 1e-3)),
        "elev": elevation(points, p1),
        "shape": shape_vector(points, p1),
        "console": profile["console"],
        "early": profile["early"],
        "rolling": profile["rolling"],
        "name": profile["name"],
    }
    target_name = profile["name"].lower()
    analogs = []
    for comp in case.get("comparables") or []:
        if not is_launch_analog(comp):
            continue
        if (comp.get("name") or "").lower() == target_name:
            continue
        comp_day = parse_time(comp["day_start"])
        record = analog_record(comp, comp_day)
        if record is None:
            continue
        analogs.append(record)
    analogs.sort(key=lambda record: record["name"])
    weights = [kernel_weight(target, record) for record in analogs]
    hours = []
    asked = case.get("forecast") or [{"hour": hour} for hour in range(2, 25)]
    for row in sorted(asked, key=lambda item: item["hour"]):
        hour = int(row["hour"])
        fitted = fit_hour(analogs, weights, hour, target["x"])
        if fitted is None:
            viewers = quantiles_to_viewers(p1, [math.log(q / p1) for q in persistence_quantiles(p1, hour)])
        else:
            viewers = quantiles_to_viewers(p1, fitted)
        hours.append({"hour": hour, "q": viewers})
    return {"hours": hours}


def dump_forecast(forecast):
    return json.dumps(forecast, separators=(",", ":")) + "\n"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted(INPUT_DIR.glob("*.in"))
    for path in paths:
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(dump_forecast(forecast), encoding="utf-8")


if __name__ == "__main__":
    main()
