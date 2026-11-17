"""Hour-by-hour Twitch CCU quantile forecast for a launch day.

The grader runs `python3 -P -s /bot/main.py` with one case in the input directory and
expects `/output/<id>.out`. This program reads every `*.in` case and writes the five
quantiles (5%, 25%, 50%, 75%, 95%) of concurrent viewers for hours 2 through 24.

The forecast is of log(CCU_h / CCU_hour1). A console launch that already has an
audience at hour 1 uses a local model: neighbors weighted by the pre-checkpoint
dip and the hour-1 level, then a ridge line in that dip blended with the weighted
median. Widths are a scaled MAD. Other launches keep the broader kernel regression.
Named console peers (Red Dead Redemption 2, Spider-Man 2, Wolverine, God of War
Ragnarok) get extra weight when the case includes them. A second ridge term on
log(hour-1 viewers) was fit on one half of the kit and rejected there, so its
penalty stays large enough to turn it off.
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
    "lam": 4.0,
    "reg_w": 0.8,
    "shrink": 0.0,
    "shape_bw": 1.8,
    "log_bw": 3.2,
    "elev_bw": 2.4,
    "console_mult": 6.0,
    "cross_mult": 0.5,
    "early_mult": 0.45,
    "early_match": 1.4,
    "sm": 0.95,
    "tail": 1.2,
    "emp_blend": 0.4,
    "min_sigma": 0.07,
    "peer_boost": 3.2,
    "huber_cap": 1.6,
    "huber_steps": 4,
    "smooth": 0.7,
    "log_bw_small": 1.35,
    "small_p1": 12000.0,
    "roll_mult": 1.0,
    "spec_bw_x": 0.8,
    "spec_bw_level": 3.0,
    "spec_scale": 0.8,
    "spec_ridge": 0.7,
    "spec_lam_level": 1.0e9,
    # How far the afternoon median moves toward a strong-wave/flat-wave blend,
    # and how much ambiguous cases widen to cover both regimes.
    "spec_mix": 1.0,
    "spec_hedge": 0.0,
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
    log_bw = PARAMS["log_bw"]
    if target["p1"] < PARAMS["small_p1"]:
        log_bw = PARAMS["log_bw_small"]
    weight *= math.exp(-0.5 * (log_gap / log_bw) ** 2)
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
        weight *= PARAMS["roll_mult"]
    weight *= peer_boost(analog["name"], target["console"], target["early"])
    return weight


def persistence_quantiles(p1, hour):
    spread = PERSIST_SIGMA * math.sqrt(hour - 1)
    return [p1 * math.exp(z * spread) for z in Z_SCORES]


def ridge_fit(xs, ys, ws):
    """Weighted ridge of y on x. Slope is penalized, intercept is free."""
    total = sum(ws)
    mean_x = sum(w * x for w, x in zip(ws, xs)) / total
    mean_y = sum(w * y for w, y in zip(ws, ys)) / total
    var_x = sum(w * (x - mean_x) ** 2 for w, x in zip(ws, xs))
    cov = sum(w * (x - mean_x) * (y - mean_y) for w, x, y in zip(ws, xs, ys))
    slope = cov / (var_x + PARAMS["lam"])
    intercept = mean_y - slope * mean_x
    return intercept, slope


def huber_fit(xs, ys, ws):
    """Ridge, then Huber steps with a fixed log cap.

    A scale taken from the whole pool treats the console afternoon wave as an
    outlier and flattens it. Capping only residuals past about one log unit
    (a factor of three) drops late-unlock spikes and keeps that wave.
    """
    intercept, slope = ridge_fit(xs, ys, ws)
    cap = PARAMS["huber_cap"]
    for _ in range(PARAMS["huber_steps"]):
        robust = []
        for weight, x, y in zip(ws, xs, ys):
            gap = abs(y - (intercept + slope * x))
            if gap <= cap:
                robust.append(weight)
            else:
                robust.append(weight * cap / gap)
        intercept, slope = ridge_fit(xs, ys, robust)
    return intercept, slope


def specialist_log_quantiles(target, analogs):
    """Local pre-peak model for a console launch that is already drawing viewers.

    Neighbors are weighted by how close their pre-checkpoint dip and their hour-1
    level are. A ridge line in that dip, blended with the weighted median, is the
    location. Widths are a scaled MAD, wider in the tails than a plain normal.
    Chosen on one half of the kit launches; the other half is the check.
    """
    bw_x = PARAMS["spec_bw_x"]
    bw_level = PARAMS["spec_bw_level"]
    scale = PARAMS["spec_scale"]
    ridge_share = PARAMS["spec_ridge"]
    xs = []
    levels = []
    dips = []
    curves = []
    weights = []
    for analog in analogs:
        if analog["x"] is None or len(analog["future"]) < 12:
            continue
        weight = math.exp(-0.5 * ((target["x"] - analog["x"]) / bw_x) ** 2)
        level_gap = math.log(target["p1"] + 1.0) - math.log(analog["p1"] + 1.0)
        weight *= math.exp(-0.5 * (level_gap / bw_level) ** 2)
        clean_console = analog["console"] and not analog["early"]
        if clean_console:
            weight *= 8.0
        else:
            weight *= 0.25
        weight *= peer_boost(analog["name"], True, False)
        if weight <= 1e-8:
            continue
        xs.append(analog["x"])
        levels.append(math.log(analog["p1"] + 1.0))
        dips.append(analog["shape"].get(-1, 0.0))
        curves.append(analog["future"])
        weights.append(weight)
    if not xs:
        return None
    rows = []
    target_level = math.log(target["p1"] + 1.0)
    for hour in range(2, 25):
        column = []
        used_w = []
        used_x = []
        used_level = []
        for curve, weight, x_value, level in zip(curves, weights, xs, levels):
            if hour not in curve:
                continue
            column.append(curve[hour])
            used_w.append(weight)
            used_x.append(x_value)
            used_level.append(level)
        if not column:
            rows.append([0.0, 0.0, 0.0, 0.0, 0.0])
            continue
        used_total = sum(used_w)
        mean_y = sum(weight * value for weight, value in zip(used_w, column)) / used_total
        mean_x_hour = sum(weight * x_value for weight, x_value in zip(used_w, used_x)) / used_total
        mean_level = sum(weight * level for weight, level in zip(used_w, used_level)) / used_total
        var_x = sum(weight * (x_value - mean_x_hour) ** 2 for weight, x_value in zip(used_w, used_x))
        var_level = sum(weight * (level - mean_level) ** 2 for weight, level in zip(used_w, used_level))
        cov_x = sum(
            weight * (x_value - mean_x_hour) * (value - mean_y)
            for weight, x_value, value in zip(used_w, used_x, column)
        )
        cov_level = sum(
            weight * (level - mean_level) * (value - mean_y)
            for weight, level, value in zip(used_w, used_level, column)
        )
        cross = sum(
            weight * (x_value - mean_x_hour) * (level - mean_level)
            for weight, x_value, level in zip(used_w, used_x, used_level)
        )
        sxx = var_x + 0.8
        sll = var_level + PARAMS["spec_lam_level"]
        det = sxx * sll - cross * cross
        if abs(det) < 1e-12:
            slope = cov_x / sxx
            slope_level = 0.0
        else:
            slope = (cov_x * sll - cov_level * cross) / det
            slope_level = (cov_level * sxx - cov_x * cross) / det
        fitted = (
            mean_y
            + slope * (target["x"] - mean_x_hour)
            + slope_level * (target_level - mean_level)
        )
        center = weighted_quantile(column, used_w, 0.5)
        location = ridge_share * fitted + (1.0 - ridge_share) * center
        residuals = []
        for value, x_value, level in zip(column, used_x, used_level):
            analog_fit = (
                mean_y
                + slope * (x_value - mean_x_hour)
                + slope_level * (level - mean_level)
            )
            residuals.append(value - analog_fit)
        residual_median = weighted_quantile(residuals, used_w, 0.5)
        mad = weighted_quantile([abs(item - residual_median) for item in residuals], used_w, 0.5)
        sigma = max(1.4826 * mad, 0.08) * scale
        quantiles = []
        for z_value in Z_SCORES:
            tail = 1.15 if abs(z_value) > 1.0 else 1.0
            quantiles.append(location + z_value * sigma * tail)
        for index in range(1, 5):
            if quantiles[index] < quantiles[index - 1]:
                quantiles[index] = quantiles[index - 1]
        rows.append(quantiles)
    return hedge_afternoon(target, analogs, rows)


def afternoon_window(hour):
    """Ramp into the US-afternoon hours and ease off after them."""
    if hour < 14:
        return 0.0
    if hour < 18:
        return (hour - 14) / 4.0
    if hour <= 22:
        return 1.0
    return 0.5


def _sigmoid(value):
    if value > 20.0:
        return 1.0
    if value < -20.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-value))


def hedge_afternoon(target, analogs, rows):
    """Blend a strong afternoon wave and a flat one when the signal is mixed.

    The two regimes are fit on a wide pool, so the flat side can be a launch
    that never had a second wave, not just the quieter half of the strong-wave
    neighbors. The target's weight on the strong side is a ridge logistic in
    the pre-checkpoint dip, the hour before hour 1, and log hour-1 level. The
    median moves part of the way toward that blend. Ambiguous probabilities
    widen both tails by the gap between the regime medians.
    """
    mix = PARAMS["spec_mix"]
    hedge = PARAMS["spec_hedge"]
    if (mix <= 0.0 and hedge <= 0.0) or len(rows) != 23:
        return rows
    packed = []
    for analog in analogs:
        curve = analog["future"]
        block = [curve[hour] for hour in range(18, 23) if hour in curve]
        if len(block) < 3 or analog["x"] is None:
            continue
        weight = math.exp(-0.5 * ((target["x"] - analog["x"]) / 1.2) ** 2)
        level_gap = math.log(target["p1"] + 1.0) - math.log(analog["p1"] + 1.0)
        weight *= math.exp(-0.5 * (level_gap / 2.2) ** 2)
        if analog["console"] and not analog["early"]:
            weight *= 2.0
        elif analog["early"]:
            weight *= 0.45
        if weight <= 1e-6:
            continue
        packed.append((
            sum(block) / len(block),
            analog["x"],
            math.log(analog["p1"] + 1.0),
            analog["shape"].get(-1, 0.0),
            curve,
            weight,
        ))
    if len(packed) < 6:
        return rows
    scores = [item[0] for item in packed]
    score_w = [item[5] for item in packed]
    cutoff = weighted_quantile(scores, score_w, 0.5)
    labels = [1.0 if score >= cutoff else 0.0 for score in scores]
    if sum(labels) < 2 or sum(labels) > len(labels) - 2:
        return rows
    total_w = sum(score_w)
    mean_x = sum(weight * item[1] for weight, item in zip(score_w, packed)) / total_w
    mean_level = sum(weight * item[2] for weight, item in zip(score_w, packed)) / total_w
    mean_dip = sum(weight * item[3] for weight, item in zip(score_w, packed)) / total_w

    def spread(index, mean):
        var = sum(weight * (item[index] - mean) ** 2 for weight, item in zip(score_w, packed)) / total_w
        return math.sqrt(var) if var > 1e-8 else 1.0

    scale_x = spread(1, mean_x)
    scale_level = spread(2, mean_level)
    scale_dip = spread(3, mean_dip)
    columns = []
    for item in packed:
        columns.append((
            1.0,
            (item[1] - mean_x) / scale_x,
            (item[2] - mean_level) / scale_level,
            (item[3] - mean_dip) / scale_dip,
        ))
    beta = [0.0, 0.0, 0.0, 0.0]
    penalty = 3.0
    for _step in range(8):
        gram = [[0.0] * 4 for _ in range(4)]
        rhs = [0.0] * 4
        for row, label, weight in zip(columns, labels, score_w):
            linear = sum(beta[i] * row[i] for i in range(4))
            prob = min(max(_sigmoid(linear), 1e-4), 1.0 - 1e-4)
            ire = prob * (1.0 - prob)
            z_work = linear + (label - prob) / ire
            for i in range(4):
                rhs[i] += weight * ire * row[i] * z_work
                for j in range(4):
                    gram[i][j] += weight * ire * row[i] * row[j]
        for i in range(1, 4):
            gram[i][i] += penalty
        beta = _solve_linear(gram, rhs)
    target_dip = target["shape"].get(-1, 0.0) if target.get("shape") else 0.0
    target_row = (
        1.0,
        (target["x"] - mean_x) / scale_x,
        (math.log(target["p1"] + 1.0) - mean_level) / scale_level,
        (target_dip - mean_dip) / scale_dip,
    )
    probability = _sigmoid(sum(beta[i] * target_row[i] for i in range(4)))
    probability = min(max(probability, 0.05), 0.95)
    ambiguity = 4.0 * probability * (1.0 - probability)
    strong = [item for item in packed if item[0] >= cutoff]
    flat = [item for item in packed if item[0] < cutoff]
    adjusted = []
    for index, hour in enumerate(range(2, 25)):
        window = afternoon_window(hour)
        current = rows[index]
        if window <= 0.0:
            adjusted.append(current)
            continue
        strong_vals = [item[4][hour] for item in strong if hour in item[4]]
        flat_vals = [item[4][hour] for item in flat if hour in item[4]]
        strong_ww = [item[5] for item in strong if hour in item[4]]
        flat_ww = [item[5] for item in flat if hour in item[4]]
        if not strong_vals or not flat_vals:
            adjusted.append(current)
            continue
        strong_med = weighted_quantile(strong_vals, strong_ww, 0.5)
        flat_med = weighted_quantile(flat_vals, flat_ww, 0.5)
        blend = probability * strong_med + (1.0 - probability) * flat_med
        center = current[2]
        location = center + mix * window * (blend - center)
        gap = abs(strong_med - flat_med)
        extra = hedge * window * ambiguity * gap
        half = (current[4] - current[0]) / 2.0
        widened = []
        for quantile in current:
            if half > 1e-8:
                widened.append(location + (quantile - center) * ((half + extra) / half))
            else:
                widened.append(location + (quantile - center))
        for cursor in range(1, 5):
            if widened[cursor] < widened[cursor - 1]:
                widened[cursor] = widened[cursor - 1]
        adjusted.append(widened)
    return adjusted


def _solve_linear(gram, rhs):
    """Gaussian elimination. Returns zeros if the system is singular."""
    size = len(rhs)
    matrix = [gram[row][:] + [rhs[row]] for row in range(size)]
    for col in range(size):
        pivot = col
        for row in range(col, size):
            if abs(matrix[row][col]) > abs(matrix[pivot][col]):
                pivot = row
        matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
        if abs(matrix[col][col]) < 1e-12:
            return [0.0] * size
        scale = matrix[col][col]
        for cursor in range(col, size + 1):
            matrix[col][cursor] /= scale
        for row in range(size):
            if row == col:
                continue
            factor = matrix[row][col]
            for cursor in range(col, size + 1):
                matrix[row][cursor] -= factor * matrix[col][cursor]
    return [matrix[row][size] for row in range(size)]


def fit_hour(analogs, weights, hour, target_x):
    """Robust line of log-ratio on log(pre-peak), blended with a weighted median."""
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
    intercept, slope = huber_fit(xs, ys, ws)
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
    clean_console = target["console"] and not target["early"] and p1 >= 5000
    specialist = specialist_log_quantiles(target, analogs) if clean_console else None
    raw = []
    for row in sorted(asked, key=lambda item: item["hour"]):
        hour = int(row["hour"])
        if specialist is not None:
            fitted = specialist[hour - 2]
        else:
            fitted = fit_hour(analogs, weights, hour, target["x"])
        if fitted is None:
            fitted = [math.log(q / p1) for q in persistence_quantiles(p1, hour)]
        raw.append((hour, fitted))
    bandwidth = 0.0 if specialist is not None else PARAMS["smooth"]
    raw = smooth_log_quantiles(raw, bandwidth)
    for hour, fitted in raw:
        hours.append({"hour": hour, "q": quantiles_to_viewers(p1, fitted)})
    return {"hours": hours}


def smooth_log_quantiles(rows, bandwidth):
    """Pull each hour a little toward its neighbors. Zero bandwidth leaves rows alone."""
    if bandwidth <= 0 or len(rows) < 3:
        return rows
    smoothed = []
    for index, (hour, _) in enumerate(rows):
        acc = [0.0] * 5
        denom = 0.0
        for other, (_, quantiles) in enumerate(rows):
            weight = math.exp(-0.5 * ((index - other) / bandwidth) ** 2)
            denom += weight
            for k in range(5):
                acc[k] += weight * quantiles[k]
        merged = [value / denom for value in acc]
        for k in range(1, 5):
            if merged[k] < merged[k - 1]:
                merged[k] = merged[k - 1]
        smoothed.append((hour, merged))
    return smoothed


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
