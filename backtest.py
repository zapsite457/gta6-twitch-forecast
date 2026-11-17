"""Leave-one-out pinball backtest on the kit's analog launches.

Each comparable launch with a full release-day curve is held out in turn. Hour 1 and
the pre-checkpoint history are shown to main.forecast_case; hours 2-24 are scored
against the same log-scale pinball loss the contest uses. The persistence baseline
is the house formula (hour 1 held, normal spread 0.15 * sqrt(hour - 1)).

Point BACKTEST_INPUT at the unpacked kit's input directory if it is not in the
default location. Nothing here is read by the grader.
"""

import json
import math
import os
import sys
from pathlib import Path

import main

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_SCORES = [-1.645, -0.674, 0.0, 0.674, 1.645]
PERSIST_SIGMA = 0.15


def default_input():
    env = os.environ.get("BACKTEST_INPUT")
    if env:
        return Path(env)
    here = Path(__file__).resolve().parent
    candidates = [
        here / "kit-input",
        Path("/tmp/makefun/gta6-twitch-ccu-forecast-01-kit/input"),
    ]
    for path in candidates:
        if (path / "p00.in").is_file():
            return path
    return candidates[-1]


def pinball(actual, quantile, tau):
    y = actual if actual >= 1 else 1.0
    q = quantile if quantile > 0 else 1e-12
    gap = math.log(y) - math.log(q)
    return 2.0 * max(tau * gap, (tau - 1.0) * gap)


def baseline_quantiles(p1, hour):
    spread = PERSIST_SIGMA * math.sqrt(hour - 1)
    return [p1 * math.exp(z * spread) for z in Z_SCORES]


def loss_of(p1, actual_by_hour, quantiles_by_hour):
    total = 0.0
    count = 0
    for hour in range(2, 25):
        actual = actual_by_hour[hour]
        for tau, quantile in zip(LEVELS, quantiles_by_hour[hour]):
            total += pinball(actual, quantile, tau)
            count += 1
    return total / count


def release_points(comp):
    day = main.parse_time(comp["day_start"])
    points = {}
    for row in comp["hourly"]:
        if row.get("viewers") is None:
            continue
        off = main.hour_offset(row["from"], day)
        points[off] = float(row["viewers"])
    return points


def build_case(comp, others):
    """A grader-shaped case whose future hours are hidden in history."""
    day = main.parse_time(comp["day_start"])
    points = release_points(comp)
    hourly = []
    for row in comp["hourly"]:
        off = main.hour_offset(row["from"], day)
        if off <= 0 and row.get("viewers") is not None:
            hourly.append({
                "from": row["from"],
                "viewers": row["viewers"],
                "channels": row.get("channels"),
                "reads": row.get("reads") or 1,
            })
    hourly.sort(key=lambda row: row["from"])
    p1 = points[0]
    notes = (
        f"{comp['name']}, released on {comp.get('platforms') or ''}; "
        f"unlock: {comp.get('unlock') or ''}"
    )
    return {
        "schema": "makefun/twitch-ccu-case/v1",
        "id": "loo",
        "game": {"name": comp["name"], "twitch_id": "0"},
        "day_start": comp["day_start"],
        "checkpoint": comp["day_start"],
        "levels": LEVELS,
        "known": [{
            "hour": 1,
            "from": comp["day_start"],
            "until": comp["day_start"],
            "viewers": p1,
            "channels": None,
            "reads": 1,
        }],
        "forecast": [{"hour": hour, "from": "", "until": ""} for hour in range(2, 25)],
        "history": [],
        "history_hourly": hourly,
        "streams": None,
        "categories_hourly": None,
        "comparables": others,
        "notes": notes,
    }


def classify(comp, p1):
    unlock = (comp.get("unlock") or "").lower()
    plat = (comp.get("platforms") or "").lower()
    early = ("early access" in unlock) or ("early-access" in unlock)
    console = ("pc" not in plat) and ("stadia" not in plat)
    gta_like = console and (not early) and p1 >= 5000
    return gta_like


def main_backtest():
    folder = default_input()
    case_path = folder / "p00.in"
    if not case_path.is_file():
        print(f"kit input not found at {folder}", file=sys.stderr)
        return 1
    case = json.loads(case_path.read_text(encoding="utf-8"))
    launches = []
    for comp in case["comparables"]:
        if not main.is_launch_analog(comp):
            continue
        points = release_points(comp)
        if not all(off in points for off in range(0, 24)):
            continue
        if points[0] < 1:
            continue
        launches.append(comp)
    launches.sort(key=lambda comp: comp["name"])

    rows = []
    model_losses = []
    base_losses = []
    gta_model = []
    gta_base = []
    for held in launches:
        others = [comp for comp in launches if comp["name"] != held["name"]]
        synthetic = build_case(held, others)
        forecast = main.forecast_case(synthetic)
        points = release_points(held)
        p1 = max(points[0], 1.0)
        actual = {hour: points[hour - 1] for hour in range(2, 25)}
        predicted = {row["hour"]: row["q"] for row in forecast["hours"]}
        base = {hour: baseline_quantiles(p1, hour) for hour in range(2, 25)}
        model_loss = loss_of(p1, actual, predicted)
        base_loss = loss_of(p1, actual, base)
        gta_like = classify(held, p1)
        rows.append((model_loss, base_loss, gta_like, held["name"], p1))
        model_losses.append(model_loss)
        base_losses.append(base_loss)
        if gta_like:
            gta_model.append(model_loss)
            gta_base.append(base_loss)

    def mean(values):
        return sum(values) / len(values)

    print(f"launches {len(rows)}")
    print(f"model_all {mean(model_losses):.6f}")
    print(f"baseline_all {mean(base_losses):.6f}")
    print(f"model_console {mean(gta_model):.6f}")
    print(f"baseline_console {mean(gta_base):.6f}")
    print(f"console_n {len(gta_model)}")
    rows.sort(key=lambda row: row[0], reverse=True)
    print("worst")
    for model_loss, base_loss, gta_like, name, p1 in rows[:8]:
        flag = "console" if gta_like else "other"
        print(f"  {model_loss:.3f} base {base_loss:.3f} {flag} h1={p1:.0f} {name}")
    print("console")
    for model_loss, base_loss, gta_like, name, p1 in sorted(rows, key=lambda row: row[3]):
        if gta_like:
            print(f"  {model_loss:.3f} base {base_loss:.3f} h1={p1:.0f} {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main_backtest())
