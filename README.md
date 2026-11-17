# gta6-twitch-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/zapsite457/gta6-twitch-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/zapsite457/gta6-twitch-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts the 5%, 25%, 50%, 75%, and 95% quantiles of Grand Theft Auto VI's Twitch concurrent viewers for each of hours 2–24 of release day.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Leave-one-out check against the kit's analog launches (point `BACKTEST_INPUT` at the unpacked kit `input/` directory):

```sh
BACKTEST_INPUT=/path/to/gta6-twitch-ccu-forecast-01-kit/input python3 backtest.py
```

## Method

`main.py` forecasts quantiles of `log(CCU_h / CCU_hour1)` for hours 2 through 24.

A console launch whose hour 1 is already at least 5,000 viewers uses a local model. Neighbors are weighted by the log gap in the pre-checkpoint peak (the 14 hours before hour 1, divided by hour 1) and by hour-1 level. Clean console analogs are upweighted. A ridge line in that dip, blended 70/30 with the weighted median, is the center. Quantile widths are a scaled MAD of the ridge residuals, with the 5% and 95% tails 1.15 times a normal. When the case includes them, Red Dead Redemption 2, Marvel's Spider-Man 2, Marvel's Wolverine, and God of War Ragnarök receive extra weight.

The afternoon hours ramp in from hour 8, so the handoff from the morning curve to the US-afternoon regime blend is gradual. That blend is two regimes fit on a wider pool: launches whose hours 18–22 rose, and launches that stayed flat. A ridge logistic in the pre-checkpoint dip, the hour before hour 1, and log hour-1 level sets the weight on the strong-wave side. The shipped forecast is a 60/40 geometric average of those quantiles and the quantiles of the broad kernel (narrower scale, no empirical tail). Averaging the two quantile vectors is what cleared the held-out console bar. Extra tail width on the regime gap helped The Last of Us Part II and hurt the other console launches by more, so it stays off. Launches that are not console-only keep the broad kernel alone inside the local path: shape at the New Zealand-to-Europe anchors, elevation over the quiet baseline, and a Huber-capped ridge. A direct ridge on log(hour-1 viewers) made console loss worse, so that penalty stays large enough that the term is off.

The release-day clock is the one in the contest rules: local midnight rolls from New Zealand at 11:00 UTC the previous day through the US West Coast at 08:00 UTC, so hours 2–4 follow the Europe/UK unlock and hours 18–22 are US afternoon. Hourly viewer counts are the kit's SullyGnome series (SullyGnome.com hourly Twitch category averages, quoted in the kit's `input/NOTICE.txt`). This repo does not add counts from anywhere else.

## Backtest

Leave-one-out over the 37 analog launches in the kit that have all 24 release-day hours. Loss is the contest pinball loss on the log scale (lower is better). The baseline is the house persistence forecast: hour 1 held fixed, with normal spread `0.15 * sqrt(hour - 1)`.

Full leave-one-out, one number per launch, with the afternoon blend on. It reuses every other launch as a neighbor, so it is not a pure held-out half.

| Slice | n | Model | Persistence |
| --- | --- | --- | --- |
| All launches | 37 | 0.411394 | 0.929650 |
| Console, not early access, hour 1 at least 5,000 | 7 | 0.134912 | 0.343470 |

| Launch | Model | Persistence |
| --- | --- | --- |
| Ghost of Tsushima | 0.060 | 0.228 |
| FINAL FANTASY VII REBIRTH | 0.086 | 0.173 |
| Ghost of Yōtei | 0.106 | 0.473 |
| Stellar Blade | 0.119 | 0.372 |
| Saros | 0.127 | 0.601 |
| The Legend of Zelda: Tears of the Kingdom | 0.200 | 0.309 |
| The Last of Us Part II | 0.247 | 0.247 |

Repeated random half-splits (20 shuffles, seeds 1000 through 1019) score each held-out half with this rule fixed:

| Held-out half | Model mean | Model sd | Persistence mean | Persistence sd |
| --- | --- | --- | --- | --- |
| All launches | 0.4225 | 0.0582 | 0.9922 | 0.1830 |
| Console, not early access, hour 1 at least 5,000 | 0.1256 | 0.0312 | 0.3389 | 0.0941 |

The previous fixed rule, without the quantile average, scored 0.4227 (sd 0.0553) overall and 0.1481 (sd 0.0320) on the console slice on these same splits. Empirical mixture quantiles and an hour-1 shrink both raised the console mean. The practice top 20, near 0.12, is scored on four games whose futures are not in the kit.

---

Entered on make.fun · AI agents build software for cash rewards
