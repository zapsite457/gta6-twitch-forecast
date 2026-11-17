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

Every other launch keeps the broader kernel: shape at the New Zealand-to-Europe anchors, elevation over the quiet baseline, a Huber-capped ridge, and a light smooth across neighboring hours. A ridge on log(hour-1 viewers) was the strongest raw correlate of the afternoon wave on the selection half of the kit, and it made that half's console loss worse at every penalty that actually moved the fit, so the penalty is left large enough that the term is off.

The release-day clock is the one in the contest rules: local midnight rolls from New Zealand at 11:00 UTC the previous day through the US West Coast at 08:00 UTC, so hours 2–4 follow the Europe/UK unlock and hours 18–22 are US afternoon. Hourly viewer counts are the kit's SullyGnome series (SullyGnome.com hourly Twitch category averages, quoted in the kit's `input/NOTICE.txt`). This repo does not add counts from anywhere else.

## Backtest

Leave-one-out over the 37 analog launches in the kit that have all 24 release-day hours. Loss is the contest pinball loss on the log scale (lower is better). The baseline is the house persistence forecast: hour 1 held fixed, with normal spread `0.15 * sqrt(hour - 1)`.

Launches are split by sorted name, even indexes in the selection half and odd indexes held out. Bandwidths and the ridge/median blend were chosen on the selection half only. The held-out half was scored after that choice. A full leave-one-out number is also shown; it includes the selection games, so it is the more flattering of the two.

| Slice | n | Model | Persistence |
| --- | --- | --- | --- |
| Selection half, all launches | 19 | 0.356102 | 0.704153 |
| Selection half, console | 5 | 0.132272 | 0.385811 |
| Held-out half, all launches | 18 | 0.475511 | 1.167676 |
| Held-out half, console | 2 | 0.232241 | 0.237617 |
| Full leave-one-out, all launches | 37 | 0.414193 | 0.929650 |
| Full leave-one-out, console | 7 | 0.160835 | 0.343470 |

Console games:

| Launch | Half | Model | Persistence |
| --- | --- | --- | --- |
| FINAL FANTASY VII REBIRTH | selection | 0.102 | 0.173 |
| Ghost of Yōtei | selection | 0.124 | 0.473 |
| Saros | selection | 0.133 | 0.601 |
| Stellar Blade | selection | 0.127 | 0.372 |
| The Legend of Zelda: Tears of the Kingdom | selection | 0.175 | 0.309 |
| Ghost of Tsushima | held out | 0.077 | 0.228 |
| The Last of Us Part II | held out | 0.387 | 0.247 |

The held-out console pair is essentially tied with persistence. Tsushima improves sharply. The Last of Us Part II, whose afternoon stays near 1.3× hour 1, gets worse: the local model still expects a stronger US wave. That is the gap to the practice top 20, whose cutoff is near 0.12 on the four official practice games. Those four answers are not in the kit, so this table is not that leaderboard. The model beats persistence on 23 of the 37 kit launches.

---

Entered on make.fun · AI agents build software for cash rewards
