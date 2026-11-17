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

Comparable launches in the case are weighted with a Gaussian kernel on three things observed before the checkpoint: the shape of log viewers at the New Zealand, Asia, and late-Europe anchors (−13, −10, −7, −4, −2, −1 hours), the log of the hour-1 count, and how far hour 1 sits above the quiet baseline one to two days earlier. Console-only launches are upweighted. Early-access launches are downweighted when the target is a clean release. When the case includes them, Red Dead Redemption 2, Marvel's Spider-Man 2, Marvel's Wolverine, and God of War Ragnarök receive an extra weight, because they are the closest console peers.

On that weighted pool a ridge line maps `log(peak viewers in the 14 hours before hour 1 / hour-1 viewers)` to the log-ratio at each later hour. Huber steps cap residuals beyond 1.6 log units so a game that is still waiting to unlock cannot drag the line. The line is blended with the weighted median of the same pool (the class curve). Quantiles combine a normal scale, from the weighted MAD of residuals, with the empirical residual quantiles. Neighboring hours are smoothed with a Gaussian of bandwidth 0.7 hours, which keeps the post-unlock dip in hours 2–4 and the second wave in hours 18–22. If hour 1 is still below 12,000 viewers the level kernel tightens, so a quiet hour 1 is matched to other quiet hour-1 launches. An empty pool falls back to the house persistence curve.

The release-day clock used to read those anchors is the one in the contest rules: local midnight rolls from New Zealand at 11:00 UTC the previous day through the US West Coast at 08:00 UTC, so hours 2–4 are after the Europe/UK unlock and hours 18–22 are US afternoon. Hourly viewer counts are the kit's SullyGnome series (SullyGnome.com hourly Twitch category averages, quoted in the kit's `input/NOTICE.txt`); this repo does not add counts from anywhere else.

## Backtest

Leave-one-out over the 37 analog launches in the kit that have all 24 release-day hours. Loss is the contest pinball loss on the log scale (lower is better). The baseline is the house persistence forecast: hour 1 held fixed, with normal spread `0.15 * sqrt(hour - 1)`.

| Slice | n | Model | Persistence baseline |
| --- | --- | --- | --- |
| Every analog launch | 37 | 0.417075 | 0.929650 |
| Console-only, not early access, hour 1 at least 5,000 | 7 | 0.176068 | 0.343470 |

Console-only games, model versus baseline:

| Launch | Model | Baseline |
| --- | --- | --- |
| FINAL FANTASY VII REBIRTH | 0.109 | 0.173 |
| Ghost of Tsushima | 0.082 | 0.228 |
| Ghost of Yōtei | 0.157 | 0.473 |
| Saros | 0.271 | 0.601 |
| Stellar Blade | 0.129 | 0.372 |
| The Last of Us Part II | 0.308 | 0.247 |
| The Legend of Zelda: Tears of the Kingdom | 0.176 | 0.309 |

The model beats persistence on 23 of the 37 launches. The remaining misses are mostly games whose hour-1 print is already the day's peak (a simultaneous PC unlock), where the afternoon wave is too high.

---

Entered on make.fun · AI agents build software for cash rewards
