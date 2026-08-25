# H₂S levels and odour complaints — reproducible report

The report is **[`REPORT.md`](REPORT.md)**. Start there.

This directory holds the analysis behind it: one standalone Python script per
figure, each reading only **public** URLs (no S3 keys, no `.env`), plus the
generated figures and tables that the report embeds.

## Run it

```bash
uv venv .venv
uv pip install -r requirements.txt
.venv/bin/python run_all.py --manifest
```

~15 s with the inputs cached, a couple of minutes cold. Every script is also
runnable on its own:

```bash
.venv/bin/python fig02_dose_response.py
```

`run_all.py --only fig02_dose_response fig03_complaint_conditions` runs a subset.

## Layout

| Path | What |
|---|---|
| `REPORT.md` | the report |
| `APPENDIX.md` | provenance — figure → script → source URLs. Generated; do not hand-edit |
| `common.py` | data loaders, thresholds, palette, plot style, `SOURCES` |
| `fig*.py`, `tbl*.py` | one script per figure or table; the module docstring is that figure's methods note |
| `run_all.py` | runs everything in report order; `--manifest` rewrites `APPENDIX.md` |
| `figures/`, `tables/` | generated outputs, committed so the report renders on GitHub |
| `data/` | cached raw pulls — **gitignored**, delete to force a refresh |

## Conventions

- **Thresholds and colours mirror `h2s/constants.py`** (5 / 10 / 30 ppb, the
  4-tier palette) so the report and the production view cannot drift apart. If
  the production thresholds move, change them in `common.py` to match.
- **Gap-filled H₂S hours are excluded everywhere.** `load_h2s()` defaults to
  `measured_only=True`. A synthetic value must never be counted as an observed
  exceedance.
- **Complaints come from the county's ArcGIS service, not from the published
  S3 asset.** The published `complaints.csv` carries date-only timestamps, which
  makes the hourly analysis impossible. See the `common.py` docstring and §6.2
  of the report.
- **Nothing is hard-coded that can be read from the deployed system.** The
  feature inventory and every model metric are pulled from the deployed models'
  own `training_report.json`, so the tables cannot go stale. `tbl07_model_features.py`
  fails loudly if production adds a feature that has no description here.

## Adding a figure

1. New `figNN_name.py`, importing `common as C`. Write the methods note as the
   module docstring — that is what `APPENDIX.md` picks up.
2. Use `C.save(fig, "figNN_name")` and `C.save_table(df, "tblNN_name")`.
3. Add it to `STEPS` in `run_all.py`.
4. Re-run `run_all.py --manifest`.

If it needs a new remote input, add it to `C.SOURCES` rather than fetching
inline — that dict is what the provenance appendix is generated from.
