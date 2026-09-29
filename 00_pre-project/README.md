# 00 · Pre-project: getting the data and a first look

Work done during idea selection, before the project itself starts: downloading PISA 2022, converting it to a usable format, and a tour of every file so the group can decide what questions to ask.

It does not contain research questions, hypotheses or models; those start in `01_topic/`. It was prepared with AI assistance; see [AI_USE.md](AI_USE.md).

## Notebooks

| Notebook | What it does | Run time |
|---|---|---|
| `notebooks/01_extract.ipynb` | Downloads the 8 SPSS zips from the OECD, unzips them, converts every file to Parquet (split into parts under 90 MB), and builds `data/data_dictionary.csv` and `data/inventory.csv` | ~10 min plus download; **you don't need to run it**, its output is already in `data/` |
| `notebooks/02_explore.ipynb` | Guided tour of every file, using the OECD's weighting method, with charts | ~1 min |
| `notebooks/_build_02.py` | Source that generates `02_explore.ipynb`. To change 02, edit this file and run `python _build_02.py` | |

### 02_explore at a glance

Section 0 reproduces the OECD's published averages (math 472, reading 476, science 485; Singapore 575 in math). If that check fails, the notebook stops. Sections 1 to 17 then walk through every file:

- coverage, scores and proficiency levels
- math sub-scales, gender gaps, and socio-economic status (equity)
- phones in class, COVID-19 school closures
- schools, teachers, creative thinking, financial literacy
- individual test questions, questionnaire timing, and ESCS trends since 2012

Section 17 profiles one country: set `FOCUS` at the top (default `'ARE'`, the UAE). Section 18 writes `outputs/key_findings.md`. Everything is descriptive: associations, not causes. Some parts (item statistics, teacher file, ESCS trend) are unweighted because those files carry no student weights; the notebook labels each one.

## Outputs

```
outputs/key_findings.md        headline numbers, computed by notebook 02
outputs/country_summary.csv    headline measures for all 80 countries/economies
outputs/item_statistics.csv    % correct and median time for every test question
outputs/figures/               the charts from notebook 02
```

## Rebuilding the data from the OECD source

Only needed to check the extraction or to recover the original missing codes. Run `01_extract.ipynb`. It downloads from <https://webfs.oecd.org/pisa2022/index.html> (about 2 GB, about 9 GB unzipped into `data/raw/`, which git ignores). It then rebuilds everything in `data/`.
