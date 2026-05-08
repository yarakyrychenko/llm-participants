# Personas

This folder contains the packaged persona datasets and the local build scripts used by `human-ai-eval`.

## Files

- `us_personas.json`
  Default synthetic participant pool used by `load_default_personas()` and by `Lab(...)` when no dataframe is provided.
  This is the lighter 10,000-person US dataset built from `Tianyi-Lab/Personas` and stratified for general simulation use.

- `us_personas_full.json`
  Full uncapped adult-US dataset built locally from `Tianyi-Lab/Personas`.
  This file is larger and is intended for cases where you need the widest possible US pool, such as quota sampling for rarer combinations.
  It is generated locally and ignored by git by default.

- `gss_personas.csv`
  General Social Survey-derived U.S. persona frame for GSS-style prompt templates and quota-sampled study runs.

- `personas.json`
  Smaller hand-built multi-country persona dataset used for non-US or lightweight cross-country workflows.

- `prompts.json`
  Prompt templates and related persona-prompt assets.

- `build_us_personas.py`
  Builder for the US persona datasets sourced from `Tianyi-Lab/Personas`.

- `build_personas.py`
  Builder for the smaller multi-country `personas.json` dataset.

- `extract_attributes.py`
  Helpers for extracting structured persona attributes from free-text profiles.

## Loaders

Use these helpers from Python:

```python
from simulate import load_default_personas, load_gss_personas, load_us_personas, load_us_personas_full, load_personas
```

- `load_default_personas()`
  Returns the default `us_personas.json` dataframe.

- `load_us_personas()`
  Returns the default 10,000-person US dataframe explicitly.

- `load_us_personas_full()`
  Returns the locally generated full US dataframe from `us_personas_full.json`.
  This will raise `FileNotFoundError` until you build that file locally.

- `load_gss_personas()`
  Returns the packaged GSS persona dataframe from `gss_personas.csv`, preserving GSS fields such as `sex`, `degree`, `partyid`, and `polviews` while adding common aliases such as `gender`, `education`, `party`, and `political_orientation`.

- `load_personas()`
  Returns the multi-country dataframe from `personas.json`.

## Local Builds

Recommended environment:

```bash
conda activate human-ai-eval
```

Build the default 10,000-person US dataset:

```bash
python -m simulate.personas.build_us_personas
```

Build the full uncapped US dataset:

```bash
python -m simulate.personas.build_us_personas --full
```

Optionally choose a custom output filename:

```bash
python -m simulate.personas.build_us_personas --full --output my_us_personas_full.json
```

Build the smaller multi-country dataset:

```bash
python -m simulate.personas.build_personas
```

## US Dataset Distributions

### `us_personas.json` (10,000 stratified US personas)

- Race: `White 5,950`, `Hispanic 1,900`, `Black 1,299`, `Asian 600`, `Other 251`
- Gender: `Woman 5,069`, `Man 4,931`
- Age group: `18-29 2,199`, `30-44 2,500`, `45-59 2,400`, `60+ 2,901`
- Region: `Northeast 1,893`, `Midwest 2,345`, `South 3,364`, `West 2,398`

### `us_personas_full.json` (48,000 full US personas)

- Race: `White 31,931`, `Hispanic 5,152`, `Black 4,676`, `Asian 1,787`, `Other 4,454`
- Gender: `Woman 24,508`, `Man 23,492`
- Age group: `18-29 9,777`, `30-44 12,385`, `45-59 14,858`, `60+ 10,980`
- Region: `Northeast 9,000`, `Midwest 12,000`, `South 16,000`, `West 11,000`

## Notes

- `build_us_personas.py` requires the Hugging Face `datasets` package and downloads from `Tianyi-Lab/Personas`.
- The full US file is much larger than the default 10k file.
- Even the full US dataset still reflects the source distribution, so very aggressive party quotas may remain infeasible.
