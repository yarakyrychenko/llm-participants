# Agents

## Runtime Context

- Primary language: Python
- Preferred environment: the Conda environment from [environment.yml](/llm-participants/environment.yml)
- Expected activation:
  - `conda env create -f environment.yml`
  - `conda activate human-ai-eval`
- Preferred install workflow for notebooks and scripts outside the repo root:
  - `pip install -e .`

## Working Conventions For Agents

- Prefer `python` inside the active `human-ai-eval` conda env, or `conda run -n human-ai-eval python ...` when running non-interactively.
- If a local shell does not have `python`, use `python3` outside the env or `conda run -n human-ai-eval python`.
- After a feature, behavior change, or API change is implemented and validated successfully, do a docs sweep before finishing. Update any affected user-facing documentation such as `README.md`, example notebooks, inline examples, and related plan notes so the documented behavior matches the code.
- In the final handoff to the user, explicitly point out the documentation updates you made. If any docs or notebooks were intentionally left unchanged, say that plainly and note the remaining follow-up.

## Small Test Commands

Use small targeted checks before attempting large notebook or API runs.

- Compile touched modules:
  - `conda run -n human-ai-eval python -m py_compile simulate/survey.py simulate/qsf.py simulate/lab.py simulate/__init__.py simulate/personas/__init__.py`

- Inspect parsed scale names:
  - `conda run --no-capture-output -n human-ai-eval python - <<'PY'`
    `from simulate.survey import Survey`
    `print(Survey('example/survey/human.qsf').scales.keys())`
    `PY`

- Smoke-test flow visualization:
  - `MPLCONFIGDIR=/tmp/mpl conda run --no-capture-output -n human-ai-eval python - <<'PY'`
    `from simulate import visualize_survey_flow`
    `from simulate.survey import Survey`
    `survey = Survey('example/survey/human.qsf')`
    `fig, ax = visualize_survey_flow(survey, figsize=(12, 18))`
    `fig.canvas.draw()`
    `print('FLOW_VIZ_OK')`
    `PY`

- Smoke-test block exclusion:
  - `MPLCONFIGDIR=/tmp/mpl conda run --no-capture-output -n human-ai-eval python - <<'PY'`
    `from simulate.survey import Survey`
    `survey = Survey('example/survey/survey_hybrid.qsf')`
    `print('Consent' in survey.get_scales(None))`
    `survey.set_excluded_blocks(['Info + Consent'])`
    `print('Consent' in survey.get_scales(None))`
    `PY`

- Smoke-test answer normalization:
  - `MPLCONFIGDIR=/tmp/mpl conda run --no-capture-output -n human-ai-eval python - <<'PY'`
    `from simulate.survey import Survey`
    `survey = Survey('example/survey/human.qsf')`
    `scale = survey.scales['Consent']`
    `print(survey._normalize_assessment_dict({'answer': 'Yes, I agree'}, scale))`
    `PY`

## Live API Smoke Tests

- If the user has supplied API credentials in the conda env, a minimal real-API smoke test is acceptable before claiming end-to-end success.
- Keep live tests small: 1 participant, 1-2 scales, low turn counts.
- Prefer setting `MPLCONFIGDIR=/tmp/mpl` to avoid matplotlib cache noise in sandboxed environments.
- When testing hybrid intervention flow, include the actual top-level intervention config shape with `defaults` and `conditions`.
- For the bridging study, prefer `conda run -n human-ai-eval python studies/bridging/01_generate_bridging_data.py --participant-limit 1` as the smallest end-to-end pilot before larger runs.
- For larger bridging runs, prefer explicit slices such as `--participant-start 0 --participant-end 30`; the generator now saves range-specific outputs like `bridging_raw_results_n0_m30.json` so batches do not overwrite each other.