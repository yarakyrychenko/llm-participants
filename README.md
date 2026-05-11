# LLM Participants

A research framework for simulating human survey responses from Qualtrics files with Large Language Models. Load a survey, create participant profiles from real demographic data, let LLMs role-play as those participants, and compare simulated responses to ground-truth human answers (when available).

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Project Structure](#project-structure)
5. [Core Concepts](#core-concepts)
6. [Loading Surveys](#loading-surveys)
7. [Running Simulations](#running-simulations)
8. [Analyzing Results](#analyzing-results)
9. [Prompt Engineering with Researcher](#prompt-engineering-with-researcher)
10. [API Reference](#api-reference)
11. [Configuration](#configuration)
12. [Examples](#examples)

---

## Overview

**human-ai-eval** answers the question: *how well can an LLM simulate a specific human respondent in a survey?*

Persona data comes in a few shapes: `simulate/personas/us_personas.json` is the default 10,000-person US pool, `simulate/personas/us_personas_full.json` is the larger locally built full US pool, and `simulate/personas/personas.json` is the smaller multi-country dataset. See [`simulate/personas/README.md`](/Users/yara/GitHub/human-ai-eval/simulate/personas/README.md) for details.

Given a participant's demographic profile (age, gender, political party, education, etc.) and a survey instrument, the framework:

1. **Simulates** the participant by prompting an LLM with a system message encoding their profile.
2. **Administers** the survey — presenting each scale as a structured prompt and parsing the model's JSON response.
3. **Evaluates** simulation fidelity by comparing AI answers to the participant's real survey responses.
4. **Supports experiments** where a chatbot converses with the simulated participant as part of the survey flow or alongside prior real-conversation context to measure attitude change.

The framework is model-agnostic and works with any OpenAI-compatible API (OpenAI, OpenRouter, Together AI, local models, etc.).

---

## Installation

Recommended setup uses the validated conda environment:

```bash
conda env create -f environment.yml
conda activate human-ai-eval
pip install -e .
```

To persist an API key inside your conda environment, you can set it as an environment variable on the env itself:

```bash
conda activate myenv
conda env config vars set OPENROUTER_API_KEY="your_api_key_here"
conda deactivate
conda activate myenv
```

If you prefer pip:

```bash
git clone https://github.com/yarakyrychenko/llm-participants.git
cd llm-participants
pip install -e .
```

---

## Quick Start

This example uses the packaged GSS persona frame and the trust survey, so it does
not require any separate example data files.

```python
import os
from openai import OpenAI
from simulate import load_gss_personas, load_prompts
from simulate.survey import Survey
from simulate.lab import Lab

# 1. Connect to an LLM API
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

# 2. Load the trust survey
survey = Survey("trust/survey_trust_new.qsf")
survey.set_excluded_blocks(["Consent Form", "Filter", "Attention Check"])

# 3. Load adult GSS personas
df = load_gss_personas()

# 4. Run a survey simulation on 5 GSS profiles
lab = Lab(client, df, user_model="openai/gpt-4o-mini")
participants = lab.sample(5)

# load_prompts handles any prompt file format → always returns {name: text}
prompts = load_prompts("simulate/personas/prompts.json")
participant_prompt = {"gss_prompt": prompts["gss_prompt"]}

results = lab.run_survey(participants, participant_prompt, survey)

# 5. Inspect results
results_df = lab.format_results(results, survey)
print(results_df.head())
```

To run the trust-study generator directly on the first 10 GSS profiles:

```bash
conda run --no-capture-output -n human-ai-eval python trust/01_generate_trust_data.py --participant-range 0:10 --chunk-size 10
```

---

## Project Structure

```
llm-participants/
├── simulate/                          # Core package
│   ├── __init__.py                    # Public exports
│   ├── agent.py                       # LLM agent classes (User, Chatbot, BaseBot)
│   ├── lab.py                         # Simulation orchestration + save/load helpers
│   ├── parallel.py                    # Multithreading decorator with retry logic
│   ├── qsf.py                         # Qualtrics .qsf parsing + flow visualization
│   ├── researcher.py                  # Prompt-engineering and evaluation helpers
│   ├── survey.py                      # Flow-aware survey execution
│   ├── utils.py                       # Constants, distance metrics, plotting helpers
│   ├── data/prompts/researcher/
│   │   └── default.json               # Meta-prompts used by Researcher
│   └── personas/
│       ├── __init__.py                # Persona-loading helpers
│       ├── us_personas.json           # Default synthetic participant pool
│       ├── us_personas_full.json      # Optional full US pool built locally from Tianyi-Lab/Personas
│       ├── personas.json              # Alternate persona dataset
│       ├── gss_personas.csv           # GSS Personas (9000)
│       └── *.py                       # Persona build/extraction scripts
│
├── trust/                             # Trust megastudy
├── environment.yml                    # Validated conda environment
├── AGENTS.md                          # Agent workflow guidance for this repo
└── README.md
```

---

## Core Concepts

### Scale types

Every survey is a list of **scales**. Each scale has a `type` that controls how responses are structured and scored:

| Type | Description | Example |
|---|---|---|
| `numeric` | Slider — model picks a number in a min–max range | Feeling thermometer 0–100 |
| `multiple choice` | Select one option per item from a fixed list | Likert matrix |
| `ranking` | Rank a list of items from most to least preferred | Policy priorities |
| `free text` | Return a short open-ended response as a JSON string value | Feedback or reflection prompt |

### Context modes

`context_mode` controls whether a prior real conversation is injected before the flow-backed survey runs:

| Context Mode | Description |
|---|---|
| `"none"` | Run the survey flow with no prior real conversation |
| `"real"` | Inject a real prior conversation from the dataframe `messages` field before survey execution |

### Participant profile tags

Participant system prompts can embed placeholder tags that are substituted at runtime from the dataframe:

```
[AGE]  [GENDER]  [PARTY]  [RACE]  [EDUCATION]  [POLITICAL_ORIENTATION]
```

Example:
```
You are a [AGE]-year-old [GENDER] who identifies as [PARTY].
Your highest level of education is [EDUCATION].
```

### Distance metrics

Simulation quality is measured with normalised distances (both in `[0, 1]`, lower = better):

- **Absolute distance** — for `numeric` and `multiple choice` scales. Mean absolute difference between simulated and human responses, normalised by the response range.
- **Footrule distance** — for `ranking` scales. Normalised Spearman footrule distance between two rank orderings.
- `free text` scales are preserved in outputs but currently do not receive an automatic distance metric or `*_sim_error` column.

---

## Loading Surveys

### From a Qualtrics .qsf export

Export your survey from Qualtrics via *Survey → Tools → Export Survey*, then load it directly:

```python
from simulate.survey import Survey

survey = Survey("example/qsf/survey.qsf")
```

**Supported Qualtrics question types:**

| Qualtrics type | Mapped to |
|---|---|
| Slider (HBAR / HSLIDER) | `numeric` |
| Matrix (Likert / SingleAnswer) | `multiple choice` |
| Multiple Choice (SAVR / MAVR / DL) | `multiple choice` |
| Rank Order (DND) | `ranking` |
| Text Entry (TE) with participant-facing prompt text | `free text` |

Questions of type `DB` (descriptive text), `Timing`, `Meta`, and `CS` are not emitted as scales. For `TE` questions, the parser now promotes participant-facing prompts into `free text` scales, while hidden or workflow-oriented text-entry fields (for example blank prompt slots or auto-filled embedded values) remain non-administered flow steps.

Saved item keys are taken directly from Qualtrics `DataExportTag` identifiers (e.g. `Q1_1`, `Q3_2`) so formatted result columns remain stable and machine-readable. For matrix questions, row item keys honor Qualtrics `ChoiceDataExportTags` when present. Multiple-choice and matrix answer values honor Qualtrics `RecodeValues`, so author-defined exports such as `0`-to-`10` scales are preserved instead of exposing Qualtrics' internal choice IDs. Option prompts follow Qualtrics display ordering, including advanced fixed choice order metadata when present. The model does not see those internal keys during survey administration; prompts use neutral aliases such as `item_1`, `item_2`, and the runtime maps the returned JSON back to the original keys before scoring or saving results.

Participant-facing Qualtrics piped text is resolved at runtime when the needed value is available. The runtime supports selected-choice pipes such as `${q://QID123/ChoiceGroup/SelectedChoices}` and embedded-field pipes such as `${e://Field/Condition}`; unresolved tokens are left unchanged and recorded as flow warnings rather than guessed.

Supported branch operators include selected-choice and equality checks (`Selected`, `NotSelected`, `EqualTo`, `NotEqualTo`), numeric comparisons (`GreaterThan`, `GreaterThanOrEqual`, `LessThan`, `LessThanOrEqual`), emptiness checks (`Empty`, `NotEmpty`), and containment checks (`DoesContain`, `DoesNotContain`). Selected-choice branches are evaluated against the same `RecodeValues` used for parsed answer outputs, so attention checks and eligibility gates stay aligned with the values the model returns. Containment branches respect Qualtrics `IgnoreCase` metadata for text responses and embedded fields.

### From a JSON file

A survey JSON is a list of scale objects. Keys in `items` must be machine-readable identifiers (alphanumeric + underscores only):

```json
[
  {
    "name": "feel_therm_dem",
    "type": "numeric",
    "question": "How would you rate your feelings toward Democrats?",
    "items": { "feel_therm_dem": "Feeling thermometer" },
    "options": { "min": 0, "max": 100, "minLabel": "Cold", "maxLabel": "Warm" }
  },
  {
    "name": "climate_knowledge",
    "type": "multiple choice",
    "question": "For each statement, select the most accurate answer.",
    "items": {
      "Q1_1": "Percentage of climate scientists who agree on warming",
      "Q1_2": "Main cause of recent global warming"
    },
    "options": {
      "Less than 50%": 1, "About 60%": 2, "About 97%": 3, "About 99%": 4
    },
    "answers": { "Q1_1": 4, "Q1_2": 3 }
  },
  {
    "name": "reflection",
    "type": "free text",
    "question": "What do you think this study was trying to investigate?",
    "items": { "reflection": "What do you think this study was trying to investigate?" },
    "options": {}
  }
]
```

The optional `"answers"` field provides ground-truth responses so that `Survey.get_score()` can compute accuracy.

---

## Running Simulations

### `.qsf` runtime with default personas and intervention config

```python
import os
import pandas as pd
from openai import OpenAI

from simulate import load_default_personas, load_prompts, visualize_survey_flow
from simulate.survey import Survey
from simulate.lab import Lab

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

df = load_default_personas()
survey = Survey("example/survey/survey_hybrid.qsf")
lab = Lab(client, df, user_model="openai/gpt-4.1-mini", seed=42)

participant_prompts = load_prompts("example/prompts/participant/suggested.json")
participant_prompt = {"profile_full_instructions": participant_prompts["profile_full_instructions"]}

intervention_config = {
    "defaults": {
        "first_speaker": "participant",
        "user_start_qs": "Ask, request, or talk to the chatbot about something you consider politically important as a [PARTY] participant.",
        "base_url": "https://openrouter.ai/api/v1",
        "api_secret": "OPENROUTER_API_KEY",
        "model": "deepseek/deepseek-chat-v3-0324",
    },
    "conditions": {
        "1": {
            "label": "base",
            "prompt_path": "example/prompts/chatbot/baseline.txt",
        },
        "2": {
            "label": "bridging",
            "prompt_path": "example/prompts/chatbot/bridging.txt",
        },
        "3": {
            "label": "personalized",
            "prompt_path": "example/prompts/chatbot/personalization.txt",
        },
        "4": {
            "label": "personalized_bridging",
            "prompt_path": "example/prompts/chatbot/pers_bridge.txt",
        },
        "5": {
            "label": "true control",
            "base_url": None,
            "model": None,
            "api_secret": None,
            "text": "Please proceed without a chatbot conversation for this condition.",
            "collect_participant_opener": True,
            "completion_code": "TRUE_CONTROL",
        },
    },
}

# Tags like [PARTY], [AGE], and [RACE] are now filled in before
# intervention prompt text is executed. This applies to inline `text`,
# inline `system_prompt`, prompt files loaded through `prompt_path`,
# and opener strings such as `user_start_qs` and
# `chatbot_opening_prompt`.
#
# If you set `"system_prompt": ""` explicitly, the chatbot intervention
# is sent with no system message at all. If you omit both `system_prompt`
# and `prompt_path`, the runtime also sends no system message by default.
#
# Chatbot interventions now keep balanced turn counts. `n_turns` means turns
# per side, so both the participant and the chatbot speak `n_turns` times. If
# `"first_speaker": "participant"`, the chatbot gives the final reply. If
# `"first_speaker": "chatbot"`, the chatbot opens with `chatbot_opening_prompt`
# and the participant gives the final reply.
#
# For a true-control path with no chatbot API call, set
# `"collect_participant_opener": True` to let the participant produce
# one opening question and then stop with the control text instead of
# continuing the chatbot exchange.

# Inspect the parsed flow before running
visualize_survey_flow(survey, figsize=(16, 28));

# See where block randomizers appear in the flow and whether they use
# Qualtrics "Evenly Present Elements"
randomizers_df = pd.DataFrame(survey.describe_flow_randomizers())
print(randomizers_df[["flow_id", "path", "subset", "option_count", "even_presentation", "options"]])

# Optionally remove blocks once inspection is complete
survey.set_excluded_blocks(["Info + Consent"])

participants = lab.sample(5)

# Preview the planned counts for even-presentation randomizers before execution
plan_df = pd.DataFrame(
    lab.describe_even_presentation_plan(
        sampled_users=participants,
        user_system_texts=participant_prompt,
        survey=survey,
    )
)
print(plan_df[["flow_id", "path", "planned_counts"]])

# Optionally override the exact counts you want for a single
# even-presentation randomizer. If your survey has multiple such
# randomizers, pass {"FLOW_ID": {...counts...}} instead.
requested_condition_counts = {
    "Control LLM": 2,
    "Bridging LLM": 2,
    "Personalized LLM": 3,
    "Personalized Bridging LLM": 2,
    "True Control": 3,
}

results = lab.run_survey(
    sampled_users=participants,
    user_system_texts=participant_prompt,
    survey=survey,
    intervention_overrides=intervention_config,
    even_presentation_counts=requested_condition_counts,
)

results_df = lab.format_results(results, survey)
print(results_df[["run_id", "id", "survey_chatbot_model", "survey_flow_trace", "survey_intervention_messages"]].head())
```

If you want to use the model provider's default temperature for participant simulation, pass `user_temperature=None` when constructing `Lab(...)`. You can also pin the participant completion budget with `user_max_tokens=...`.

### Quota sample personas before running

If you want the synthetic participant pool to match specific demographic margins, use `lab.quota_sample(...)` instead of `lab.sample(...)`.

```python
quota_participants = lab.quota_sample(
    200,
    quotas={
        "party": {"Republican": 0.5, "Democrat": 0.5},
        "gender": {"Woman": 0.5, "Man": 0.5},
    },
)
```

Quota targets can be specified either as proportions that sum to `1.0` within each column or as exact counts that sum to `n` within each column.

`quota_sample(...)` enforces exact marginal counts without replacement and raises a clear error if the available dataframe cannot satisfy them. For example, the bundled `simulate/personas/us_personas.json` pool currently contains far fewer Republicans than Democrats, so a request for `1500` participants with `50% Republican` and `50% Democrat` is infeasible against that default dataframe and will fail unless you provide a different participant pool.

If you want the widest available US pool, build the full uncapped Tianyi-Lab dataset locally:

```bash
conda run -n human-ai-eval python -m simulate.personas.build_us_personas --full
```

That writes `simulate/personas/us_personas_full.json`. You can then load it with:

```python
from simulate import load_us_personas_full

df = load_us_personas_full()
lab = Lab(client, df, user_model="openai/gpt-4.1-mini")
```

If you exclude blocks, scales, or specific questions, restart from the survey object and then call `run_survey(...)`. By default, excluded blocks and excluded scales are also removed from the default scale list used for execution.

By default, `Survey(...)` now preserves prior survey text, display blocks, intervention text, and previous answers in the active agent context across the whole survey. If you want the older behavior that clears context between scale questions, pass `preserve_context=False`.

For flow-driven surveys, the raw result payload and formatted dataframe keep a top-level `survey_user_messages` field with the simulator-visible message history for the entire run. Conversational interventions additionally keep a dedicated `survey_intervention_messages` field with the turn-by-turn participant/chatbot transcript for each executed intervention. Each intervention record includes both `user_messages`, which preserves the simulator-visible message history at the end of that intervention, and `chatbot_messages`, which preserves the chatbot-side message history exactly as sent to the chatbot API, including the system prompt and the alternating user/assistant turns from the chatbot's perspective. Embedded survey-flow assignments are materialized both as nested metadata and as explicit dataframe columns such as `survey_embedded_Condition` and `survey_embedded_label_Condition`. Each run also gets a deterministic `run_id`, and when a single chatbot-config intervention is used the effective chatbot settings are surfaced in columns such as `survey_chatbot_model`, `survey_chatbot_temperature`, `survey_chatbot_max_tokens`, `survey_chatbot_first_speaker`, and `survey_chatbot_prompt_source`. Per-scale `*_user_memory` columns are no longer emitted because that context is already covered by `survey_user_messages`. Per-scale `*_score` columns are emitted only for scales that actually have answer keys and can be scored.

Parallel batch execution is best effort by default. If a worker crashes outside the normal per-participant error-handling path, the batch call returns the successful rows and emits a `RuntimeWarning` summarizing how many threaded tasks were dropped. For `Lab.run_survey(...)`, ordinary participant-level runtime failures are still captured directly in the raw results as rows with `error`, `error_type`, and `traceback`; `lab.format_results(...)` then removes those error rows before building the dataframe.

When you run a flow-backed survey against real human data that already contains an observed condition assignment (for example a `condition` column), the runtime now replays that participant into the matching condition branch instead of re-randomizing them. If no condition is present, seeded survey randomization still applies normally. For Qualtrics `BlockRandomizer` nodes that use `Evenly Present Elements`, `Lab.run_survey(...)` now allocates those branches across the full batch before threaded execution so synthetic runs stay balanced overall while remaining deterministic for a fixed seed.

You can inspect those randomizers before execution with `survey.describe_flow_randomizers()`. If you also want to know the exact pre-run element counts for a specific batch, call `lab.describe_even_presentation_plan(...)` with the sampled participant IDs and prompt dict you plan to pass into `run_survey(...)`. To force a specific condition mix, pass `even_presentation_counts=...` into either `describe_even_presentation_plan(...)` or `run_survey(...)`. For a survey with one even-presentation randomizer, you can pass a simple option-to-count dict such as `{"Control LLM": 2, "Bridging LLM": 2, "Personalized LLM": 3, "Personalized Bridging LLM": 2, "True Control": 3}`. When there are multiple even-presentation randomizers, pass a mapping from `flow_id` to those per-option count dicts.

### Inspect and exclude blocks, scales, or questions before running

For `.qsf` surveys, you can inspect the parsed block inventory and remove blocks by block ID or exact block description:

```python
import pandas as pd

survey = Survey("example/survey/survey_hybrid.qsf")
pd.DataFrame(survey.describe_flow_blocks())[
    ["block_id", "description", "question_count", "runtime_summary", "excluded"]
]

survey.set_excluded_blocks(["BL_3sWjIstItGpOrwG", "Info + Consent"])
```

Excluded blocks are recorded in the flow trace as `skipped_block` with reason `excluded_by_user`.

You can also suppress individual scales or specific flow questions while still running the rest of the containing block:

```python
survey.set_excluded_scales(["ChatbotCode"])
survey.set_excluded_questions(
    [
        "ChatbotCodeTextTimer",
        "After you have submitted your interaction, you will be able to click the “Next page” button. This button will only appear after 3 minutes.",
    ]
)
```

`set_excluded_questions(...)` accepts question IDs, export tags, scale names, exact descriptions, or exact prompt text. Excluded questions are recorded in the flow trace as `skipped_question` with reason `excluded_by_user`.

### Visualize the parsed survey flow

Use the flow visualizer to inspect the ingested `.qsf` before running simulations:

```python
from simulate import visualize_survey_flow

survey = Survey("example/survey/human.qsf")
visualize_survey_flow(survey, figsize=(16, 28));
```

The diagram is a static tree of the parsed flow. It color-codes blocks, branches, randomizers, embedded-data nodes, and survey endings. Randomizer labels also note when the source `.qsf` requested Qualtrics even presentation.

### Validate randomized flow replay locally

You can run a no-API validation script that checks seeded randomization replay and block exclusion behavior:

```bash
MPLCONFIGDIR=/tmp/mpl conda run --no-capture-output -n human-ai-eval \
  python scripts/validate_qsf_randomization.py
```

The script verifies that:

- the same seed reproduces the same randomized trace,
- different seeds change at least one randomizer selection,
- excluded blocks appear as `skipped_block` and do not execute.

### Human-data evaluation with a matched QSF

The repository includes a matched pair:

- `example/survey/human.qsf`
- `example/data/human.json`

You can run the survey against the human-backed dataframe and compute mean simulation error on a selected subset of scales:

```python
import pandas as pd

human_df = pd.read_json("example/data/human.json")
human_survey = Survey("example/survey/human.qsf")
human_lab = Lab(client, human_df, user_model="openai/gpt-4.1-mini", seed=42)

evaluation_scales = ["feel_therm_dem", "feel_therm_rep"]
results = human_lab.run_survey(
    sampled_users=human_lab.sample(5),
    user_system_texts=participant_prompt,
    survey=human_survey,
    scales=evaluation_scales,
    intervention_overrides=intervention_config,
)

results_df = human_lab.format_results(results, human_survey, scales=evaluation_scales)
average_error = human_lab.average_simulation_error(
    results_df,
    human_survey,
    scales=evaluation_scales,
)
```

### Survey only (no conversation)

```python
import os
import pandas as pd
from openai import OpenAI
from simulate import load_prompts
from simulate.survey import Survey
from simulate.lab import Lab

client = OpenAI(base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"])

df = pd.read_json("example/data/human.json")
survey = Survey("example/survey/human.qsf")

lab = Lab(client, df, user_model="openai/gpt-4.1-mini", seed=42)
participants = lab.sample(10)

# load_prompts handles any prompt file format → always returns {name: text}
prompts = load_prompts("example/prompts/participant/suggested.json")

results = lab.run_survey(
    sampled_users=participants,
    user_system_texts=prompts,
    survey=survey,
    scales=["feel_therm_dem", "feel_therm_rep"],  # None = all scales
    context_mode="none",
)

df_results = lab.format_results(results, survey)
```

### Survey With Real Prior Conversation

```python
survey = Survey("example/survey/human.qsf")
lab = Lab(client, human_df, user_model="openai/gpt-4.1-mini")

results = lab.run_survey(
    sampled_users=participants,
    user_system_texts=prompts,
    survey=survey,
    context_mode="real",
    intervention_overrides=intervention_config,
)
```

### Saving and loading results

```python
lab.save_results(results, "example/results/my_run.json")
loaded = lab.load_results("example/results/my_run.json")
```

When a `.qsf` run includes conversational intervention steps, the saved raw results include `survey_intervention_messages` alongside `survey_flow_trace`, embedded-data metadata, and scale outputs.

---

## Analyzing Results

`lab.format_results()` merges simulation outputs with the original participant dataframe and computes per-scale simulation error only when matched human-answer columns exist for the selected scales:

```python
df_results = lab.format_results(results, survey)
# Key columns produced per run and per scale:
#   run_id                     — deterministic unique identifier for that simulated run
#   {item_key}                 — simulated item response
#   {scale}_score              — accuracy in [0,1] (only for scales with "answers")
#   {scale}_sim_error          — distance vs. human response when matched human answers exist for scorable scale types
#   survey_user_messages       — simulator-visible message history for the whole survey run
#   survey_chatbot_*           — effective chatbot config surfaced from flow interventions
```

```python
from simulate.utils import wilcoxon_test, plot_error_hists

# Statistical test: are two prompts significantly different?
wilcoxon_test(
    df_results[df_results["user_system"] == "baseline"]["feel_therm_dem_sim_error"],
    df_results[df_results["user_system"] == "demographic"]["feel_therm_dem_sim_error"],
)

# Plot error distributions across prompts
plot_error_hists(df_results, scale="feel_therm_dem")
```

### Average error on a subset of scales

If you have a paired `.qsf` and human-response dataframe, you can compute average simulation error over any subset of scales:

```python
human_df = pd.read_json("example/data/human.json")
survey = Survey("example/survey/human.qsf")
lab = Lab(client, human_df, user_model="openai/gpt-4.1-mini", seed=42)

scales = ["feel_therm_dem", "feel_therm_rep"]
participants = lab.sample(10)

results = lab.run_survey(
    sampled_users=participants,
    user_system_texts=participant_prompt,
    survey=survey,
    scales=scales,
    intervention_overrides=intervention_config,
)

results_df = lab.format_results(results, survey, scales=scales)
avg_error = lab.average_simulation_error(results_df, survey, scales=scales)
print(avg_error)
```

---

## Prompt Engineering with Researcher

`Researcher` now provides a smaller set of prompt-development and analysis helpers on top of `Lab`:

```python
from simulate.researcher import Researcher

researcher = Researcher(client, model="openai/gpt-4.1", lab=lab, survey=survey)

# 1. Generate candidate participant prompts
candidates = researcher.suggest_initial_prompts(
    n_prompts=5,
    sampled_users=participants,
    prompt_objective="Simulate politically polarised US adults accurately",
    base_user_prompt="You are a [AGE]-year-old [PARTY] voter...",
)

# 2. Run the flow-backed survey with candidate prompts
results = researcher.run_survey(
    sampled_users=participants,
    user_system_texts=candidates,
    scales=["feel_therm_dem"],
    context_mode="none",
)
df_results = researcher.format_results(results, scales=["feel_therm_dem"])

# 3. Collect qualitative reviews of each prompt's performance
reviews = researcher.get_reviews(
    sampled_users=participants,
    scale="feel_therm_dem",
    results_df=df_results,
)

# 4. Revise prompts based on reviews
better_prompts = researcher.revise(
    n_prompts=3,
    reviews=reviews,
    prompt_objective="Simulate politically polarised US adults accurately",
    current_system_message=candidates["baseline"],
)

# 5. Compare performance across all prompts
researcher.test_prompts(results, scales=["feel_therm_dem"])
researcher.plot_prompt_performance(results, scales=["feel_therm_dem"])
```

---

## API Reference

### `simulate.save_as_survey_json(scales, output_path)`

Write a list of parsed scales to a JSON file compatible with `Survey`.

---

### `class Survey(scales, max_tokens=2500, preserve_context=True)`

| Method | Description |
|---|---|
| `administer_survey(user, scales=None)` | Run a flow-backed `.qsf` survey with one `User` agent and return `__flow__` metadata including traces and conversational intervention transcripts. |
| `get_score(scale, assessment)` | Compare an assessment dict to `scale["answers"]` and return a `[0, 1]` accuracy score for scorable scale types; `free text` currently returns `None`. |
| `describe_flow_blocks()` | Summarise executable survey blocks, question counts, runtime types, and exclusions. |
| `describe_flow_randomizers()` | Summarise parsed block randomizers, their location in the flow, whether they use even presentation, and which elements they can select. |
| `set_excluded_blocks(block_refs)` | Exclude whole flow blocks by block ID or exact block description. |
| `set_excluded_scales(scale_refs)` | Exclude scale questions by scale name without excluding the rest of the block. |
| `set_excluded_questions(question_refs)` | Exclude individual flow questions by question ID, export tag, scale name, description, or full text. |

`Survey(...)` can still be initialised from scale metadata, but runtime administration through `administer_survey(...)` and `Lab(...)` now expects flow-backed `.qsf` survey data.

`preserve_context=True` keeps prior survey content and responses in the agent's active context for later questions. Set `preserve_context=False` to reset the active message history between scale questions.

---

### `class Lab(client, df, user_model, **kwargs)`

| Method | Description |
|---|---|
| `sample(n, offset=0)` | Sample `n` participant IDs from the dataframe. |
| `quota_sample(n, quotas)` | Sample `n` participant IDs without replacement while matching exact per-column quota targets such as party or gender distributions. |
| `describe_even_presentation_plan(sampled_users, user_system_texts, survey, context_mode="none", even_presentation_counts=None)` | Preview the pre-run counts that Qualtrics-style even-presentation randomizers will assign across the requested batch, optionally with explicit user-requested counts. |
| `run_survey(sampled_users, user_system_texts, survey, scales, context_mode, even_presentation_counts=None)` | Run a flow-backed survey simulation across all participants (parallelised, best effort by default), optionally forcing explicit counts for even-presentation randomizers. |
| `format_results(results, survey, scales)` | Merge raw results into a DataFrame with simulation errors and explicit embedded-data columns. |
| `calculate_simulation_error(df, survey, scales)` | Compute per-scale distances between simulated and human responses. |
| `save_results(results, path)` | Serialise raw simulation results to JSON. |
| `load_results(path)` | Load raw simulation results from JSON. |
| `average_simulation_error(df, survey, scales)` | Compute mean simulation error over a subset of scales. |

**Constructor kwargs:** `seed`, `user_temperature`, `user_max_tokens`.

---

### `class Researcher(client, model, lab, survey, temperature=0.6, max_tokens=5000)`

| Method | Description |
|---|---|
| `run_survey(sampled_users, user_system_texts, scales=None, context_mode="none")` | Run the survey via `lab.run_survey(...)` once the researcher metaprompt formatting has been applied to participant prompts. |
| `format_results(results, scales=None)` | Delegate to `lab.format_results(...)`. |
| `suggest_initial_prompts(n, sampled_users, prompt_objective, base_user_prompt)` | Use an LLM to generate `n` candidate system prompts. |
| `get_reviews(sampled_users, scale, results_df)` | Generate qualitative reviews of prompt performance per participant. |
| `revise(n, reviews, prompt_objective, current_system_message)` | Revise prompts based on reviews. |
| `test_prompts(results, scales, baseline, chatbot)` | Print Wilcoxon test comparisons across all prompts. |
| `plot_prompt_performance(results, scales)` | Plot simulation error distributions per prompt. |
| `evaluate_dialogue(sampled_users, system, schema, user_prompt, convo_column)` | Evaluate conversation quality with a custom rubric and schema. |

---

### `class User / Chatbot(client, user_params, model, temperature, max_tokens)`

Both extend `BaseBot`. Key methods:

| Method | Description |
|---|---|
| `format_messages(system_text)` | Initialise the message history with a system prompt (applies `[TAG]` substitutions). |
| `get_response(messages, ...)` | Call the LLM and return `(response_text, thoughts)`. |
| `update_memory(memory)` | Inject a prior conversation into the agent's context. |
| `clear()` | Reset message history. |

---

## Configuration

All defaults are defined in `simulate/utils.py`:

```python
BOT_TEMPERATURE      = 1      # Temperature for User and Chatbot agents
BOT_MAX_TOKENS       = 1500   # Max tokens for conversational turns
SURVEY_MAX_TOKENS    = 2500   # Max tokens for survey responses
RESEARCH_TEMPERATURE = 0.6    # Temperature for Researcher meta-operations
RESEARCH_MAX_TOKENS  = 5000   # Max tokens for Researcher meta-operations
MAX_WORKERS          = 10     # Threads for participant-level parallelism
RESEARCH_MAX_WORKERS = 3      # Threads for Researcher-level parallelism
```

Override at construction time:
```python
lab = Lab(client, df, user_model="...", user_temperature=0.7, user_max_tokens=2000)
```

**API key** — pass your key when constructing the OpenAI client:
```python
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)
```

---

## Examples

In the `example/noteboos/` folder.

| Notebook | Description |
|---|---|
| `Example_QSF_Runtime.ipynb` | End-to-end `.qsf` runtime example: visualize flow, optionally exclude blocks, run with default personas, and evaluate against matched human data |
| `Participants.ipynb` | End-to-end walkthrough: load participant data, run a survey simulation, compare simulated vs. human responses |
| `Evaluate.ipynb` | Detailed analysis: error histograms, prompt comparisons, statistical tests |

## Trust Study Workflow

The trust study now has a batch-safe generator:

- `studies/trust/01_generate_trust_data.py` runs `studies/trust/survey_trust_new.qsf` with the packaged GSS personas and the `gss_prompt` participant prompt.
- By default, it runs the full 9,000-participant allocation using `openai/gpt-4o-mini`.
- Before any slice is run, the script builds the deterministic full-study `FL_18` condition plan, so separately executed ranges such as `0:30` and `30:60` preserve the same condition allocation they would have had in one full run.
- It saves selected participants, the selected condition plan, per-chunk raw JSON files, partial compiled raw/formatted outputs, and final compiled raw/formatted files under `studies/trust/results/`.

Run a small range:

```bash
MPLCONFIGDIR=/tmp/mpl conda run -n human-ai-eval python trust/01_generate_trust_data.py --participant-range 0:30
```

Run the next range with an explicit model:

```bash
MPLCONFIGDIR=/tmp/mpl conda run -n human-ai-eval python trust/01_generate_trust_data.py --participant-range 0:500 --user-model openai/gpt-4o-mini
```

The default chunk size is 100 participants. After each chunk, the script writes a chunk JSON file and recompiles partial outputs, so completed work is preserved if a later participant fails or the run is interrupted. Use `--retry-errors` to rerun only IDs whose existing rows contain errors, or `--fresh` to ignore existing partial output for the exact range and model.

After running one or more batches, open `studies/trust/02_compile_and_visualize_trust_results.ipynb` to combine completed formatted range files.

---

## Acknowledgements

This repository were co-created with Claude and Codex as collaborative coding partners alongside the project author.

---

## License

MIT © Yara Kyrychenko, 2025
