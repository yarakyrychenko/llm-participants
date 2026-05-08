from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI

SCRIPT_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from simulate import load_gss_personas, load_prompts
from simulate.lab import Lab
from simulate.survey import Survey


SEED = 42
USER_MODEL = "openai/gpt-4o-mini"
USER_TEMPERATURE = 1.2
USER_MAX_TOKENS = 512
PARTICIPANT_PROMPT_NAME = "gss_prompt"

FULL_TREATMENT_COUNT = 500
FULL_CONTROL_TOTAL = 1_000
FULL_PARTICIPANT_COUNT = 9_000
TOP_LEVEL_RANDOMIZER_ID = "FL_18"
DEFAULT_CHUNK_SIZE = 100

BLOCKS_TO_SKIP = [
    "Consent Form",
    "Filter",
    "Attention Check",
]


def get_root() -> Path:
    cwd = Path.cwd().resolve()
    if (cwd / "simulate").exists():
        return cwd
    return SCRIPT_ROOT


ROOT = get_root()
SURVEY_PATH = ROOT / "trust" / "survey_trust_new.qsf"
PROMPTS_PATH = ROOT / "simulate" / "personas" / "prompts.json"
RESULTS_ROOT = ROOT / "trust" / "results"


def make_client() -> OpenAI:
    if os.environ.get("OPENROUTER_API_KEY"):
        return OpenAI(
            base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    raise RuntimeError(
        "Set either OPENAI_API_KEY or OPENROUTER_API_KEY before running trust generation."
    )


def configure_survey() -> Survey:
    survey = Survey(str(SURVEY_PATH))
    survey.set_excluded_blocks(BLOCKS_TO_SKIP)
    return survey


def load_participant_prompt(prompt_name: str = PARTICIPANT_PROMPT_NAME) -> dict[str, str]:
    prompt_library = load_prompts(str(PROMPTS_PATH))
    if prompt_name not in prompt_library:
        available = ", ".join(sorted(prompt_library))
        raise KeyError(
            f"Unknown participant prompt {prompt_name!r}. Available prompts: {available}"
        )
    return {prompt_name: prompt_library[prompt_name]}


def load_adult_gss_personas() -> pd.DataFrame:
    personas = load_gss_personas().copy()
    personas["age"] = pd.to_numeric(personas["age"], errors="coerce")
    personas = personas[personas["age"] >= 18].copy()
    personas["id"] = personas["id"].astype(str)
    return personas


def sample_gss_personas(df: pd.DataFrame, n: int, *, seed: int, sample_name: str) -> pd.DataFrame:
    if n <= 0:
        raise ValueError("n must be positive.")
    if df.empty:
        raise ValueError("Cannot sample from an empty GSS persona pool.")

    replace = n > len(df)
    sampled = df.sample(n=n, replace=replace, random_state=seed).reset_index(drop=True).copy()
    sampled["source_persona_id"] = sampled["id"].astype(str)
    sampled["sample_name"] = sample_name
    sampled["sample_index"] = range(1, n + 1)
    sampled["id"] = [f"{sample_name}_{index:05d}" for index in sampled["sample_index"]]
    return sampled


def find_randomizer_node(survey: Survey, flow_id: str = TOP_LEVEL_RANDOMIZER_ID) -> dict:
    stack = list(survey.flow_data.get("flow", {}).get("Flow", []))
    while stack:
        node = stack.pop(0)
        if node.get("FlowID") == flow_id:
            return node
        stack[:0] = list(node.get("Flow", []))
    raise KeyError(f"Could not find randomizer {flow_id}.")


def extract_condition_options(
    survey: Survey,
    flow_id: str = TOP_LEVEL_RANDOMIZER_ID,
) -> list[dict[str, str]]:
    node = find_randomizer_node(survey, flow_id)
    active_children = (
        survey._get_active_randomizer_children(node)
        if hasattr(survey, "_get_active_randomizer_children")
        else node.get("Flow", [])
    )
    options = []
    for child in active_children:
        embedded = child.get("EmbeddedData", [])
        condition_label = (
            str(embedded[0].get("Value"))
            if embedded
            else str(child.get("Description") or child.get("FlowID"))
        )
        option_key = child.get("Description") or child.get("ID") or child.get("FlowID")
        options.append(
            {
                "flow_option": str(option_key),
                "flow_id": str(child.get("FlowID") or option_key),
                "condition_label": condition_label,
            }
        )
    return options


def split_condition_options(
    condition_options: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    controls = [
        option
        for option in condition_options
        if option["condition_label"].strip().lower().startswith("control")
    ]
    treatments = [option for option in condition_options if option not in controls]
    return treatments, controls


def build_full_condition_counts(condition_options: list[dict[str, str]]) -> dict[str, int]:
    treatment_options, control_options = split_condition_options(condition_options)
    counts = {option["flow_option"]: FULL_TREATMENT_COUNT for option in treatment_options}
    if control_options:
        base, remainder = divmod(FULL_CONTROL_TOTAL, len(control_options))
        for index, option in enumerate(control_options):
            counts[option["flow_option"]] = base + (1 if index < remainder else 0)

    ordered_counts = {
        option["flow_option"]: counts[option["flow_option"]]
        for option in condition_options
    }
    total = sum(ordered_counts.values())
    if total != FULL_PARTICIPANT_COUNT:
        raise ValueError(
            f"Full condition counts sum to {total:,}, expected {FULL_PARTICIPANT_COUNT:,}."
        )
    return ordered_counts


def build_text_only_intervention_overrides(survey: Survey) -> dict[str, dict]:
    conditions = {}
    for meta in survey.flow_data.get("questions", {}).values():
        if meta.get("runtime_type") != "intervention":
            continue
        raw_unsafe = str(meta.get("raw_payload", {}).get("QuestionText_Unsafe", ""))
        if "<iframe" in raw_unsafe.lower():
            text = "[Embedded interactive treatment content omitted in text-only simulation.]"
        else:
            text = meta.get("text") or ""
        conditions[meta["data_export_tag"]] = {
            "text": text,
            "completion_code": "SIMULATED_INTERVENTION",
        }
    return {"defaults": {}, "conditions": conditions}


def sanitize_model_name(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", model_name).strip("-")


def build_results_dir(user_model: str) -> Path:
    model_label = sanitize_model_name(user_model)
    if not model_label:
        raise ValueError("user_model must contain at least one path-safe character.")
    return RESULTS_ROOT / model_label


def build_run_label(participant_start: int, participant_end: int, user_model: str) -> str:
    model_label = sanitize_model_name(user_model)
    return f"n{participant_start}_m{participant_end}_model{model_label}_seed{SEED}"


def build_output_paths(run_label: str, results_dir: Path) -> dict[str, Path]:
    temp_dir = results_dir / f"trust_new_generation_{run_label}"
    return {
        "temp_dir": temp_dir,
        "chunks_dir": temp_dir / "chunks",
        "participants": temp_dir / f"trust_new_participants_{run_label}.csv",
        "condition_plan": temp_dir / f"trust_new_condition_plan_{run_label}.json",
        "partial_raw": temp_dir / f"trust_new_raw_results_{run_label}.partial.json",
        "partial_formatted": temp_dir / f"trust_new_formatted_results_{run_label}.partial.csv",
        "raw": results_dir / f"trust_new_raw_results_{run_label}.json",
        "formatted": results_dir / f"trust_new_formatted_results_{run_label}.csv",
    }


def save_json(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved {path}", flush=True)


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def serialize_even_plan(even_plan: dict) -> list[dict]:
    rows = []
    for (participant_id, user_system), plan in sorted(even_plan.items()):
        rows.append(
            {
                "id": participant_id,
                "user_system": user_system,
                "even_presentation_plan": plan,
            }
        )
    return rows


def make_progress_callback(prefix: str, enabled: bool = True):
    if not enabled:
        return None

    def callback(completed: int, total: int) -> None:
        total = max(total, 1)
        width = 24
        filled = int(width * completed / total)
        bar = "#" * filled + "-" * (width - filled)
        print(f"\r{prefix}: [{bar}] {completed}/{total}", end="", flush=True)
        if completed >= total:
            print(flush=True)

    return callback


def chunked(values: list[str], chunk_size: int):
    for start in range(0, len(values), chunk_size):
        yield start, values[start : start + chunk_size]


def load_existing_results(paths: dict[str, Path]) -> list[dict]:
    if paths["partial_raw"].exists():
        existing = load_json(paths["partial_raw"])
        print(f"Loaded {len(existing)} existing partial rows from {paths['partial_raw']}", flush=True)
        return existing

    chunk_files = sorted(paths["chunks_dir"].glob("*.json"))
    if not chunk_files:
        return []

    rows = []
    for chunk_file in chunk_files:
        rows.extend(load_json(chunk_file))
    print(f"Loaded {len(rows)} existing rows from {len(chunk_files)} chunk files.", flush=True)
    return rows


def completed_participant_ids(
    raw_results: list[dict],
    *,
    retry_errors: bool,
) -> set[str]:
    completed = set()
    for row in raw_results:
        if retry_errors and "error" in row:
            continue
        completed.add(str(row.get("id")))
    return completed


def sort_results(raw_results: list[dict], participant_order: list[str]) -> list[dict]:
    order = {participant_id: index for index, participant_id in enumerate(participant_order)}
    return sorted(
        raw_results,
        key=lambda row: (
            order.get(str(row.get("id")), len(order)),
            str(row.get("user_system", "")),
            str(row.get("run_id", "")),
        ),
    )


def format_and_save(
    lab: Lab,
    raw_results: list[dict],
    survey: Survey,
    output_path: Path,
) -> pd.DataFrame:
    successful_results = [row for row in raw_results if "error" not in row]
    if not successful_results:
        pd.DataFrame(raw_results).to_csv(output_path, index=False)
        print(f"No successful rows yet; saved raw-shaped partial CSV to {output_path}", flush=True)
        return pd.DataFrame(raw_results)

    formatted = lab.format_results(successful_results, survey)
    formatted.to_csv(output_path, index=False)
    print(f"Saved {output_path}", flush=True)
    return formatted


def run_generation(
    participant_start: int = 0,
    participant_end: int = FULL_PARTICIPANT_COUNT,
    user_model: str = USER_MODEL,
    prompt_name: str = PARTICIPANT_PROMPT_NAME,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    show_progress: bool = True,
    fresh: bool = False,
    retry_errors: bool = False,
) -> tuple[list[dict], pd.DataFrame]:
    if participant_end > FULL_PARTICIPANT_COUNT:
        raise ValueError(
            f"participant_end must be <= {FULL_PARTICIPANT_COUNT:,} for the full trust study."
        )

    results_dir = build_results_dir(user_model)
    results_dir.mkdir(parents=True, exist_ok=True)
    survey = configure_survey()
    participant_prompt = load_participant_prompt(prompt_name)
    condition_options = extract_condition_options(survey)
    full_condition_counts = build_full_condition_counts(condition_options)
    intervention_overrides = build_text_only_intervention_overrides(survey)

    gss_personas_df = load_adult_gss_personas()
    full_personas_df = sample_gss_personas(
        gss_personas_df,
        FULL_PARTICIPANT_COUNT,
        seed=SEED + 1,
        sample_name="full",
    )
    full_participant_ids = full_personas_df["id"].astype(str).tolist()
    selected_participant_ids = full_participant_ids[participant_start:participant_end]
    if not selected_participant_ids:
        raise ValueError("No participant IDs were selected. Check the requested participant range.")

    run_label = build_run_label(participant_start, participant_end, user_model)
    paths = build_output_paths(run_label, results_dir)
    paths["chunks_dir"].mkdir(parents=True, exist_ok=True)

    selected_personas = full_personas_df[
        full_personas_df["id"].isin(selected_participant_ids)
    ].copy()
    selected_personas.to_csv(paths["participants"], index=False)
    print(f"Saved selected participant frame to {paths['participants']}", flush=True)

    planning_lab = Lab(
        client=None,
        df=full_personas_df.copy(),
        user_model=user_model,
        user_temperature=USER_TEMPERATURE,
        user_max_tokens=USER_MAX_TOKENS,
        seed=SEED,
    )
    full_even_plan = planning_lab._build_even_presentation_plans(
        full_participant_ids,
        participant_prompt,
        survey,
        "none",
        even_presentation_counts={TOP_LEVEL_RANDOMIZER_ID: full_condition_counts},
    )
    selected_even_plan = {
        key: value
        for key, value in full_even_plan.items()
        if key[0] in set(selected_participant_ids)
    }
    plan_description = planning_lab.describe_even_presentation_plan(
        full_participant_ids,
        participant_prompt,
        survey,
        even_presentation_counts={TOP_LEVEL_RANDOMIZER_ID: full_condition_counts},
    )
    save_json(
        {
            "seed": SEED,
            "user_model": user_model,
            "participant_prompt_name": prompt_name,
            "participant_start": participant_start,
            "participant_end": participant_end,
            "full_participant_count": FULL_PARTICIPANT_COUNT,
            "selected_participant_count": len(selected_participant_ids),
            "top_level_randomizer_id": TOP_LEVEL_RANDOMIZER_ID,
            "condition_options": condition_options,
            "full_condition_counts": full_condition_counts,
            "plan_description": plan_description,
            "selected_even_presentation_plan": serialize_even_plan(selected_even_plan),
        },
        paths["condition_plan"],
    )

    print("Trust generation setup:", flush=True)
    print(
        {
            "participant_start": participant_start,
            "participant_end": participant_end,
            "selected_n": len(selected_participant_ids),
            "full_n": FULL_PARTICIPANT_COUNT,
            "user_model": user_model,
            "results_dir": str(results_dir),
            "prompt_name": prompt_name,
            "chunk_size": chunk_size,
            "intervention_overrides": len(intervention_overrides["conditions"]),
        },
        flush=True,
    )
    print("Full condition counts:", full_condition_counts, flush=True)

    client = make_client()
    lab = Lab(
        client=client,
        df=full_personas_df.copy(),
        user_model=user_model,
        user_temperature=USER_TEMPERATURE,
        user_max_tokens=USER_MAX_TOKENS,
        seed=SEED,
    )
    scales = survey.get_scales(None)

    raw_results = [] if fresh else load_existing_results(paths)
    done_ids = completed_participant_ids(raw_results, retry_errors=retry_errors)
    ids_to_run = [participant_id for participant_id in selected_participant_ids if participant_id not in done_ids]
    if done_ids:
        print(
            f"Resuming with {len(done_ids)} participant IDs already present; "
            f"{len(ids_to_run)} remain.",
            flush=True,
        )

    selected_index = {
        participant_id: index for index, participant_id in enumerate(selected_participant_ids)
    }
    for chunk_index, (_chunk_offset, participant_chunk) in enumerate(
        chunked(ids_to_run, chunk_size),
        start=1,
    ):
        absolute_start = participant_start + selected_index[participant_chunk[0]]
        absolute_end = participant_start + selected_index[participant_chunk[-1]] + 1
        chunk_label = f"chunk{chunk_index:04d}_n{absolute_start}_m{absolute_end}"
        chunk_path = paths["chunks_dir"] / f"trust_new_raw_results_{run_label}_{chunk_label}.json"
        print(
            f"Running {chunk_label}: {len(participant_chunk)} participants "
            f"({participant_chunk[0]} to {participant_chunk[-1]})",
            flush=True,
        )

        chunk_even_plan = {
            key: value
            for key, value in selected_even_plan.items()
            if key[0] in set(participant_chunk)
        }
        chunk_results = lab.simulate_survey_run(
            participant_chunk,
            non_iterables={
                "user_system_texts": participant_prompt,
                "survey": survey,
                "scales": scales,
                "context_mode": "none",
                "intervention_overrides": intervention_overrides,
                "even_presentation_plans": chunk_even_plan,
                "_progress_callback": make_progress_callback(chunk_label, show_progress),
            },
        )

        save_json(chunk_results, chunk_path)
        participant_chunk_set = set(participant_chunk)
        raw_results = [
            row for row in raw_results if str(row.get("id")) not in participant_chunk_set
        ]
        raw_results.extend(chunk_results)
        raw_results = sort_results(raw_results, selected_participant_ids)
        save_json(raw_results, paths["partial_raw"])
        format_and_save(lab, raw_results, survey, paths["partial_formatted"])
        print(
            f"Completed {chunk_label}; cumulative rows: {len(raw_results)}",
            flush=True,
        )

    raw_results = sort_results(raw_results, selected_participant_ids)
    save_json(raw_results, paths["raw"])
    formatted_results = format_and_save(lab, raw_results, survey, paths["formatted"])

    error_count = sum(1 for row in raw_results if "error" in row)
    print(f"Saved final raw results to {paths['raw']}", flush=True)
    print(f"Saved final formatted results to {paths['formatted']}", flush=True)
    print(
        f"Final rows: {len(raw_results)} total, {len(raw_results) - error_count} successful, {error_count} errors.",
        flush=True,
    )
    return raw_results, formatted_results


def parse_range(value: str) -> tuple[int, int]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("Ranges must use START:END syntax, for example 0:30.")
    start_text, end_text = value.split(":", 1)
    if not start_text or not end_text:
        raise argparse.ArgumentTypeError("Both START and END are required, for example 0:30.")
    try:
        start = int(start_text)
        end = int(end_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Range bounds must be integers.") from exc
    if start < 0 or end <= start:
        raise argparse.ArgumentTypeError("Range must satisfy 0 <= START < END.")
    return start, end


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate trust-study simulation data from survey_trust_new.qsf. "
            "By default this runs the full 9,000-participant study."
        )
    )
    parser.add_argument(
        "--participant-range",
        type=parse_range,
        default=None,
        help="Half-open participant slice in START:END form, for example 0:30 or 30:60.",
    )
    parser.add_argument(
        "--participant-start",
        type=int,
        default=None,
        help="Optional start index for a half-open participant slice.",
    )
    parser.add_argument(
        "--participant-end",
        type=int,
        default=None,
        help="Optional end index for a half-open participant slice.",
    )
    parser.add_argument(
        "--user-model",
        default=USER_MODEL,
        help=f"Participant simulator model name. Default: {USER_MODEL}.",
    )
    parser.add_argument(
        "--participant-prompt-name",
        default=PARTICIPANT_PROMPT_NAME,
        help=f"Prompt name from {PROMPTS_PATH}. Default: {PARTICIPANT_PROMPT_NAME}.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=(
            "Number of participants to run before saving a chunk and recompiling partial outputs. "
            f"Default: {DEFAULT_CHUNK_SIZE}."
        ),
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore existing partial/chunk output for this exact range and model.",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="When resuming, rerun participant IDs whose existing rows contain errors.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable per-chunk terminal progress bars.",
    )
    args = parser.parse_args()

    if args.participant_range is not None:
        if args.participant_start is not None or args.participant_end is not None:
            parser.error("Use either --participant-range or --participant-start/--participant-end, not both.")
        args.participant_start, args.participant_end = args.participant_range

    if args.participant_start is None:
        args.participant_start = 0
    if args.participant_end is None:
        args.participant_end = FULL_PARTICIPANT_COUNT

    if args.participant_start < 0:
        parser.error("--participant-start must be >= 0.")
    if args.participant_end <= args.participant_start:
        parser.error("--participant-end must be greater than --participant-start.")
    if args.participant_end > FULL_PARTICIPANT_COUNT:
        parser.error(f"--participant-end must be <= {FULL_PARTICIPANT_COUNT}.")
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be > 0.")

    return args


def main() -> None:
    started_at = time.perf_counter()
    args = parse_args()
    try:
        run_generation(
            participant_start=args.participant_start,
            participant_end=args.participant_end,
            user_model=args.user_model,
            prompt_name=args.participant_prompt_name,
            chunk_size=args.chunk_size,
            show_progress=not args.no_progress,
            fresh=args.fresh,
            retry_errors=args.retry_errors,
        )
    finally:
        elapsed_seconds = time.perf_counter() - started_at
        elapsed_minutes, remaining_seconds = divmod(elapsed_seconds, 60)
        print(
            "Total runtime:",
            f"{int(elapsed_minutes)}m {remaining_seconds:.1f}s ({elapsed_seconds:.1f}s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
