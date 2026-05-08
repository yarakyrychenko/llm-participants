"""
Qualtrics Survey Format (.qsf) parser.

This module exposes flow-aware parsing through ``parse_qsf()`` for runtime
execution.

Supported answerable question types:
  - Slider (HBAR / HSLIDER)  → "numeric"
  - Matrix / Likert           → "multiple choice"
  - RO / DND (rank order)     → "ranking"
  - MC (multiple choice)      → "multiple choice"
  - TE (text entry)           → "free text"
"""

from __future__ import annotations

import json
import re
import textwrap
from html.parser import HTMLParser


SUPPORTED_MC_SELECTORS = {"SAVR", "SAHR", "SACOL", "MAVR", "DL"}
IFRAME_CONTENT_PATTERN = re.compile(r"<iframe\b|iframe", re.IGNORECASE)
FLOW_NODE_COLORS = {
    "Root": "#1d3557",
    "Block": "#457b9d",
    "Standard": "#457b9d",
    "EmbeddedData": "#2a9d8f",
    "Branch": "#f4a261",
    "BlockRandomizer": "#e76f51",
    "Group": "#8d99ae",
    "EndSurvey": "#6c757d",
}


# ---------------------------------------------------------------------------
# HTML utilities
# ---------------------------------------------------------------------------


class _HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def get_text(self) -> str:
        return re.sub(r"\s+", " ", " ".join(self._parts)).strip()


def _strip_html(html_text: str) -> str:
    """Strip HTML tags and collapse whitespace."""
    if not html_text:
        return ""
    s = _HTMLStripper()
    s.feed(html_text)
    return s.get_text()


def _get_question_text(payload: dict) -> str:
    """Return the best available human-readable text for a Qualtrics element."""
    question_text = _strip_html(payload.get("QuestionText", ""))
    if question_text:
        return question_text
    return _strip_html(payload.get("QuestionDescription", ""))


def _contains_iframe_content(payload: dict) -> bool:
    text = " ".join(
        str(payload.get(key, ""))
        for key in ("QuestionText", "QuestionDescription")
    )
    return bool(IFRAME_CONTENT_PATTERN.search(text))


def _normalize_ordered_values(raw) -> list[str]:
    """Normalise Qualtrics list-or-dict ordered payloads into a list."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(item) for item in raw]
    if isinstance(raw, dict):
        try:
            ordered_keys = sorted(raw, key=lambda key: int(str(key)))
        except ValueError:
            ordered_keys = list(raw.keys())
        return [str(raw[key]) for key in ordered_keys]
    return [str(raw)]


def _materialize_advanced_randomization(
    question_ids: list[str],
    randomization: dict,
) -> list[str]:
    """Build the block's question order from Qualtrics advanced randomization."""
    advanced = randomization.get("Advanced", {})
    fixed_order = _normalize_ordered_values(advanced.get("FixedOrder"))
    randomize_all = _normalize_ordered_values(advanced.get("RandomizeAll"))
    undisplayed = set(_normalize_ordered_values(advanced.get("Undisplayed")))

    randomised_queue = [qid for qid in randomize_all if qid in question_ids and qid not in undisplayed]
    ordered: list[str] = []

    for token in fixed_order:
        if token == "{~Randomized~}":
            if randomised_queue:
                ordered.append(randomised_queue.pop(0))
        elif token in question_ids and token not in undisplayed:
            ordered.append(token)

    for qid in randomised_queue:
        if qid not in ordered:
            ordered.append(qid)

    for qid in question_ids:
        if qid not in ordered and qid not in undisplayed:
            ordered.append(qid)

    return ordered


# ---------------------------------------------------------------------------
# Per-type parsers
# ---------------------------------------------------------------------------


def _parse_slider(payload: dict) -> dict:
    """Slider (HBAR / HSLIDER) → numeric scale."""
    tag = payload["DataExportTag"]
    choices = payload.get("Choices", {})
    config = payload.get("Configuration", {})

    min_val = config.get("CSSliderMin", 0)
    max_val = config.get("CSSliderMax", 100)

    if len(choices) == 1:
        _, choice_val = next(iter(choices.items()))
        display = _strip_html(choice_val.get("Display", ""))
        items = {tag: display if display else tag}
    else:
        items = {}
        for key, value in choices.items():
            display = _strip_html(value.get("Display", ""))
            item_key = f"{tag}_{key}"
            items[item_key] = display if display else item_key

    options = {
        f"Min ({min_val})": min_val,
        f"Max ({max_val})": max_val,
    }

    return {
        "name": tag,
        "type": "numeric",
        "question": _get_question_text(payload),
        "items": items,
        "options": options,
    }


def _parse_matrix(payload: dict) -> dict:
    """Matrix / Likert / SingleAnswer → multiple choice scale."""
    tag = payload["DataExportTag"]
    choices = payload.get("Choices", {})
    answers = payload.get("Answers", {})

    items = {}
    for key, value in choices.items():
        display = _strip_html(value.get("Display", ""))
        item_key = f"{tag}_{key}"
        items[item_key] = display if display else item_key

    options = {}
    for key, value in answers.items():
        display = _strip_html(value.get("Display", ""))
        label = display if display else str(key)
        options[label] = int(key)

    return {
        "name": tag,
        "type": "multiple choice",
        "question": _get_question_text(payload),
        "items": items,
        "options": options,
    }


def _parse_mc(payload: dict) -> dict:
    """Multiple-choice question → multiple choice scale (single item)."""
    tag = payload["DataExportTag"]
    choices = payload.get("Choices", {})

    options = {}
    for key, value in choices.items():
        display = _strip_html(value.get("Display", ""))
        label = display if display else str(key)
        options[label] = int(key)

    question_text = _get_question_text(payload)

    return {
        "name": tag,
        "type": "multiple choice",
        "question": question_text,
        "items": {tag: question_text if question_text else tag},
        "options": options,
    }


def _parse_ranking(payload: dict) -> dict:
    """Rank-order (RO / DND) → ranking scale."""
    tag = payload["DataExportTag"]
    choices = payload.get("Choices", {})

    items = {}
    for key, value in choices.items():
        display = _strip_html(value.get("Display", ""))
        item_key = f"{tag}_{key}"
        items[item_key] = display if display else item_key

    return {
        "name": tag,
        "type": "ranking",
        "question": _get_question_text(payload),
        "items": items,
        "options": {},
    }


def _has_embedded_default_text(payload: dict) -> bool:
    default_choices = payload.get("DefaultChoices", {})
    if not isinstance(default_choices, dict):
        return False

    text_payload = default_choices.get("TEXT", {})
    if isinstance(text_payload, dict):
        text_value = str(text_payload.get("Text", "")).strip()
    else:
        text_value = str(text_payload).strip()
    return bool(text_value.startswith("${") and text_value.endswith("}"))


def _is_participant_facing_text_entry(payload: dict) -> bool:
    if payload.get("QuestionType") != "TE":
        return False
    if _has_embedded_default_text(payload):
        return False
    return bool(_get_question_text(payload))


def _parse_free_text(payload: dict) -> dict:
    """Text-entry question → free-text scale (single item)."""
    tag = payload["DataExportTag"]
    question_text = _get_question_text(payload)
    return {
        "name": tag,
        "type": "free text",
        "question": question_text,
        "items": {tag: question_text if question_text else tag},
        "options": {},
    }


def _parse_scale(payload: dict) -> dict | None:
    qtype = payload.get("QuestionType")
    selector = payload.get("Selector")
    sub_selector = payload.get("SubSelector")

    if qtype == "Slider":
        return _parse_slider(payload)
    if qtype == "Matrix" and selector == "Likert" and sub_selector == "SingleAnswer":
        return _parse_matrix(payload)
    if qtype == "RO" and selector == "DND":
        return _parse_ranking(payload)
    if qtype == "MC" and selector in SUPPORTED_MC_SELECTORS:
        return _parse_mc(payload)
    if qtype == "TE" and _is_participant_facing_text_entry(payload):
        return _parse_free_text(payload)
    return None


def _classify_question(payload: dict) -> str:
    """Return a runtime-oriented question classification."""
    if _parse_scale(payload) is not None:
        return "scale"

    qtype = payload.get("QuestionType")
    if qtype == "DB":
        return "intervention" if _contains_iframe_content(payload) else "display"
    if qtype == "TE":
        return "text_entry"
    if qtype in {"Timing", "Captcha"}:
        return "passive"
    return "unsupported"


# ---------------------------------------------------------------------------
# Rich parser
# ---------------------------------------------------------------------------


def parse_qsf(
    qsf_path: str,
    question_ids: list[str] | None = None,
    skip_trash: bool = True,
) -> dict:
    """Parse a Qualtrics `.qsf` into flow-aware metadata."""
    with open(qsf_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    elements: list[dict] = data.get("SurveyElements", [])
    if not elements:
        raise ValueError("QSF file does not contain SurveyElements.")

    survey_entry = data.get("SurveyEntry", {})
    warnings: list[str] = []

    block_payload = {}
    flow_payload = None
    for elem in elements:
        if elem.get("Element") == "BL":
            block_payload = elem.get("Payload", {})
        elif elem.get("Element") == "FL":
            flow_payload = elem.get("Payload", {})

    trash_question_ids: set[str] = set()
    blocks: dict[str, dict] = {}
    for _, block in block_payload.items():
        if not isinstance(block, dict):
            continue

        question_list = [
            entry["QuestionID"]
            for entry in block.get("BlockElements", [])
            if entry.get("Type") == "Question"
        ]
        block_type = block.get("Type")
        if skip_trash and block_type == "Trash":
            trash_question_ids.update(question_list)

        block_id = block.get("ID")
        if not block_id:
            continue

        options = block.get("Options", {})
        blocks[block_id] = {
            "id": block_id,
            "type": block_type,
            "description": block.get("Description", ""),
            "question_ids": question_list,
            "block_elements": block.get("BlockElements", []),
            "options": options,
            "randomization": options.get("Randomization", {}),
        }

    questions: dict[str, dict] = {}
    scales: list[dict] = []
    qid_to_scale: dict[str, str] = {}

    for elem in elements:
        if elem.get("Element") != "SQ":
            continue

        payload = elem.get("Payload", {})
        qid = payload.get("QuestionID", "")
        if skip_trash and qid in trash_question_ids:
            continue
        if question_ids is not None and qid not in question_ids:
            continue

        scale = _parse_scale(payload)
        runtime_type = _classify_question(payload)

        question_meta = {
            "question_id": qid,
            "data_export_tag": payload.get("DataExportTag", qid),
            "question_type": payload.get("QuestionType"),
            "selector": payload.get("Selector"),
            "sub_selector": payload.get("SubSelector"),
            "text": _get_question_text(payload),
            "description": _strip_html(payload.get("QuestionDescription", "")),
            "runtime_type": runtime_type,
            "raw_payload": payload,
            "scale_name": None,
        }

        if scale is not None:
            question_meta["scale_name"] = scale["name"]
            scales.append(scale)
            qid_to_scale[qid] = scale["name"]

        questions[qid] = question_meta

    for block in blocks.values():
        filtered_question_ids = [
            qid for qid in block["question_ids"] if qid in questions
        ]
        block["question_ids"] = filtered_question_ids

        if block["options"].get("RandomizeQuestions") == "Advanced":
            block["materialized_question_ids"] = _materialize_advanced_randomization(
                filtered_question_ids,
                block["randomization"],
            )
        else:
            block["materialized_question_ids"] = filtered_question_ids[:]

    if flow_payload is None:
        warnings.append("QSF file does not contain a survey flow element.")
        flow_payload = {"Type": "Root", "Flow": []}

    return {
        "survey_id": survey_entry.get("SurveyID"),
        "survey_name": survey_entry.get("SurveyName"),
        "survey_entry": survey_entry,
        "scales": scales,
        "questions": questions,
        "blocks": blocks,
        "flow": flow_payload,
        "trash_question_ids": sorted(trash_question_ids),
        "qid_to_scale": qid_to_scale,
        "warnings": warnings,
    }


def _summarize_runtime_types(question_ids: list[str], questions: dict[str, dict]) -> str:
    counts: dict[str, int] = {}
    for qid in question_ids:
        runtime_type = questions.get(qid, {}).get("runtime_type", "unknown")
        counts[runtime_type] = counts.get(runtime_type, 0) + 1
    if not counts:
        return "no questions"
    order = ["scale", "display", "intervention", "text_entry", "passive", "unsupported", "unknown"]
    parts = [f"{counts[key]} {key}" for key in order if key in counts]
    return ", ".join(parts)


def _extract_branch_description(branch_logic: dict) -> str:
    if not isinstance(branch_logic, dict):
        return ""

    descriptions: list[str] = []

    def walk(node):
        if isinstance(node, dict):
            raw = node.get("Description")
            if raw:
                text = _strip_html(raw)
                if text and text not in descriptions:
                    descriptions.append(text)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(branch_logic)
    return descriptions[0] if descriptions else ""


def _flow_node_label(node: dict, parsed: dict) -> tuple[str, str]:
    node_type = node.get("Type", "Unknown")
    blocks = parsed.get("blocks", {})
    questions = parsed.get("questions", {})

    if node_type in {"Block", "Standard"}:
        block = blocks.get(node.get("ID"), {})
        description = block.get("description") or node.get("Description") or node.get("ID", "Unnamed block")
        summary_parts = []
        if block.get("question_ids"):
            summary_parts.append(_summarize_runtime_types(block["question_ids"], questions))
        if block.get("options", {}).get("RandomizeQuestions") == "Advanced":
            summary_parts.append("question randomization")
        return description, " | ".join(summary_parts)

    if node_type == "EmbeddedData":
        fields = [entry.get("Field") for entry in node.get("EmbeddedData", []) if entry.get("Field")]
        return "Embedded Data", ", ".join(fields[:5]) + (" ..." if len(fields) > 5 else "")

    if node_type == "BlockRandomizer":
        child_labels = []
        for child in node.get("Flow", []):
            block = blocks.get(child.get("ID"), {})
            child_labels.append(block.get("description") or child.get("ID") or child.get("Type", "Flow node"))
        subset = node.get("SubSet")
        prefix = f"present {subset} of {len(child_labels)}" if subset else f"{len(child_labels)} options"
        if node.get("EvenPresentation"):
            prefix += ", even presentation"
        detail = ", ".join(child_labels[:4])
        if len(child_labels) > 4:
            detail += ", ..."
        return "Block Randomizer", f"{prefix} | {detail}".strip(" |")

    if node_type == "Branch":
        description = node.get("Description") or "Conditional branch"
        logic = _extract_branch_description(node.get("BranchLogic", {}))
        return description, logic

    if node_type == "Group":
        return node.get("Description") or "Group", f"{len(node.get('Flow', []))} child nodes"

    if node_type == "EndSurvey":
        ending_type = node.get("EndingType")
        return "End Survey", ending_type or "terminate survey"

    if node_type == "Root":
        return parsed.get("survey_name") or "Survey Flow", parsed.get("survey_id") or ""

    return node_type, node.get("Description", "")


def _collect_flow_rows(node: dict, parsed: dict, rows: list[dict], depth: int = 0, parent_index: int | None = None) -> None:
    title, detail = _flow_node_label(node, parsed)
    rows.append(
        {
            "index": len(rows),
            "parent_index": parent_index,
            "depth": depth,
            "node_type": node.get("Type", "Unknown"),
            "title": title,
            "detail": detail,
        }
    )
    current_index = rows[-1]["index"]
    for child in node.get("Flow", []):
        _collect_flow_rows(child, parsed, rows, depth + 1, current_index)


def _coerce_parsed_qsf(survey_or_qsf) -> dict:
    if isinstance(survey_or_qsf, str):
        return parse_qsf(survey_or_qsf)
    if isinstance(survey_or_qsf, dict) and "flow" in survey_or_qsf and "blocks" in survey_or_qsf:
        return survey_or_qsf
    flow_data = getattr(survey_or_qsf, "flow_data", None)
    if flow_data is not None:
        return flow_data
    raise TypeError("Expected a .qsf path, parsed QSF dict, or Survey instance with flow_data.")


def visualize_survey_flow(
    survey_or_qsf,
    figsize: tuple[float, float] | None = None,
    wrap_width: int = 46,
):
    """Render the parsed survey flow as a simple tree diagram."""
    import matplotlib.pyplot as plt

    parsed = _coerce_parsed_qsf(survey_or_qsf)
    rows: list[dict] = []
    _collect_flow_rows(parsed.get("flow", {"Type": "Root", "Flow": []}), parsed, rows)

    if not rows:
        raise ValueError("Parsed survey does not contain any flow nodes to visualize.")

    max_depth = max(row["depth"] for row in rows)
    height = max(6.0, 1.2 * len(rows))
    width = max(10.0, 4.5 + max_depth * 2.8)
    if figsize is None:
        figsize = (width, height)

    fig, ax = plt.subplots(figsize=figsize)
    x_step = 2.8
    y_step = 1.3
    box_width = 2.35
    y_top = (len(rows) - 1) * y_step
    positions = {}

    for row in rows:
        x = row["depth"] * x_step
        y = y_top - row["index"] * y_step
        positions[row["index"]] = (x, y)

        color = FLOW_NODE_COLORS.get(row["node_type"], "#adb5bd")
        title = textwrap.fill(row["title"], width=wrap_width)
        detail = textwrap.fill(row["detail"], width=wrap_width) if row["detail"] else ""
        label = title if not detail else f"{title}\n{detail}"

        ax.text(
            x,
            y,
            label,
            ha="left",
            va="center",
            fontsize=9,
            color="#111111",
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": color,
                "edgecolor": "#1f1f1f",
                "alpha": 0.18,
            },
        )

    for row in rows:
        parent_index = row["parent_index"]
        if parent_index is None:
            continue
        parent_x, parent_y = positions[parent_index]
        child_x, child_y = positions[row["index"]]
        ax.annotate(
            "",
            xy=(child_x - 0.12, child_y),
            xytext=(parent_x + box_width, parent_y),
            arrowprops={"arrowstyle": "->", "color": "#6c757d", "lw": 1.2},
        )

    ax.set_title(parsed.get("survey_name") or "Survey Flow", fontsize=14, loc="left")
    ax.set_xlim(-0.4, max_depth * x_step + box_width + 1.8)
    ax.set_ylim(-y_step, y_top + y_step)
    ax.axis("off")
    fig.tight_layout()
    return fig, ax


def save_as_survey_json(scales: list[dict], output_path: str) -> None:
    """Save a list of scales to a survey-compatible JSON file."""
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(scales, handle, ensure_ascii=False, indent=2)
