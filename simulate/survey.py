import json
import os
import re
import ast
import copy
from random import Random

from openai import OpenAI

from .agent import Chatbot
from .qsf import parse_qsf
from .utils import SURVEY_MAX_TOKENS, abs_distance, footrule_distance, get_int


class MissingInterventionConfigError(RuntimeError):
    """Raised when a QSF flow requires an intervention override to proceed."""


class UnsupportedSurveyFlowError(RuntimeError):
    """Raised when a QSF flow uses semantics the runtime cannot execute safely."""


class Survey:
    def __init__(self, scales, max_tokens=SURVEY_MAX_TOKENS, preserve_context=True):
        self.max_tokens = max_tokens
        self.preserve_context = preserve_context
        self.flow_data = None
        self.warnings: list[str] = []

        if isinstance(scales, str):
            if scales.endswith(".qsf"):
                self.flow_data = parse_qsf(scales)
                self.warnings.extend(self.flow_data.get("warnings", []))
                scales = self.flow_data["scales"]
            else:
                with open(scales, "r", encoding="utf-8") as handle:
                    scales = json.load(handle)
        elif isinstance(scales, dict) and "scales" in scales and "flow" in scales:
            self.flow_data = scales
            self.warnings.extend(self.flow_data.get("warnings", []))
            scales = self.flow_data["scales"]

        assert all(
            key in scale.keys()
            for scale in scales
            for key in ["name", "type", "question", "items", "options"]
        ), "Each scale must have 'name', 'type', 'question', 'items', and 'options' keys."

        self.scales = {scale["name"]: scale for scale in scales}
        self._default_chatbot_prompts = self._load_default_chatbot_prompts()
        self._default_user_start_qs = self._load_default_user_start_qs()
        self.excluded_block_ids: set[str] = set()
        self.excluded_scale_names: set[str] = set()
        self.excluded_question_ids: set[str] = set()
        self.excluded_condition_tokens: set[str] = set()
        self.runtime_type_overrides: dict[str, str] = {}

    def get_scales(self, scales):
        if isinstance(scales, (list, tuple, set)):
            selected = list(scales)
        elif scales is not None:
            selected = [scales]
        else:
            selected = self._default_scale_names()
        return [
            scale_name
            for scale_name in selected
            if scale_name not in self.excluded_scale_names
        ]

    def _default_scale_names(self):
        if (
            self.flow_data is None
            or not self.excluded_block_ids
            and not self.excluded_question_ids
            and not self.excluded_scale_names
        ):
            return [
                scale_name
                for scale_name in self.scales.keys()
                if scale_name not in self.excluded_scale_names
            ]

        excluded_scales = set(self.excluded_scale_names)
        for block_id in self.excluded_block_ids:
            block = self.flow_data["blocks"].get(block_id, {})
            for qid in block.get("question_ids", []):
                meta = self.flow_data["questions"].get(qid, {})
                scale_name = meta.get("scale_name")
                if scale_name:
                    excluded_scales.add(scale_name)

        for qid in self.excluded_question_ids:
            meta = self.flow_data["questions"].get(qid, {})
            scale_name = meta.get("scale_name")
            if scale_name:
                excluded_scales.add(scale_name)

        return [scale_name for scale_name in self.scales.keys() if scale_name not in excluded_scales]

    def get_postamble(self, scale):
        if scale["type"] == "multiple choice":
            postamble = (
                'Return raw JSON using the item identifiers (shown in quotes) '
                'as keys. Values must correspond to: '
            )
            postamble += ", ".join(
                [f"{value} for {key}" for key, value in scale["options"].items()]
            )
            postamble += ". "
        elif scale["type"] == "ranking":
            postamble = (
                "Return raw JSON using the item identifiers (shown in quotes) "
                "as keys. Values must give the relative rank of the items "
                "from highest (1) to lowest (# of items). "
            )
        elif scale["type"] == "numeric":
            postamble = (
                "Return raw JSON using the item identifiers (shown in quotes) "
                "as keys. Values must be numeric and range from: "
            )
            postamble += " to ".join(
                [f"{value} for {key}" for key, value in scale["options"].items()]
            )
            postamble += ". "
        elif scale["type"] == "free text":
            postamble = (
                "Return raw JSON using the item identifiers (shown in quotes) "
                "as keys. Values must be strings written in the participant's own words. "
                "Escape any double quotes inside string values. "
            )
        else:
            raise ValueError(f"Invalid scale type: {scale['type']!r}")
        postamble += "Return only valid JSON."
        return postamble

    def _item_key_mask(self, scale):
        return {
            item_key: f"item_{index}"
            for index, item_key in enumerate(scale["items"].keys(), start=1)
        }

    def _scale_for_prompt(self, scale):
        key_mask = self._item_key_mask(scale)
        prompt_scale = dict(scale)
        prompt_scale["items"] = {
            key_mask[item_key]: item_label
            for item_key, item_label in scale["items"].items()
        }
        return prompt_scale

    def _remap_masked_assessment_keys(self, assessment, scale):
        if not isinstance(assessment, dict):
            return assessment

        reverse_key_mask = {
            masked_key: item_key
            for item_key, masked_key in self._item_key_mask(scale).items()
        }
        if not reverse_key_mask:
            return assessment

        remapped = {}
        for key, value in assessment.items():
            remapped[reverse_key_mask.get(key, key)] = value
        return remapped

    def _format_prompt_item(self, key, label):
        if label is None:
            return f'"{key}"'
        label = str(label).strip()
        if not label or label == key:
            return f'"{key}"'
        return f'"{key}" ({label})'

    def _get_answer(self, scale, user):
        preamble = ""
        prompt_scale = self._scale_for_prompt(scale)
        question = prompt_scale["question"] + " Items: "

        if len(prompt_scale["items"]) > 1:
            items = ", ".join(
                [
                    self._format_prompt_item(key, value)
                    for key, value in prompt_scale["items"].items()
                ]
            )
        else:
            key0 = next(iter(prompt_scale["items"].keys()))
            items = f'"{key0}"'

        text = preamble + question + items + ". " + self.get_postamble(prompt_scale)
        user.update("user", text, "")
        response = user._get_response(user.messages, max_tokens=self.max_tokens, json=True)
        return response

    def get_answer(self, scale, user):
        response = self._get_answer(scale, user)
        return self.format_answer(response, scale, user)

    def format_answer(self, response, scale, user):
        answer, thoughts = user.format_response(response)
        assessment_cleaned = self._clean_assessment_text(answer)

        try:
            assessment = self._parse_assessment_json(assessment_cleaned)
            try:
                assessment = self._remap_masked_assessment_keys(assessment, scale)
                assessment = self._normalize_assessment_dict(assessment, scale)
            except Exception as exc:
                assessment = None
                print(assessment_cleaned)
                print(f"Failed to parse dictionary: {exc}")
        except Exception as exc:
            assessment = self._recover_free_text_assessment(assessment_cleaned, scale)
            if assessment is None:
                print(assessment_cleaned)
                print(f"Failed to parse json: {exc}")

        return assessment, thoughts, answer

    def _recover_free_text_assessment(self, assessment_cleaned, scale):
        if scale["type"] != "free text" or len(scale["items"]) != 1:
            return None

        match = re.fullmatch(
            r'\{\s*"(?P<key>[^"]+)"\s*:\s*"(?P<value>.*)"\s*\}\s*',
            assessment_cleaned,
            re.DOTALL,
        )
        if match is None:
            return None

        value = match.group("value")
        value = value.replace('\\"', '"')
        value = value.replace("\\n", "\n").replace("\\r", "\r").replace("\\t", "\t")
        value = value.replace("\\\\", "\\")
        return self._normalize_assessment_dict({match.group("key"): value}, scale)

    def _clean_assessment_text(self, answer):
        assessment_cleaned = answer.replace("```json", "```").replace("```JSON", "```")
        assessment_cleaned = assessment_cleaned.replace("```", "").strip()
        assessment_cleaned = assessment_cleaned.replace("“", '"').replace("”", '"')
        assessment_cleaned = assessment_cleaned.replace("‘", "'").replace("’", "'")
        assessment_cleaned = re.sub(
            r'["\']{2,}(\w+)["\']{2,}',
            r'"\1"',
            assessment_cleaned,
        ).strip()
        return assessment_cleaned

    def _parse_assessment_json(self, assessment_cleaned):
        candidates = [assessment_cleaned]
        match = re.search(r"\{.*\}", assessment_cleaned, re.DOTALL)
        if match and match.group(0) not in candidates:
            candidates.append(match.group(0))

        for candidate in candidates:
            try:
                return json.loads(candidate)
            except Exception:
                try:
                    return ast.literal_eval(candidate)
                except Exception:
                    continue

        raise ValueError("Could not parse response as a JSON object.")

    def _normalize_assessment_dict(self, assessment, scale):
        if not isinstance(assessment, dict):
            raise TypeError("Assessment must be a JSON object.")

        assessment = self._remap_masked_assessment_keys(assessment, scale)
        expected_keys = list(scale["items"].keys())
        if not expected_keys:
            return {}

        if len(expected_keys) == 1:
            key = expected_keys[0]
            value = self._resolve_single_item_value(assessment, scale, key)
            return {key: self._coerce_answer_value(value, scale)}

        alias_map = self._build_item_alias_map(scale, assessment)
        normalized = {}
        for item_key in expected_keys:
            matched_key = alias_map.get(self._canonicalize_key(item_key))
            if matched_key is None:
                raise KeyError(item_key)
            normalized[item_key] = self._coerce_answer_value(assessment[matched_key], scale)
        return normalized

    def _resolve_single_item_value(self, assessment, scale, item_key):
        direct_candidates = [
            item_key,
            scale.get("name"),
            scale["items"].get(item_key),
            "answer",
            "response",
            "value",
            "selection",
            "selected",
        ]
        alias_map = self._build_item_alias_map(scale, assessment)
        for candidate in direct_candidates:
            if candidate is None:
                continue
            matched_key = alias_map.get(self._canonicalize_key(candidate))
            if matched_key is not None:
                return assessment[matched_key]

        if len(assessment) == 1:
            return next(iter(assessment.values()))

        raise KeyError(item_key)

    def _build_item_alias_map(self, scale, assessment=None):
        alias_map = {}
        if assessment is None:
            assessment = {}

        for raw_key in assessment.keys():
            alias_map[self._canonicalize_key(raw_key)] = raw_key

        alias_sources = [scale.get("name")]
        alias_sources.extend(scale["items"].keys())
        alias_sources.extend(scale["items"].values())
        for alias in alias_sources:
            if alias is None:
                continue
            canonical = self._canonicalize_key(alias)
            if canonical in alias_map:
                continue
            for raw_key in assessment.keys():
                if self._canonicalize_key(raw_key) == canonical:
                    alias_map[canonical] = raw_key
                    break
        return alias_map

    def _canonicalize_key(self, value):
        return re.sub(r"[^a-z0-9]+", "", str(value).lower())

    def _coerce_answer_value(self, value, scale):
        if scale["type"] == "free text":
            if value is None:
                return None
            if isinstance(value, str):
                return value.strip()
            if isinstance(value, (int, float, bool)):
                return str(value)
            return json.dumps(value, ensure_ascii=False)

        parsed = get_int(value)
        if parsed is not None:
            return parsed

        if scale["type"] == "multiple choice" and isinstance(value, str):
            normalized_options = {
                self._canonicalize_key(label): option_value
                for label, option_value in scale.get("options", {}).items()
            }
            canonical_value = self._canonicalize_key(value)
            if canonical_value in normalized_options:
                return int(normalized_options[canonical_value])

            match = re.search(r"-?\d+", value)
            if match:
                extracted = int(match.group(0))
                if extracted in set(normalized_options.values()):
                    return extracted

        return None

    def describe_flow_blocks(self):
        if self.flow_data is None:
            return []

        described = []
        seen = set()

        def walk(nodes):
            for node in nodes:
                node_type = node.get("Type")
                if node_type in {"Block", "Standard"}:
                    block_id = node.get("ID")
                    if block_id in seen:
                        continue
                    seen.add(block_id)
                    block = self.flow_data["blocks"].get(block_id, {})
                    question_ids = block.get("question_ids", [])
                    runtime_counts = {}
                    for qid in question_ids:
                        runtime_type = self._get_question_runtime_type(qid=qid)
                        runtime_counts[runtime_type] = runtime_counts.get(runtime_type, 0) + 1
                    described.append(
                        {
                            "block_id": block_id,
                            "description": block.get("description", ""),
                            "question_count": len(question_ids),
                            "question_ids": question_ids,
                            "runtime_summary": runtime_counts,
                            "excluded": block_id in self.excluded_block_ids,
                        }
                    )
                walk(node.get("Flow", []))

        walk(self.flow_data.get("flow", {}).get("Flow", []))
        return described

    def _flow_node_display_name(self, node):
        node_type = node.get("Type")
        if node_type in {"Block", "Standard"}:
            block = self.flow_data["blocks"].get(node.get("ID"), {})
            return block.get("description") or node.get("Description") or node.get("ID") or "Block"
        if node_type == "Group":
            return node.get("Description") or node.get("FlowID") or "Group"
        if node_type == "Branch":
            return node.get("Description") or node.get("FlowID") or "Branch"
        if node_type == "BlockRandomizer":
            return node.get("FlowID") or "BlockRandomizer"
        if node_type == "EmbeddedData":
            return node.get("FlowID") or "EmbeddedData"
        return node.get("Description") or node.get("FlowID") or node_type or "FlowNode"

    def describe_flow_randomizers(self):
        if self.flow_data is None:
            return []

        described = []

        def walk(nodes, path_parts):
            for node in nodes:
                node_type = node.get("Type")
                node_name = self._flow_node_display_name(node)
                next_path = [*path_parts, node_name]

                if node_type == "BlockRandomizer":
                    active_children = self._get_active_randomizer_children(node)
                    options = [
                        self._randomizer_child_option_label(child)
                        for child in active_children
                    ]
                    excluded_options = [
                        self._randomizer_child_option_label(child)
                        for index, child in enumerate(node.get("Flow", []))
                        if self._is_randomizer_child_excluded(
                            child,
                            index=index,
                            parent_flow_id=node.get("FlowID"),
                        )
                    ]
                    described.append(
                        {
                            "flow_id": node.get("FlowID"),
                            "path": " > ".join(next_path),
                            "subset": min(int(node.get("SubSet", len(active_children))), len(active_children)),
                            "option_count": len(options),
                            "even_presentation": bool(node.get("EvenPresentation")),
                            "options": options,
                            "excluded_options": excluded_options,
                        }
                    )

                walk(node.get("Flow", []), next_path)

        walk(self.flow_data.get("flow", {}).get("Flow", []), ["Survey Flow"])
        return described

    def set_excluded_blocks(self, block_refs=None):
        if self.flow_data is None:
            self.excluded_block_ids = set()
            return []

        if not block_refs:
            self.excluded_block_ids = set()
            return []

        blocks = self.flow_data["blocks"]
        description_lookup = {}
        for block_id, block in blocks.items():
            description = (block.get("description") or "").strip()
            if description:
                description_lookup.setdefault(self._canonicalize_key(description), set()).add(block_id)

        excluded = set()
        for ref in block_refs:
            if ref in blocks:
                excluded.add(ref)
                continue
            canonical = self._canonicalize_key(ref)
            matches = description_lookup.get(canonical, set())
            if matches:
                excluded.update(matches)
                continue
            raise KeyError(f"Unknown block reference: {ref}")

        self.excluded_block_ids = excluded
        return sorted(self.excluded_block_ids)

    def _question_reference_lookup(self):
        if self.flow_data is None:
            return {}

        lookup = {}
        for qid, meta in self.flow_data["questions"].items():
            for raw_ref in (
                qid,
                meta.get("data_export_tag"),
                meta.get("scale_name"),
                meta.get("description"),
                meta.get("text"),
            ):
                if raw_ref is None:
                    continue
                text = str(raw_ref).strip()
                if not text:
                    continue
                lookup.setdefault(self._canonicalize_key(text), set()).add(qid)
        return lookup

    def _randomizer_child_option_label(self, child):
        return (
            child.get("Description")
            or self.flow_data["blocks"].get(child.get("ID"), {}).get("description")
            or child.get("ID")
            or child.get("FlowID")
            or child.get("Type")
        )

    def _randomizer_child_reference_lookup(self):
        if self.flow_data is None:
            return {}

        lookup = {}
        for node in self._iter_randomizer_nodes():
            parent_flow_id = node.get("FlowID")
            for index, child in enumerate(node.get("Flow", [])):
                token = self._child_selection_token(
                    child,
                    index=index,
                    parent_flow_id=parent_flow_id,
                )
                refs = {
                    token,
                    child.get("FlowID"),
                    child.get("ID"),
                    child.get("Description"),
                    self._randomizer_child_option_label(child),
                }
                refs.update(self._collect_condition_tokens_from_node(child))
                for raw_ref in refs:
                    if raw_ref is None:
                        continue
                    text = str(raw_ref).strip()
                    if not text:
                        continue
                    lookup.setdefault(self._canonicalize_key(text), set()).add(token)
        return lookup

    def _is_randomizer_child_excluded(self, child, index=0, parent_flow_id=None):
        token = self._child_selection_token(
            child,
            index=index,
            parent_flow_id=parent_flow_id,
        )
        return token in self.excluded_condition_tokens

    def _get_active_randomizer_children(self, node):
        parent_flow_id = node.get("FlowID")
        children = []
        for index, child in enumerate(node.get("Flow", [])):
            if child.get("Type") in {"Block", "Standard"} and child.get("ID") in self.excluded_block_ids:
                continue
            if self._is_randomizer_child_excluded(
                child,
                index=index,
                parent_flow_id=parent_flow_id,
            ):
                continue
            children.append(child)
        return children

    def _iter_randomizer_nodes(self):
        if self.flow_data is None:
            return []

        nodes = []

        def walk(flow_nodes):
            for node in flow_nodes:
                if node.get("Type") == "BlockRandomizer":
                    nodes.append(node)
                walk(node.get("Flow", []))

        walk(self.flow_data.get("flow", {}).get("Flow", []))
        return nodes

    def _resolve_question_refs(self, question_refs=None):
        if self.flow_data is None:
            return set()

        if not question_refs:
            return set()

        lookup = self._question_reference_lookup()
        resolved = set()
        for ref in question_refs:
            if ref in self.flow_data["questions"]:
                resolved.add(ref)
                continue

            canonical = self._canonicalize_key(ref)
            matches = lookup.get(canonical, set())
            if matches:
                resolved.update(matches)
                continue
            raise KeyError(f"Unknown question reference: {ref}")
        return resolved

    def _get_question_runtime_type(self, qid=None, meta=None):
        if meta is None:
            if self.flow_data is None or qid is None:
                return "unknown"
            meta = self.flow_data["questions"].get(qid, {})
        if qid is None:
            qid = meta.get("question_id")

        if qid in self.runtime_type_overrides:
            return self.runtime_type_overrides[qid]
        return meta.get("runtime_type", "unknown")

    def set_excluded_scales(self, scale_refs=None):
        if not scale_refs:
            self.excluded_scale_names = set()
            return []

        excluded = set()
        for ref in scale_refs:
            if ref in self.scales:
                excluded.add(ref)
                continue

            canonical = self._canonicalize_key(ref)
            matches = [
                scale_name
                for scale_name in self.scales.keys()
                if self._canonicalize_key(scale_name) == canonical
            ]
            if matches:
                excluded.update(matches)
                continue
            raise KeyError(f"Unknown scale reference: {ref}")

        self.excluded_scale_names = excluded
        return sorted(self.excluded_scale_names)

    def set_excluded_questions(self, question_refs=None):
        if self.flow_data is None:
            self.excluded_question_ids = set()
            return []

        if not question_refs:
            self.excluded_question_ids = set()
            return []

        self.excluded_question_ids = self._resolve_question_refs(question_refs)
        return sorted(self.excluded_question_ids)

    def set_excluded_interventions(self, question_refs=None):
        if self.flow_data is None:
            self.excluded_question_ids = set()
            return []

        if not question_refs:
            intervention_qids = {
                qid
                for qid in self.excluded_question_ids
                if self._get_question_runtime_type(qid=qid) == "intervention"
            }
            self.excluded_question_ids -= intervention_qids
            return sorted(intervention_qids)

        intervention_qids = self._resolve_question_refs(question_refs)
        invalid_qids = sorted(
            qid
            for qid in intervention_qids
            if self._get_question_runtime_type(qid=qid) != "intervention"
        )
        if invalid_qids:
            invalid_refs = [
                self.flow_data["questions"].get(qid, {}).get("data_export_tag") or qid
                for qid in invalid_qids
            ]
            raise ValueError(
                "Intervention exclusion only accepts questions currently classified as "
                f"intervention: {invalid_refs}"
            )

        self.excluded_question_ids.update(intervention_qids)
        return sorted(intervention_qids)

    def set_excluded_conditions(self, condition_refs=None):
        if self.flow_data is None:
            self.excluded_condition_tokens = set()
            return []

        if not condition_refs:
            self.excluded_condition_tokens = set()
            return []

        lookup = self._randomizer_child_reference_lookup()
        excluded = set()
        for ref in condition_refs:
            canonical = self._canonicalize_key(ref)
            matches = lookup.get(canonical, set())
            if matches:
                excluded.update(matches)
                continue
            raise KeyError(f"Unknown condition reference: {ref}")

        self.excluded_condition_tokens = excluded
        return sorted(self.excluded_condition_tokens)

    def set_runtime_type_overrides(self, overrides=None):
        if self.flow_data is None:
            self.runtime_type_overrides = {}
            return {}

        if not overrides:
            self.runtime_type_overrides = {}
            return {}

        allowed = {"scale", "display", "intervention", "text_entry", "passive", "unsupported"}
        normalized = {}
        for ref, runtime_type in dict(overrides).items():
            if runtime_type not in allowed:
                raise ValueError(
                    f"Invalid runtime_type override {runtime_type!r} for {ref!r}. "
                    f"Expected one of: {sorted(allowed)}"
                )
            matches = self._resolve_question_refs([ref])
            for qid in matches:
                normalized[qid] = runtime_type

        self.runtime_type_overrides = normalized
        return dict(self.runtime_type_overrides)

    def get_score(self, scale, assessment):
        if assessment is None or "answers" not in scale:
            return None

        if scale["type"] == "multiple choice":
            return abs_distance(scale["answers"], assessment)
        if scale["type"] == "ranking":
            return footrule_distance(scale["answers"], assessment)
        if scale["type"] == "numeric":
            return abs_distance(scale["answers"], assessment)
        if scale["type"] == "free text":
            return None
        raise ValueError(f"Invalid scale type: {scale['type']!r}")

    # ------------------------------------------------------------------
    # Flow execution helpers
    # ------------------------------------------------------------------

    def _coerce_rng(self, random_state=None):
        return random_state if random_state is not None else Random(42)

    def _shuffle(self, items, random_state):
        items = list(items)
        if hasattr(random_state, "shuffle"):
            random_state.shuffle(items)
            return items
        if isinstance(random_state, Random):
            random_state.shuffle(items)
            return items
        rng = Random(42)
        rng.shuffle(items)
        return items

    def _choice_subset(self, items, count, random_state):
        items = list(items)
        if count >= len(items):
            return self._shuffle(items, random_state)
        if hasattr(random_state, "choice"):
            selected = random_state.choice(items, size=count, replace=False)
            return list(selected)
        return self._shuffle(items, random_state)[:count]

    def _get_user_condition_value(self, user, state):
        user_params = getattr(user, "user_params", {}) or {}
        for key in (
            "condition",
            "Condition",
            "survey_embedded_label_Condition",
            "survey_embedded_Condition",
        ):
            value = user_params.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        existing = state["embedded_labels"].get("Condition") or state["embedded_data"].get("Condition")
        if existing is not None and str(existing).strip():
            return str(existing).strip()
        return None

    def _condition_word_tokens(self, value):
        words = re.findall(r"[a-z0-9]+", str(value).lower())
        return {word for word in words if word not in {"llm", "condition"}}

    def _condition_values_match(self, desired, candidate):
        return self._condition_match_score(desired, candidate) is not None

    def _condition_match_score(self, desired, candidate):
        desired_key = self._canonicalize_key(desired)
        candidate_key = self._canonicalize_key(candidate)
        if desired_key and desired_key == candidate_key:
            return (3, len(desired_key), 0)

        desired_tokens = self._condition_word_tokens(desired)
        candidate_tokens = self._condition_word_tokens(candidate)
        if desired_tokens and candidate_tokens:
            overlap = len(desired_tokens & candidate_tokens)
            if overlap > 0 and (
                desired_tokens.issubset(candidate_tokens) or candidate_tokens.issubset(desired_tokens)
            ):
                return (2, overlap, -abs(len(desired_tokens) - len(candidate_tokens)))

        return None

    def _collect_condition_tokens_from_node(self, node):
        tokens = set()

        def add_token(value):
            if value is None:
                return
            text = str(value).strip()
            if text:
                tokens.add(text)
                tokens.add(self._canonicalize_key(text))

        add_token(node.get("Description"))
        add_token(node.get("ID"))

        if node.get("Type") == "EmbeddedData":
            for field in node.get("EmbeddedData", []):
                if field.get("Field") != "Condition":
                    continue
                add_token(field.get("Value"))
                add_token(field.get("Description"))

        for child in node.get("Flow", []):
            tokens.update(self._collect_condition_tokens_from_node(child))

        return tokens

    def _select_condition_aligned_children(self, children, subset, user, state, random_state):
        desired = self._get_user_condition_value(user, state)
        if not desired:
            return None, None

        if not self._canonicalize_key(desired):
            return None, None

        matched_child = None
        matched_score = None
        for child in children:
            child_tokens = self._collect_condition_tokens_from_node(child)
            child_score = None
            for token in child_tokens:
                score = self._condition_match_score(desired, token)
                if score is not None and (child_score is None or score > child_score):
                    child_score = score
            if child_score is not None and (matched_score is None or child_score > matched_score):
                matched_child = child
                matched_score = child_score

        if matched_child is None:
            return None, desired

        remaining = [child for child in children if child is not matched_child]
        if subset <= 1:
            return [matched_child], desired

        extra_children = self._choice_subset(remaining, max(0, subset - 1), random_state)
        return [matched_child, *extra_children], desired

    def _child_selection_token(self, child, index=0, parent_flow_id=None):
        return (
            child.get("FlowID")
            or child.get("ID")
            or f"{parent_flow_id or 'flow'}:{index}"
        )

    def _resolve_planned_randomizer_selection(self, node, children, runtime_context):
        if not runtime_context:
            return None

        plan = runtime_context.get("even_presentation_plan") or {}
        selected_tokens = plan.get(node.get("FlowID"))
        if not selected_tokens:
            return None

        token_lookup = {
            self._child_selection_token(child, index=index, parent_flow_id=node.get("FlowID")): child
            for index, child in enumerate(children)
        }
        selected = [token_lookup[token] for token in selected_tokens if token in token_lookup]
        return selected or None

    def _materialize_block_question_ids(self, block, random_state):
        question_ids = list(block.get("question_ids", []))
        options = block.get("options", {})
        if options.get("RandomizeQuestions") != "Advanced":
            return question_ids

        advanced = block.get("randomization", {}).get("Advanced", {})
        fixed_order = self._normalize_ordered_values(advanced.get("FixedOrder"))
        randomize_all = self._normalize_ordered_values(advanced.get("RandomizeAll"))
        undisplayed = set(self._normalize_ordered_values(advanced.get("Undisplayed")))

        randomisable = [
            qid for qid in randomize_all if qid in question_ids and qid not in undisplayed
        ]
        randomisable = self._shuffle(randomisable, random_state)

        ordered = []
        for token in fixed_order:
            if token == "{~Randomized~}":
                if randomisable:
                    ordered.append(randomisable.pop(0))
            elif token in question_ids and token not in undisplayed:
                ordered.append(token)

        for qid in randomisable:
            if qid not in ordered:
                ordered.append(qid)

        for qid in question_ids:
            if qid not in ordered and qid not in undisplayed:
                ordered.append(qid)
        return ordered

    def _normalize_ordered_values(self, raw):
        if raw is None:
            return []
        if isinstance(raw, list):
            return [str(item) for item in raw]
        if isinstance(raw, dict):
            try:
                keys = sorted(raw, key=lambda key: int(str(key)))
            except ValueError:
                keys = list(raw.keys())
            return [str(raw[key]) for key in keys]
        return [str(raw)]

    def _resolve_embedded_value(self, value, random_state):
        if not isinstance(value, str):
            return value

        match = re.fullmatch(r"\$\{rand://int/(\d+):(\d+)\}", value)
        if match:
            low, high = int(match.group(1)), int(match.group(2))
            if hasattr(random_state, "integers"):
                return str(int(random_state.integers(low, high + 1)))
            if hasattr(random_state, "randint"):
                return str(int(random_state.randint(low, high)))
            return str(Random(42).randint(low, high))
        return value

    def _stringify_piped_value(self, value):
        if value is None:
            return None
        if isinstance(value, (list, tuple, set)):
            return ", ".join(str(item) for item in value if item is not None)
        if isinstance(value, dict):
            return ", ".join(str(item) for item in value.values() if item is not None)
        return str(value)

    def _selected_choice_labels_for_assessment(self, meta, assessment):
        if not assessment:
            return None

        scale_name = meta.get("scale_name")
        scale = self.scales.get(scale_name)
        if not scale:
            return None

        value_to_label = {
            str(option_value): option_label
            for option_label, option_value in scale.get("options", {}).items()
        }
        labels = []
        for value in assessment.values():
            label = value_to_label.get(str(value))
            if label is not None:
                labels.append(label)

        if not labels:
            return None
        if len(labels) == 1:
            return labels[0]
        return labels

    def _resolve_question_pipe(self, expression, state):
        parts = expression.split("/")
        if len(parts) < 4 or parts[0] != "q:" or parts[1] != "":
            return None

        qid = parts[2]
        pipe_path = "/".join(parts[3:])
        if pipe_path == "ChoiceGroup/SelectedChoices":
            return self._stringify_piped_value(
                state.get("question_selected_labels", {}).get(qid)
            )
        if pipe_path in {"ChoiceGroup/SelectedChoicesText", "ChoiceGroup/SelectedAnswers"}:
            return self._stringify_piped_value(
                state.get("question_selected_labels", {}).get(qid)
            )
        if pipe_path in {"QuestionText", "QuestionDescription"}:
            meta = self.flow_data.get("questions", {}).get(qid, {}) if self.flow_data else {}
            return meta.get("text") or meta.get("description")
        return None

    def _resolve_embedded_pipe(self, expression, state):
        parts = expression.split("/")
        if len(parts) < 4 or parts[0] != "e:" or parts[1] != "" or parts[2] != "Field":
            return None
        field_name = "/".join(parts[3:])
        if field_name in state.get("embedded_data", {}):
            return self._stringify_piped_value(state["embedded_data"].get(field_name))
        if field_name in state.get("embedded_labels", {}):
            return self._stringify_piped_value(state["embedded_labels"].get(field_name))
        return None

    def _resolve_piped_text(self, text, state):
        if not isinstance(text, str) or "${" not in text:
            return text

        unresolved = []

        def replace(match):
            token = match.group(0)
            expression = match.group(1)
            if expression.startswith("q://"):
                value = self._resolve_question_pipe(expression, state)
            elif expression.startswith("e://"):
                value = self._resolve_embedded_pipe(expression, state)
            else:
                value = None

            if value is None:
                unresolved.append(token)
                return token
            return value

        resolved = re.sub(r"\$\{([^}]+)\}", replace, text)
        for token in unresolved:
            warning = f"Could not resolve piped text token {token!r}; leaving it unchanged."
            if warning not in state["warnings"]:
                state["warnings"].append(warning)
        return resolved

    def _extract_choice_value(self, assessment):
        if not assessment:
            return None
        values = list(assessment.values())
        return values[0] if len(values) == 1 else None

    def _coerce_numeric_branch_value(self, value):
        parsed = get_int(value)
        if parsed is not None:
            return parsed
        try:
            return float(value)
        except Exception:
            return None

    def _is_empty_branch_value(self, value):
        if value is None:
            return True
        if isinstance(value, str):
            return value.strip() == ""
        if isinstance(value, (list, tuple, set, dict)):
            return len(value) == 0
        return False

    def _branch_ignore_case(self, expression):
        return str(expression.get("IgnoreCase", "")).lower() in {"1", "true", "yes"}

    def _resolve_branch_choice_value(self, qid, raw_choice_id):
        raw_choice_key = str(raw_choice_id)
        meta = self.flow_data.get("questions", {}).get(qid, {}) if self.flow_data else {}
        payload = meta.get("raw_payload", {})
        recode = payload.get("RecodeValues") or {}
        value = recode.get(raw_choice_key, raw_choice_id)
        parsed = get_int(value)
        return parsed if parsed is not None else value

    def _branch_value_matches_choice(self, left, right, ignore_case=False):
        if isinstance(left, dict):
            return any(
                self._branch_value_matches_choice(value, right, ignore_case)
                for value in left.values()
            )
        if isinstance(left, (list, tuple, set)):
            return any(
                self._branch_value_matches_choice(value, right, ignore_case)
                for value in left
            )

        left_text = str(left)
        right_text = str(right)
        if ignore_case:
            left_text = left_text.lower()
            right_text = right_text.lower()
        return left_text == right_text

    def _branch_value_contains_text(self, left, right, ignore_case=False):
        if isinstance(left, dict):
            return any(
                self._branch_value_contains_text(value, right, ignore_case)
                for value in left.values()
            )
        if isinstance(left, (list, tuple, set)):
            return any(
                self._branch_value_contains_text(value, right, ignore_case)
                for value in left
            )

        left_text = str(left)
        right_text = str(right)
        if ignore_case:
            left_text = left_text.lower()
            right_text = right_text.lower()
        return right_text in left_text

    def _evaluate_expression(self, expression, state):
        logic_type = expression.get("LogicType")
        operator = expression.get("Operator")
        ignore_case = self._branch_ignore_case(expression)
        is_choice_comparison = False

        if logic_type == "Question":
            qid = expression.get("QuestionID")
            left = state["question_values"].get(qid)
            locator = expression.get("ChoiceLocator", "")
            match = re.search(r"/SelectableChoice/(\d+)", locator)
            is_choice_comparison = match is not None
            right = (
                self._resolve_branch_choice_value(qid, match.group(1))
                if match
                else expression.get("RightOperand")
            )
        elif logic_type == "EmbeddedField":
            field = expression.get("LeftOperand")
            left = state["embedded_data"].get(field)
            right = expression.get("RightOperand")
        else:
            raise UnsupportedSurveyFlowError(
                f"Unsupported branch logic type: {logic_type!r}."
            )

        if operator == "Empty":
            return self._is_empty_branch_value(left)
        if operator == "NotEmpty":
            return not self._is_empty_branch_value(left)
        if left is None:
            return None

        if operator == "Selected":
            return self._branch_value_matches_choice(left, right, ignore_case)
        if operator == "NotSelected":
            return not self._branch_value_matches_choice(left, right, ignore_case)
        if operator == "DoesContain":
            if is_choice_comparison:
                return self._branch_value_matches_choice(left, right, ignore_case)
            return self._branch_value_contains_text(left, right, ignore_case)
        if operator == "DoesNotContain":
            if is_choice_comparison:
                return not self._branch_value_matches_choice(left, right, ignore_case)
            return not self._branch_value_contains_text(left, right, ignore_case)
        if operator == "EqualTo":
            return str(left) == str(right)
        if operator == "NotEqualTo":
            return str(left) != str(right)
        if operator in {"GreaterThan", "GreaterThanOrEqual", "LessThan", "LessThanOrEqual"}:
            left_num = self._coerce_numeric_branch_value(left)
            right_num = self._coerce_numeric_branch_value(right)
            if left_num is None or right_num is None:
                return None
            if operator == "GreaterThan":
                return left_num > right_num
            if operator == "GreaterThanOrEqual":
                return left_num >= right_num
            if operator == "LessThan":
                return left_num < right_num
            return left_num <= right_num
        raise UnsupportedSurveyFlowError(
            f"Unsupported branch operator: {operator!r}."
        )

    def _evaluate_boolean_group(self, group, state):
        result = None
        for key in sorted(k for k in group.keys() if str(k).isdigit()):
            expression = group[key]
            current = self._evaluate_expression(expression, state)
            if current is None:
                return None
            if result is None:
                result = current
                continue
            conjunction = expression.get("Conjuction", "And")
            if conjunction == "Or":
                result = result or current
            else:
                result = result and current
        return result

    def _evaluate_branch(self, branch_logic, state):
        if not branch_logic:
            return None
        for key in sorted(k for k in branch_logic.keys() if str(k).isdigit()):
            child = branch_logic[key]
            if child.get("Type") == "If":
                return self._evaluate_boolean_group(child, state)
        return None

    def _resolve_intervention_override(self, meta, state, intervention_overrides):
        if not intervention_overrides:
            return None

        config = self._normalize_intervention_config(intervention_overrides)
        conditions = config["conditions"]

        keys = []
        condition_value = state["embedded_data"].get("Condition")
        condition_label = state["embedded_labels"].get("Condition")

        if condition_label:
            keys.extend([condition_label, f"Condition:{condition_label}"])
        if condition_value is not None:
            keys.extend(
                [
                    f"Condition={condition_value}",
                    f"Condition:{condition_value}",
                    str(condition_value),
                ]
            )
        keys.extend(
            [
                meta["question_id"],
                meta["data_export_tag"],
            ]
        )

        for key in keys:
            if key in conditions:
                override = conditions[key]
                if isinstance(override, dict):
                    merged = dict(config["defaults"])
                    merged.update(override)
                    return merged
                return override
        return None

    def _normalize_intervention_config(self, intervention_overrides):
        def normalize_override_value(key, value):
            if not isinstance(value, str):
                return value

            stripped = value.strip()
            lowered = stripped.lower()

            if key in {"base_url", "model", "api_secret", "prompt_path"} and lowered in {
                "",
                "none",
                "null",
            }:
                return None

            if key in {"collect_participant_opener", "participant_opener_only"}:
                if lowered in {"true", "1", "yes", "on"}:
                    return True
                if lowered in {"false", "0", "no", "off", "", "none", "null"}:
                    return False

            return value

        def normalize_override_mapping(mapping):
            normalized = {}
            for key, value in mapping.items():
                normalized[key] = normalize_override_value(key, value)
            return normalized

        if (
            isinstance(intervention_overrides, dict)
            and "conditions" in intervention_overrides
        ):
            return {
                "defaults": normalize_override_mapping(
                    dict(intervention_overrides.get("defaults", {}))
                ),
                "conditions": {
                    key: (
                        normalize_override_mapping(value)
                        if isinstance(value, dict)
                        else value
                    )
                    for key, value in dict(intervention_overrides.get("conditions", {})).items()
                },
            }
        return {
            "defaults": {},
            "conditions": {
                key: (
                    normalize_override_mapping(value)
                    if isinstance(value, dict)
                    else value
                )
                for key, value in dict(intervention_overrides).items()
            },
        }

    def _load_default_chatbot_prompts(self):
        base_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "example", "prompts", "chatbot")
        )
        prompt_map = {}
        for name, filename in {
            "baseline": "baseline.txt",
            "bridging": "bridging.txt",
            "personalization": "personalization.txt",
            "pers_bridge": "pers_bridge.txt",
        }.items():
            path = os.path.join(base_dir, filename)
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as handle:
                    prompt_map[name] = handle.read()
        return prompt_map

    def _load_default_user_start_qs(self):
        path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "example",
                "prompts",
                "participant",
                "user_pol_short.txt",
            )
        )
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as handle:
                return handle.read()
        return (
            "Ask a short question about a politically polarizing topic in the US."
        )

    def _load_default_chatbot_opening_prompt(self):
        return (
            "Start the conversation with one short message inviting the user to "
            "discuss a politically polarizing topic in the US."
        )

    def _default_prompt_for_condition(self, label):
        label = (label or "").strip().lower()
        if "bridg" in label and "personal" in label:
            return self._default_chatbot_prompts.get("pers_bridge", "")
        if "personal" in label:
            return self._default_chatbot_prompts.get("personalization", "")
        if "bridg" in label:
            return self._default_chatbot_prompts.get("bridging", "")
        return self._default_chatbot_prompts.get("baseline", "")

    def _personalize_intervention_text(self, text, user):
        if not text:
            return text
        if user is None or not hasattr(user, "format_personalization"):
            return text
        return user.format_personalization(text)

    def _simulate_chat_intervention(self, override, user, meta, state):
        label = override.get("label") or state["embedded_labels"].get("Condition") or ""
        base_url = override.get("base_url")
        model = override.get("model")
        api_secret = override.get("api_secret")

        if not base_url or not model or not api_secret:
            collect_participant_opener = bool(
                override.get("collect_participant_opener")
                or override.get("participant_opener_only")
            )
            text = override.get("text") or meta["description"] or meta["text"] or ""
            text = self._resolve_piped_text(text, state)
            text = self._personalize_intervention_text(text, user)
            user_start_qs = self._personalize_intervention_text(
                override.get("user_start_qs", self._default_user_start_qs),
                user,
            )
            transcript = {
                "question_id": meta["question_id"],
                "key": meta["data_export_tag"],
                "source": "config_control",
                "label": label,
                "first_speaker": "participant" if collect_participant_opener else None,
                "prompt_source": "control_text",
                "prompt_path": None,
                "messages": [],
                "user_messages": [],
                "chatbot_messages": [],
            }
            if collect_participant_opener:
                user_message, user_thoughts = user.start_convo(user_start_qs)
                user.update("assistant", user_message, user_thoughts)
                transcript["messages"].append(
                    {
                        "speaker": "participant",
                        "role": "user",
                        "content": user_message,
                        "thoughts": user_thoughts,
                    }
                )
            if text:
                user.update("user", text, "")
                transcript["messages"].append(
                    {
                        "speaker": "control",
                        "role": "assistant",
                        "content": text,
                        "thoughts": "",
                    }
                )
            transcript["user_messages"] = copy.deepcopy(getattr(user, "messages", []))
            if transcript["messages"]:
                state["intervention_messages"].append(transcript)
            state["trace"].append(
                {
                    "type": "intervention",
                    "question_id": meta["question_id"],
                    "key": meta["data_export_tag"],
                    "source": "config_control",
                    "label": label,
                    "message_count": len(transcript["messages"]),
                }
            )
            state["pending_completion_code"] = override.get("completion_code", "SIMULATED_CONTROL")
            return

        env_name = str(api_secret)
        if env_name not in os.environ:
            raise MissingInterventionConfigError(
                f"Missing environment variable {env_name} for intervention "
                f"{meta['question_id']} ({label or meta['data_export_tag']})."
            )

        client = OpenAI(base_url=base_url, api_key=os.environ[env_name])
        chatbot = Chatbot(
            client,
            user.user_params.copy(),
            model,
            override.get("temperature", 0.6),
            override.get("max_tokens", 1000),
        )

        if "system_prompt" in override:
            system_prompt = override.get("system_prompt", "")
            prompt_source = "system_prompt" if system_prompt else "none"
        elif override.get("prompt_path"):
            with open(override["prompt_path"], "r", encoding="utf-8") as handle:
                system_prompt = handle.read()
            prompt_source = "prompt_path"
        else:
            system_prompt = ""
            prompt_source = "none"
        system_prompt = self._personalize_intervention_text(system_prompt, user)

        chatbot.format_messages(system_prompt)
        requested_first_speaker = str(
            override.get("first_speaker", "participant")
        ).strip().lower()
        if requested_first_speaker not in {"participant", "chatbot"}:
            requested_first_speaker = "participant"
        first_speaker = requested_first_speaker
        user_start_qs = self._personalize_intervention_text(
            override.get("user_start_qs", self._default_user_start_qs),
            user,
        )
        chatbot_opening_prompt = override.get(
            "chatbot_opening_prompt",
            self._load_default_chatbot_opening_prompt(),
        )
        chatbot_opening_prompt = self._personalize_intervention_text(
            chatbot_opening_prompt,
            user,
        )
        n_turns = int(override.get("n_turns", 5))
        transcript = {
            "question_id": meta["question_id"],
            "key": meta["data_export_tag"],
            "source": "chatbot_config",
            "label": label,
            "model": model,
            "temperature": override.get("temperature", 0.6),
            "max_tokens": override.get("max_tokens", 1000),
            "first_speaker": first_speaker,
            "requested_first_speaker": requested_first_speaker,
            "prompt_source": prompt_source,
            "prompt_path": override.get("prompt_path"),
            "messages": [],
            "user_messages": [],
            "chatbot_messages": [],
        }
        user_message_history = copy.deepcopy(getattr(user, "messages", []))

        for turn in range(n_turns):
            if first_speaker == "chatbot":
                chatbot_request_messages = chatbot.messages
                if turn == 0:
                    chatbot_request_messages = list(chatbot.messages)
                    chatbot_request_messages.append(
                        {"role": "user", "content": chatbot_opening_prompt}
                    )

                chatbot_response, chatbot_thoughts = chatbot.get_response(chatbot_request_messages)
                chatbot.update("assistant", chatbot_response, chatbot_thoughts)
                user.update("user", chatbot_response, chatbot_thoughts)
                user_message_history.append(
                    {"role": "user", "content": chatbot_response}
                )
                transcript["messages"].append(
                    {
                        "speaker": "chatbot",
                        "role": "assistant",
                        "content": chatbot_response,
                        "thoughts": chatbot_thoughts,
                    }
                )

                user_message, user_thoughts = user.get_response(user.messages)
                chatbot.update("user", user_message, user_thoughts)
                user.update("assistant", user_message, user_thoughts)
                user_message_history.append(
                    {"role": "assistant", "content": user_message}
                )
                transcript["messages"].append(
                    {
                        "speaker": "participant",
                        "role": "user",
                        "content": user_message,
                        "thoughts": user_thoughts,
                    }
                )
                continue

            if turn == 0:
                user_message_history.append({"role": "user", "content": user_start_qs})
                user_message, user_thoughts = user.start_convo(user_start_qs)
            else:
                user_message, user_thoughts = user.get_response(user.messages)

            chatbot.update("user", user_message, user_thoughts)
            user.update("assistant", user_message, user_thoughts)
            user_message_history.append(
                {"role": "assistant", "content": user_message}
            )
            transcript["messages"].append(
                {
                    "speaker": "participant",
                    "role": "user",
                    "content": user_message,
                    "thoughts": user_thoughts,
                }
            )

            chatbot_response, chatbot_thoughts = chatbot.get_response(chatbot.messages)
            chatbot.update("assistant", chatbot_response, chatbot_thoughts)
            user.update("user", chatbot_response, chatbot_thoughts)
            user_message_history.append(
                {"role": "user", "content": chatbot_response}
            )
            transcript["messages"].append(
                {
                    "speaker": "chatbot",
                    "role": "assistant",
                    "content": chatbot_response,
                    "thoughts": chatbot_thoughts,
                }
            )

        transcript["user_messages"] = user_message_history
        transcript["chatbot_messages"] = copy.deepcopy(chatbot.messages)
        state["intervention_messages"].append(transcript)

        state["trace"].append(
            {
                "type": "intervention",
                "question_id": meta["question_id"],
                "key": meta["data_export_tag"],
                "source": "chatbot_config",
                "label": label,
                "model": model,
                "first_speaker": first_speaker,
                "requested_first_speaker": requested_first_speaker,
                "message_count": len(transcript["messages"]),
            }
        )
        state["pending_completion_code"] = override.get("completion_code", "SIMULATED_CHATBOT")

    def _apply_intervention_step(self, meta, user, state, intervention_overrides):
        override = self._resolve_intervention_override(meta, state, intervention_overrides)
        if override is None:
            raise MissingInterventionConfigError(
                "Intervention config required for "
                f"{meta['question_id']} ({meta['data_export_tag']})."
            )

        if isinstance(override, str):
            text = override
            completion_code = None
        else:
            if any(key in override for key in ("base_url", "model", "api_secret")) or any(
                key in override for key in ("collect_participant_opener", "participant_opener_only")
            ):
                self._simulate_chat_intervention(override, user, meta, state)
                return
            text = override.get("text", "")
            completion_code = override.get("completion_code")

        text = self._resolve_piped_text(text, state)
        text = self._personalize_intervention_text(text, user)
        if text:
            user.update("user", text, "")
            state["trace"].append(
                {
                    "type": "intervention",
                    "question_id": meta["question_id"],
                    "key": meta["data_export_tag"],
                    "source": "override",
                }
            )
        if completion_code is not None:
            state["pending_completion_code"] = completion_code

    def _apply_display_step(self, meta, user, state):
        text = self._resolve_piped_text(meta["text"], state)
        if text:
            user.update("user", text, "")
            state["trace"].append(
                {
                    "type": "display",
                    "question_id": meta["question_id"],
                    "key": meta["data_export_tag"],
                }
            )

    def _apply_text_entry_step(self, meta, state):
        value = state.pop("pending_completion_code", None)
        state["question_values"][meta["question_id"]] = value
        state["trace"].append(
            {
                "type": "text_entry",
                "question_id": meta["question_id"],
                "key": meta["data_export_tag"],
                "value": value,
            }
        )

    def _ask_scale(self, meta, user, state):
        scale_name = meta["scale_name"]
        scale = dict(self.scales[scale_name])
        scale["items"] = dict(self.scales[scale_name]["items"])
        scale["question"] = self._resolve_piped_text(scale["question"], state)
        assessment, thoughts, answer = self.get_answer(scale, user)
        user.update("assistant", answer, thoughts)
        score = self.get_score(scale, assessment)
        results_entry = {
            "assessment": assessment,
            "score": score,
            "user_memory": user.memory,
        }
        state["results"][scale_name] = results_entry
        state["question_values"][meta["question_id"]] = self._extract_choice_value(assessment)
        state["question_selected_labels"][meta["question_id"]] = (
            self._selected_choice_labels_for_assessment(meta, assessment)
        )
        state["trace"].append(
            {
                "type": "scale",
                "question_id": meta["question_id"],
                "scale": scale_name,
            }
        )
        if not self.preserve_context:
            user.clear()

    def _execute_block(
        self,
        block_id,
        user,
        state,
        allowed_scales,
        random_state,
        intervention_overrides,
    ):
        if block_id in self.excluded_block_ids:
            state["trace"].append(
                {
                    "type": "skipped_block",
                    "block_id": block_id,
                    "reason": "excluded_by_user",
                }
            )
            return

        block = self.flow_data["blocks"].get(block_id)
        if block is None:
            state["warnings"].append(f"Missing block metadata for {block_id}.")
            return

        question_ids = self._materialize_block_question_ids(block, random_state)
        state["trace"].append(
            {
                "type": "block",
                "block_id": block_id,
                "description": block.get("description", ""),
                "question_ids": question_ids,
            }
        )

        for qid in question_ids:
            meta = self.flow_data["questions"].get(qid)
            if meta is None:
                continue

            if qid in self.excluded_question_ids:
                state["trace"].append(
                    {
                        "type": "skipped_question",
                        "question_id": qid,
                        "key": meta.get("data_export_tag"),
                        "runtime_type": self._get_question_runtime_type(qid=qid, meta=meta),
                        "reason": "excluded_by_user",
                    }
                )
                continue

            runtime_type = self._get_question_runtime_type(qid=qid, meta=meta)
            if runtime_type == "display":
                self._apply_display_step(meta, user, state)
            elif runtime_type == "intervention":
                self._apply_intervention_step(meta, user, state, intervention_overrides)
            elif runtime_type == "text_entry":
                self._apply_text_entry_step(meta, state)
            elif runtime_type == "scale":
                if allowed_scales is None or meta["scale_name"] in allowed_scales:
                    self._ask_scale(meta, user, state)
                else:
                    state["trace"].append(
                        {
                            "type": "skipped_scale",
                            "question_id": qid,
                            "scale": meta["scale_name"],
                            "reason": "excluded_by_user"
                            if meta["scale_name"] in self.excluded_scale_names
                            else "not_requested",
                        }
                    )
            else:
                state["trace"].append(
                    {
                        "type": "skipped_question",
                        "question_id": qid,
                        "runtime_type": runtime_type,
                    }
                )

    def _execute_flow_nodes(
        self,
        nodes,
        user,
        state,
        allowed_scales,
        random_state,
        intervention_overrides,
        runtime_context=None,
    ):
        for node in nodes:
            if state["stopped"]:
                break

            node_type = node.get("Type")
            if node_type in {"Block", "Standard"}:
                self._execute_block(
                    node.get("ID"),
                    user,
                    state,
                    allowed_scales,
                    random_state,
                    intervention_overrides,
                )
            elif node_type == "EmbeddedData":
                label = state["group_stack"][-1] if state["group_stack"] else None
                for field in node.get("EmbeddedData", []):
                    value = self._resolve_embedded_value(field.get("Value"), random_state)
                    state["embedded_data"][field.get("Field")] = value
                    if label and field.get("Field"):
                        condition_override = state.get("condition_label_override")
                        if field.get("Field") == "Condition" and condition_override:
                            state["embedded_labels"][field["Field"]] = condition_override
                        else:
                            state["embedded_labels"][field["Field"]] = label
                state["trace"].append(
                    {
                        "type": "embedded_data",
                        "flow_id": node.get("FlowID"),
                        "fields": [
                            field.get("Field") for field in node.get("EmbeddedData", [])
                        ],
                    }
                )
            elif node_type == "Group":
                state["group_stack"].append(node.get("Description"))
                self._execute_flow_nodes(
                    node.get("Flow", []),
                    user,
                    state,
                    allowed_scales,
                    random_state,
                    intervention_overrides,
                    runtime_context=runtime_context,
                )
                state["group_stack"].pop()
            elif node_type == "BlockRandomizer":
                children = self._get_active_randomizer_children(node)
                if not children:
                    state["warnings"].append(
                        f"Randomizer {node.get('FlowID')} has no remaining children after exclusions."
                    )
                    state["trace"].append(
                        {
                            "type": "skipped_randomizer",
                            "flow_id": node.get("FlowID"),
                            "reason": "excluded_by_user",
                        }
                    )
                    continue
                subset = int(node.get("SubSet", len(children)))
                selected, desired_condition = self._select_condition_aligned_children(
                    children,
                    subset,
                    user,
                    state,
                    random_state,
                )
                selection_source = "provided_condition" if selected is not None else None
                if selected is None:
                    selected = self._resolve_planned_randomizer_selection(
                        node,
                        children,
                        runtime_context,
                    )
                    if selected is not None:
                        selection_source = "even_presentation_plan"
                if selected is None:
                    selected = self._choice_subset(children, subset, random_state)
                    selection_source = "randomized"
                state["trace"].append(
                    {
                        "type": "block_randomizer",
                        "flow_id": node.get("FlowID"),
                        "selection_source": selection_source,
                        "desired_condition": desired_condition,
                        "selected": [
                            child.get("Description") or child.get("ID") or child.get("FlowID")
                            for child in selected
                        ],
                    }
                )
                previous_condition_label_override = state.get("condition_label_override")
                if selected is not None and desired_condition and self._condition_word_tokens(desired_condition):
                    state["condition_label_override"] = desired_condition
                else:
                    state["condition_label_override"] = None
                self._execute_flow_nodes(
                    selected,
                    user,
                    state,
                    allowed_scales,
                    random_state,
                    intervention_overrides,
                    runtime_context=runtime_context,
                )
                state["condition_label_override"] = previous_condition_label_override
            elif node_type == "Branch":
                branch_result = self._evaluate_branch(node.get("BranchLogic", {}), state)
                state["trace"].append(
                    {
                        "type": "branch",
                        "flow_id": node.get("FlowID"),
                        "result": branch_result,
                    }
                )
                if branch_result is None:
                    state["warnings"].append(
                        f"Could not evaluate branch {node.get('FlowID')}; skipping."
                    )
                    continue
                if branch_result:
                    self._execute_flow_nodes(
                        node.get("Flow", []),
                        user,
                        state,
                        allowed_scales,
                        random_state,
                        intervention_overrides,
                        runtime_context=runtime_context,
                    )
            elif node_type == "EndSurvey":
                state["stopped"] = True
                state["trace"].append(
                    {
                        "type": "end_survey",
                        "flow_id": node.get("FlowID"),
                    }
                )
            else:
                state["warnings"].append(f"Unsupported flow node type: {node_type}.")

    def _administer_via_flow(
        self,
        user,
        scales=None,
        random_state=None,
        intervention_overrides=None,
        runtime_context=None,
    ):
        allowed_scales = set(self.get_scales(scales)) if scales is not None else None
        rng = self._coerce_rng(random_state)
        state = {
            "embedded_data": {},
            "embedded_labels": {},
            "intervention_messages": [],
            "question_values": {},
            "question_selected_labels": {},
            "results": {},
            "trace": [],
            "warnings": list(self.warnings),
            "group_stack": [],
            "pending_completion_code": None,
            "stopped": False,
            "condition_label_override": None,
        }

        self._execute_flow_nodes(
            self.flow_data.get("flow", {}).get("Flow", []),
            user,
            state,
            allowed_scales,
            rng,
            intervention_overrides or {},
            runtime_context=runtime_context or {},
        )

        state["results"]["__flow__"] = {
            "trace": state["trace"],
            "warnings": state["warnings"],
            "embedded_data": state["embedded_data"],
            "embedded_labels": state["embedded_labels"],
            "intervention_messages": state["intervention_messages"],
            "user_messages": copy.deepcopy(getattr(user, "messages", [])),
        }
        return state["results"]

    def administer_survey(
        self,
        user,
        scales=None,
        random_state=None,
        intervention_overrides=None,
        runtime_context=None,
    ):
        if self.flow_data is None:
            raise ValueError(
                "Survey administration now requires flow-backed survey metadata. "
                "Load a .qsf survey to execute the survey runtime."
            )
        return self._administer_via_flow(
            user=user,
            scales=scales,
            random_state=random_state,
            intervention_overrides=intervention_overrides,
            runtime_context=runtime_context,
        )
