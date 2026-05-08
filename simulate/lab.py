import hashlib
import inspect
import json
import re
import traceback
from math import isclose
import numpy as np
import pandas as pd

from .parallel import multithreaded
from .agent import User
from .personas import load_default_personas
from .utils import get_int, footrule_distance, abs_distance, reformat_prior_convo
from .utils import BOT_TEMPERATURE, BOT_MAX_TOKENS


def _validate_df(df):
    """Validate and normalise a participant DataFrame in-place."""
    assert 'id' in df.columns, "DataFrame must have an 'id' column with unique participant ids. Id must be str type."
    df["id"] = df["id"].astype(str)
    assert df['id'].is_unique, "DataFrame 'id' column must have unique participant ids."
    assert pd.api.types.is_string_dtype(df['id']), "DataFrame 'id' column must be of str type."
    df.index = list(df["id"])


class Lab:
    def __init__(self, client, df=None, user_model=None, **kwargs):
        self.seed = kwargs.get("seed", 42)
        self.random = np.random.default_rng(seed=self.seed)
        self.client = client

        if df is None:
            df = load_default_personas()

        _validate_df(df)
        self.df = df

        self.model = {
            "user": user_model,
        }
        self.temperature = {
            "user": kwargs.get("user_temperature", BOT_TEMPERATURE),
        }
        self.max_tokens = {
            "user": kwargs.get("user_max_tokens", BOT_MAX_TOKENS),
        }

    def _make_rng(self, *parts):
        raw = "|".join(str(part) for part in (self.seed, *parts))
        digest = hashlib.md5(raw.encode("utf-8")).hexdigest()
        return np.random.default_rng(int(digest[:8], 16))

    def _make_run_id(self, *parts):
        raw = "|".join(str(part) for part in (self.seed, *parts))
        digest = hashlib.md5(raw.encode("utf-8")).hexdigest()
        return f"run_{digest[:12]}"

    def _normalize_context_mode(self, context_mode):
        if context_mode is None:
            return "none"
        if context_mode not in {"none", "real"}:
            raise ValueError(
                f"Invalid context_mode {context_mode!r}. Expected 'none' or 'real'."
            )
        return context_mode

    def _iter_flow_nodes(self, nodes):
        for node in nodes or []:
            yield node
            yield from self._iter_flow_nodes(node.get("Flow", []))

    def _child_selection_token(self, child, index=0, parent_flow_id=None):
        return (
            child.get("FlowID")
            or child.get("ID")
            or f"{parent_flow_id or 'flow'}:{index}"
        )

    def _participant_has_condition_override(self, user_params):
        for key in (
            "condition",
            "Condition",
            "survey_embedded_label_Condition",
            "survey_embedded_Condition",
        ):
            value = user_params.get(key)
            if value is not None and str(value).strip():
                return True
        return False

    def _get_even_presentation_randomizer_nodes(self, survey):
        flow_data = getattr(survey, "flow_data", None) or {}
        root_nodes = flow_data.get("flow", {}).get("Flow", [])
        return [
            node
            for node in self._iter_flow_nodes(root_nodes)
            if node.get("Type") == "BlockRandomizer" and bool(node.get("EvenPresentation"))
            and list(self._active_randomizer_children(survey, node))
        ]

    def _flow_child_option_label(self, survey, child):
        if hasattr(survey, "_randomizer_child_option_label"):
            return survey._randomizer_child_option_label(child)
        return (
            child.get("Description")
            or survey.flow_data["blocks"].get(child.get("ID"), {}).get("description")
            or child.get("ID")
            or child.get("FlowID")
            or child.get("Type")
        )

    def _active_randomizer_children(self, survey, node):
        if hasattr(survey, "_get_active_randomizer_children"):
            return list(survey._get_active_randomizer_children(node))
        return list(node.get("Flow", []))

    def _normalize_even_presentation_counts(
        self,
        even_presentation_counts,
        survey,
        randomizers,
        planned_slots_by_flow,
    ):
        if even_presentation_counts is None:
            return {}

        if not randomizers:
            raise ValueError("No even-presentation randomizers found for count overrides.")

        randomizer_options = {}
        for node in randomizers:
            flow_id = node.get("FlowID")
            randomizer_options[flow_id] = [
                self._flow_child_option_label(survey, child)
                for child in self._active_randomizer_children(survey, node)
            ]

        if all(flow_id not in even_presentation_counts for flow_id in randomizer_options):
            if len(randomizer_options) != 1:
                raise ValueError(
                    "Ambiguous even_presentation_counts override. Pass a mapping from flow_id to counts when multiple even-presentation randomizers exist."
                )
            flow_id = next(iter(randomizer_options))
            raw_mapping = {flow_id: even_presentation_counts}
        else:
            raw_mapping = even_presentation_counts

        normalized = {}
        for flow_id, requested_counts in raw_mapping.items():
            if flow_id not in randomizer_options:
                raise KeyError(f"Unknown even-presentation randomizer: {flow_id}")
            if not isinstance(requested_counts, dict):
                raise TypeError(
                    f"Counts for {flow_id} must be a dict mapping option labels to requested counts."
                )

            options = randomizer_options[flow_id]
            counts = {}
            for option, count in requested_counts.items():
                if option not in options:
                    raise KeyError(
                        f"Unknown option {option!r} for even-presentation randomizer {flow_id}. "
                        f"Expected one of: {options}"
                    )
                if int(count) != count or int(count) < 0:
                    raise ValueError(
                        f"Requested count for {option!r} in {flow_id} must be a non-negative integer."
                    )
                counts[option] = int(count)

            counts = {option: counts.get(option, 0) for option in options}
            expected_total = planned_slots_by_flow.get(flow_id, 0) * int(
                next(
                    min(
                        int(node.get("SubSet", len(self._active_randomizer_children(survey, node)))),
                        len(self._active_randomizer_children(survey, node)),
                    )
                    for node in randomizers
                    if node.get("FlowID") == flow_id
                )
            )
            requested_total = sum(counts.values())
            if requested_total != expected_total:
                raise ValueError(
                    f"Requested counts for {flow_id} sum to {requested_total}, but this batch needs {expected_total} selections."
                )
            normalized[flow_id] = counts

        return normalized

    def _build_even_presentation_plans(self, sampled_users, user_system_texts, survey, context_mode, even_presentation_counts=None):
        randomizers = self._get_even_presentation_randomizer_nodes(survey)
        if not randomizers:
            return {}

        plans = {}
        user_system_keys = list(user_system_texts.keys())
        run_slots = [
            (user_id, user_system_key)
            for user_id in sampled_users
            for user_system_key in user_system_keys
        ]
        user_params_by_id = {
            user_id: self.generate_user_params_from_df(user_id)
            for user_id in sampled_users
        }
        planned_slots_by_flow = {}
        for node in randomizers:
            flow_id = node.get("FlowID")
            planned_slots_by_flow[flow_id] = sum(
                1
                for user_id, _user_system_key in run_slots
                if not self._participant_has_condition_override(user_params_by_id[user_id])
            )
        requested_counts_by_flow = self._normalize_even_presentation_counts(
            even_presentation_counts,
            survey,
            randomizers,
            planned_slots_by_flow,
        )

        for node in randomizers:
            children = self._active_randomizer_children(survey, node)
            if not children:
                continue

            subset = min(int(node.get("SubSet", len(children))), len(children))
            child_tokens = [
                self._child_selection_token(child, index=index, parent_flow_id=node.get("FlowID"))
                for index, child in enumerate(children)
            ]
            counts = {token: 0 for token in child_tokens}
            flow_id = node.get("FlowID")
            token_to_option = {
                token: self._flow_child_option_label(survey, child)
                for token, child in zip(child_tokens, children)
            }
            remaining_requested_counts = None
            if flow_id in requested_counts_by_flow:
                remaining_requested_counts = {
                    token: requested_counts_by_flow[flow_id][token_to_option[token]]
                    for token in child_tokens
                }

            for user_id, user_system_key in run_slots:
                if self._participant_has_condition_override(user_params_by_id[user_id]):
                    continue

                slot_rng = self._make_rng(
                    user_id,
                    user_system_key,
                    context_mode,
                    flow_id,
                    "even_presentation",
                )
                ranked_tokens = child_tokens[:]
                slot_rng.shuffle(ranked_tokens)
                if remaining_requested_counts is not None:
                    ranked_tokens.sort(key=lambda token: remaining_requested_counts[token], reverse=True)
                    selected = [token for token in ranked_tokens if remaining_requested_counts[token] > 0][:subset]
                    if len(selected) != subset:
                        raise ValueError(
                            f"Requested even-presentation counts for {flow_id} cannot satisfy subset={subset} at every planned slot."
                        )
                else:
                    ranked_tokens.sort(key=lambda token: counts[token])
                    selected = ranked_tokens[:subset]

                plans.setdefault((user_id, user_system_key), {})[flow_id] = selected
                for token in selected:
                    counts[token] += 1
                    if remaining_requested_counts is not None:
                        remaining_requested_counts[token] -= 1

            if remaining_requested_counts is not None:
                leftovers = {token_to_option[token]: value for token, value in remaining_requested_counts.items() if value != 0}
                if leftovers:
                    raise ValueError(
                        f"Requested even-presentation counts for {flow_id} were not fully allocated: {leftovers}"
                    )

        return plans

    def describe_even_presentation_plan(self, sampled_users, user_system_texts, survey, context_mode="none", even_presentation_counts=None):
        if getattr(survey, "flow_data", None) is None:
            raise ValueError("Flow-backed survey required to inspect even-presentation randomizers.")

        context_mode = self._normalize_context_mode(context_mode)
        flow_randomizers = {
            entry["flow_id"]: entry
            for entry in survey.describe_flow_randomizers()
            if entry.get("even_presentation")
        }
        if not flow_randomizers:
            return []

        randomizer_nodes = self._get_even_presentation_randomizer_nodes(survey)
        run_slots = [
            (user_id, user_system_key)
            for user_id in sampled_users
            for user_system_key in user_system_texts.keys()
        ]
        user_params_by_id = {
            user_id: self.generate_user_params_from_df(user_id)
            for user_id in sampled_users
        }
        planned_slots_by_flow = {
            node.get("FlowID"): sum(
                1
                for user_id, _user_system_key in run_slots
                if not self._participant_has_condition_override(user_params_by_id[user_id])
            )
            for node in randomizer_nodes
        }
        requested_counts_by_flow = self._normalize_even_presentation_counts(
            even_presentation_counts,
            survey,
            randomizer_nodes,
            planned_slots_by_flow,
        )
        plans = self._build_even_presentation_plans(
            sampled_users,
            user_system_texts,
            survey,
            context_mode,
            even_presentation_counts=even_presentation_counts,
        )

        described = []
        for flow_id, randomizer in flow_randomizers.items():
            planned_counts = {option: 0 for option in randomizer["options"]}
            token_to_option = {}
            node = next(
                node
                for node in self._iter_flow_nodes(survey.flow_data.get("flow", {}).get("Flow", []))
                if node.get("Type") == "BlockRandomizer" and node.get("FlowID") == flow_id
            )
            for index, child in enumerate(self._active_randomizer_children(survey, node)):
                token = self._child_selection_token(child, index=index, parent_flow_id=flow_id)
                option = (
                    self._flow_child_option_label(survey, child)
                )
                token_to_option[token] = option

            planned_run_slots = 0
            replayed_condition_slots = 0
            for run_slot in run_slots:
                selected_tokens = plans.get(run_slot, {}).get(flow_id)
                if selected_tokens is None:
                    replayed_condition_slots += 1
                    continue
                planned_run_slots += 1
                for token in selected_tokens:
                    option = token_to_option.get(token)
                    if option is not None:
                        planned_counts[option] += 1

            described.append(
                {
                    "flow_id": flow_id,
                    "path": randomizer["path"],
                    "subset": randomizer["subset"],
                    "option_count": randomizer["option_count"],
                    "options": randomizer["options"],
                    "planned_counts": planned_counts,
                    "requested_counts": requested_counts_by_flow.get(flow_id),
                    "planned_run_slots": planned_run_slots,
                    "replayed_condition_slots": replayed_condition_slots,
                    "user_system_count": len(user_system_texts),
                }
            )

        return described

    def sample(self, n_participants, offset=0, df=None):
        if df is None:
            df = self.df
        else:
            _validate_df(df)

        index = list(df.index.copy())
        self.random.shuffle(index)
        index = [str(i) for i in index]

        return index[offset:offset+n_participants]

    def _normalize_quota_targets(self, n_participants, quotas, df):
        if not quotas:
            raise ValueError("quota_sample requires a non-empty quotas mapping.")

        normalized = {}
        for column, requested in quotas.items():
            if column not in df.columns:
                raise KeyError(f"Unknown quota column: {column}")
            if not isinstance(requested, dict) or not requested:
                raise TypeError(f"Quota spec for {column} must be a non-empty dict.")

            available_values = set(df[column].dropna().astype(str))
            raw_values = {}
            for value, target in requested.items():
                if value not in available_values:
                    raise KeyError(
                        f"Unknown quota value {value!r} for column {column!r}. "
                        f"Available values include: {sorted(available_values)[:20]}"
                    )
                raw_values[str(value)] = target

            numeric_values = list(raw_values.values())
            if all(isinstance(value, (int, np.integer)) for value in numeric_values):
                counts = {value: int(target) for value, target in raw_values.items()}
                total = sum(counts.values())
                if total != n_participants:
                    raise ValueError(
                        f"Quota counts for column {column!r} sum to {total}, expected {n_participants}."
                    )
            else:
                shares = {}
                for value, target in raw_values.items():
                    try:
                        shares[value] = float(target)
                    except Exception as exc:
                        raise TypeError(
                            f"Quota target for {column!r}={value!r} must be numeric."
                        ) from exc
                total_share = sum(shares.values())
                if not isclose(total_share, 1.0, rel_tol=1e-9, abs_tol=1e-9):
                    raise ValueError(
                        f"Quota shares for column {column!r} must sum to 1.0, got {total_share}."
                    )
                exact = {value: share * n_participants for value, share in shares.items()}
                counts = {value: int(np.floor(amount)) for value, amount in exact.items()}
                remainder = n_participants - sum(counts.values())
                ranking = sorted(
                    exact.keys(),
                    key=lambda value: (exact[value] - counts[value], str(value)),
                    reverse=True,
                )
                for value in ranking[:remainder]:
                    counts[value] += 1

            normalized[column] = counts

        return normalized

    def _build_quota_strata(self, df, quota_columns):
        strata = {}
        for row in df[["id", *quota_columns]].itertuples(index=False):
            row_dict = row._asdict()
            participant_id = str(row_dict.pop("id"))
            stratum = tuple((column, str(row_dict[column])) for column in quota_columns)
            strata.setdefault(stratum, []).append(participant_id)
        return strata

    def _allocate_quota_counts(self, strata, quota_targets, n_participants):
        remaining = {
            column: dict(targets)
            for column, targets in quota_targets.items()
        }
        allocations = {stratum: 0 for stratum in strata}
        capacities = {stratum: len(ids) for stratum, ids in strata.items()}

        def stratum_score(stratum):
            return sum(
                remaining[column].get(value, 0)
                for column, value in stratum
            )

        for _ in range(n_participants):
            eligible = [
                stratum
                for stratum, capacity in capacities.items()
                if capacity > 0
                and all(remaining[column].get(value, 0) > 0 for column, value in stratum)
            ]
            if not eligible:
                unmet = {
                    column: {value: count for value, count in targets.items() if count > 0}
                    for column, targets in remaining.items()
                }
                unmet = {column: targets for column, targets in unmet.items() if targets}
                raise ValueError(
                    "Could not satisfy requested quotas with the available participant pool. "
                    f"Remaining unmet quotas: {unmet}"
                )

            self.random.shuffle(eligible)
            eligible.sort(key=lambda stratum: (stratum_score(stratum), capacities[stratum]), reverse=True)
            chosen = eligible[0]
            allocations[chosen] += 1
            capacities[chosen] -= 1
            for column, value in chosen:
                remaining[column][value] -= 1

        return {stratum: count for stratum, count in allocations.items() if count > 0}

    def quota_sample(self, n_participants, quotas, df=None):
        if df is None:
            df = self.df
        else:
            _validate_df(df)

        quota_targets = self._normalize_quota_targets(n_participants, quotas, df)
        quota_columns = list(quota_targets.keys())
        eligible_mask = pd.Series(True, index=df.index)
        for column, targets in quota_targets.items():
            eligible_mask &= df[column].astype(str).isin(targets.keys())
        eligible_df = df.loc[eligible_mask].copy()
        if len(eligible_df) < n_participants:
            raise ValueError(
                f"Only {len(eligible_df)} participants match the requested quota categories, "
                f"but {n_participants} were requested."
            )

        strata = self._build_quota_strata(eligible_df, quota_columns)
        stratum_allocations = self._allocate_quota_counts(strata, quota_targets, n_participants)

        selected = []
        for stratum, count in stratum_allocations.items():
            ids = list(strata[stratum])
            self.random.shuffle(ids)
            selected.extend(ids[:count])

        self.random.shuffle(selected)
        return selected

    def generate_user_params_from_df(self, id):
        row = self.df.loc[id]
        user_params = row.to_dict()
        return user_params

    def update_user_result(self, user_result, res, survey, scales):
        for scale in scales:
            scale_result = res.get(scale)
            include_score = "answers" in survey.scales[scale]
            if scale_result is None:
                user_result.update({key: None for key in survey.scales[scale]["items"].keys()})
                if include_score:
                    user_result[f"{scale}_score"] = None
                continue

            if scale_result["assessment"] is not None:
                scale_type = survey.scales[scale]["type"]
                user_result.update(
                    {
                        key: (value if scale_type == "free text" else get_int(value))
                        for key, value in scale_result["assessment"].items()
                    }
                )
            else:
                user_result.update({key: None for key in survey.scales[scale]["items"].keys()})

            if include_score:
                user_result[f"{scale}_score"] = scale_result["score"]

        flow_meta = res.get("__flow__", {})
        if flow_meta:
            intervention_messages = flow_meta.get("intervention_messages")
            flow_fields = {
                "survey_flow_trace": flow_meta.get("trace"),
                "survey_flow_warnings": flow_meta.get("warnings"),
                "survey_embedded_data": flow_meta.get("embedded_data"),
                "survey_embedded_labels": flow_meta.get("embedded_labels"),
                "survey_intervention_messages": intervention_messages,
                "survey_user_messages": flow_meta.get("user_messages"),
            }
            user_result.update(flow_fields)
            embedded_data = flow_meta.get("embedded_data") or {}
            embedded_labels = flow_meta.get("embedded_labels") or {}
            for key, value in embedded_data.items():
                user_result[f"survey_embedded_{self._normalize_flow_key(key)}"] = value
            for key, value in embedded_labels.items():
                user_result[f"survey_embedded_label_{self._normalize_flow_key(key)}"] = value
            chatbot_configs = self._extract_intervention_chatbot_configs(intervention_messages)
            if chatbot_configs:
                user_result["survey_chatbot_configs"] = chatbot_configs
                if len(chatbot_configs) == 1:
                    config = chatbot_configs[0]
                    user_result["survey_chatbot_model"] = config.get("model")
                    user_result["survey_chatbot_temperature"] = config.get("temperature")
                    user_result["survey_chatbot_max_tokens"] = config.get("max_tokens")
                    user_result["survey_chatbot_first_speaker"] = config.get("first_speaker")
                    user_result["survey_chatbot_prompt_source"] = config.get("prompt_source")
                    user_result["survey_chatbot_prompt_path"] = config.get("prompt_path")
        return user_result

    def _normalize_flow_key(self, key):
        return re.sub(r"[^a-zA-Z0-9_]+", "_", str(key)).strip("_")

    def _extract_intervention_chatbot_configs(self, intervention_messages):
        if not intervention_messages:
            return []
        configs = []
        for message in intervention_messages:
            if message.get("source") != "chatbot_config":
                continue
            configs.append(
                {
                    "question_id": message.get("question_id"),
                    "key": message.get("key"),
                    "label": message.get("label"),
                    "model": message.get("model"),
                    "temperature": message.get("temperature"),
                    "max_tokens": message.get("max_tokens"),
                    "first_speaker": message.get("first_speaker"),
                    "prompt_source": message.get("prompt_source"),
                    "prompt_path": message.get("prompt_path"),
                }
            )
        return configs

    def _administer_survey_with_runtime_context(
        self,
        survey,
        user,
        scales,
        flow_rng,
        intervention_overrides,
        runtime_context,
    ):
        administer = survey.administer_survey
        parameters = inspect.signature(administer).parameters
        kwargs = {
            "user": user,
            "scales": scales,
            "random_state": flow_rng,
            "intervention_overrides": intervention_overrides,
        }
        if "runtime_context" in parameters:
            kwargs["runtime_context"] = runtime_context
        return administer(**kwargs)

    @multithreaded()
    def simulate_survey_run(self, user_id, user_system_texts, survey, scales, context_mode="none", intervention_overrides=None, even_presentation_plans=None):
        flow_driven_runtime = getattr(survey, "flow_data", None) is not None
        if not flow_driven_runtime:
            raise ValueError("Lab now supports only flow-backed surveys. Load a .qsf survey so execution follows survey flow.")
        context_mode = self._normalize_context_mode(context_mode)

        user_params = self.generate_user_params_from_df(user_id)
        results = []

        for user_system_key, user_system_text in user_system_texts.items():
            try:
                run_id = self._make_run_id(user_id, user_system_key, context_mode)
                flow_rng = self._make_rng(user_id, user_system_key, context_mode, "flow")

                user = User(self.client, user_params, self.model["user"], self.temperature["user"], self.max_tokens["user"])
                user.format_messages(user_system_text)

                user_result = {
                    "run_id": run_id,
                    "id": user_id,
                    "user_system": user_system_key,
                    "context_mode": context_mode,
                    "user_model": self.model["user"],
                    "user_temperature": self.temperature["user"],
                    "user_max_tokens": self.max_tokens["user"]
                }

                if context_mode == "real":
                    assert "messages" in user_params.keys(), "User parameters must include 'messages' key with prior user messages."
                    prior_conversation = reformat_prior_convo(user_params["messages"])
                    user.update_memory(prior_conversation)

                res = self._administer_survey_with_runtime_context(
                    survey=survey,
                    user=user,
                    scales=scales,
                    flow_rng=flow_rng,
                    intervention_overrides=intervention_overrides,
                    runtime_context={
                        "even_presentation_plan": (
                            (even_presentation_plans or {}).get((user_id, user_system_key), {})
                        )
                    },
                )
                user_result = self.update_user_result(user_result, res, survey, scales)

                results.append(user_result)

            except Exception as e:
                results.append({
                    "run_id": self._make_run_id(user_id, user_system_key, context_mode),
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "traceback": traceback.format_exc(),
                    "id": user_id,
                    "user_system": user_system_key,
                    "context_mode": context_mode,
                    "user_model": self.model["user"],
                    "user_temperature": self.temperature["user"],
                    "user_max_tokens": self.max_tokens["user"],
                })

        return results

    def run_survey(self, sampled_users, user_system_texts, survey, scales=None, context_mode="none", intervention_overrides=None, even_presentation_counts=None, progress_callback=None):
        """Runs simulation for multiple users."""
        if getattr(survey, "flow_data", None) is None:
            raise ValueError("Lab now supports only flow-backed surveys. Load a .qsf survey so execution follows survey flow.")
        scales = survey.get_scales(scales)
        context_mode = self._normalize_context_mode(context_mode)
        even_presentation_plans = self._build_even_presentation_plans(
            sampled_users,
            user_system_texts,
            survey,
            context_mode,
            even_presentation_counts=even_presentation_counts,
        )
        results = self.simulate_survey_run(sampled_users, non_iterables={"user_system_texts": user_system_texts, "survey": survey, "scales": scales, "context_mode": context_mode, "intervention_overrides": intervention_overrides, "even_presentation_plans": even_presentation_plans, "_progress_callback": progress_callback})
        return results

    def format_results(self, sim_results, survey, scales=None):
        """Formats simulation results by merging with the original dataframe and calculating errors."""
        errors = [result for result in sim_results if 'error' in result.keys()]
        if len(errors) > 0:
            print(f"Removing {len(errors)} rows with an error.")
            sim_results = [result for result in sim_results if 'error' not in result.keys()]

        if not sim_results:
            base_cols = ["run_id", "id", "user_system"]
            empty = pd.DataFrame(columns=base_cols)
            if "id" in self.df.columns:
                return empty
            return empty

        temp_df = pd.DataFrame(sim_results)
        scale_item_cols = {
            item_key
            for scale_name in survey.get_scales(scales)
            for item_key in survey.scales[scale_name]["items"].keys()
            if item_key in self.df.columns
        }
        overlapping_cols = [col for col in temp_df.columns if col != "id" and col in self.df.columns]
        human_rename_cols = {col: f"{col}_human" for col in (scale_item_cols.union(overlapping_cols))}
        source_df = self.df
        if human_rename_cols:
            source_df = self.df.rename(columns=human_rename_cols)
        temp_df = temp_df.merge(source_df, on="id")

        if self._has_human_scale_data(temp_df, survey, scales):
            temp_df = self.calculate_simulation_error(temp_df, survey, scales)

        return temp_df

    def save_results(self, results, path):
        """Save raw simulation results to JSON."""
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=4)
        print("Saved to", path)

    def load_results(self, path):
        """Load raw simulation results from JSON."""
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _has_human_scale_data(self, temp_df, survey, scales=None):
        """Return True when the merged dataframe contains baseline human answers for the selected scales."""
        scales = survey.get_scales(scales)
        for scale_name in scales:
            if self._resolve_scale_data_pair(temp_df, survey.scales[scale_name]) is not None:
                return True
        return False

    def _resolve_scale_data_pair(self, temp_df, scale):
        item_keys = list(scale["items"].keys())

        if not all(key in temp_df.columns for key in item_keys):
            return None
        simulated_keys = {key: key for key in item_keys}

        if not all(f"{key}_human" in temp_df.columns for key in item_keys):
            return None
        human_keys = {key: f"{key}_human" for key in item_keys}

        return human_keys, simulated_keys

    def calculate_simulation_error(self, temp_df, survey, scales=None):
        """Calculates error for the simulation results."""
        scales = survey.get_scales(scales)

        print("Calculating simulation errors...")
        for scale_name in scales:
            scale = survey.scales[scale_name]
            column_pair = self._resolve_scale_data_pair(temp_df, scale)
            if column_pair is None:
                continue
            human_keys, simulated_keys = column_pair
            if scale["type"] in ("multiple choice", "numeric"):
                temp_df[scale["name"] + "_sim_error"] = temp_df.apply(
                    lambda row: abs_distance(
                        {key: row[human_keys[key]] for key in scale["items"].keys()},
                        {key: row[simulated_keys[key]] for key in scale["items"].keys()},
                    ),
                    axis=1,
                )
            elif scale["type"] == "ranking":
                temp_df[scale["name"] + "_sim_error"] = temp_df.apply(
                    lambda row: footrule_distance(
                        {key: row[human_keys[key]] for key in scale["items"].keys()},
                        {key: row[simulated_keys[key]] for key in scale["items"].keys()},
                    ),
                    axis=1,
                )
            elif scale["type"] == "free text":
                continue
            else:
                raise ValueError(f"Invalid scale type for scale {scale['name']!r}: {scale['type']!r}")
        return temp_df

    def average_simulation_error(self, results_df, survey, scales=None):
        """Return the mean simulation error over the selected scales."""
        scales = survey.get_scales(scales)
        error_cols = [f"{scale}_sim_error" for scale in scales if f"{scale}_sim_error" in results_df.columns]
        if not error_cols:
            raise ValueError("No simulation error columns found for the requested scales.")
        return float(results_df[error_cols].mean(axis=1, skipna=True).mean(skipna=True))
