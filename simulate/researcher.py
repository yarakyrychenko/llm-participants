import os
from .agent import User, BaseBot
from .parallel import multithreaded
from .utils import RESEARCH_TEMPERATURE, RESEARCH_MAX_TOKENS, RESEARCH_MAX_WORKERS, MAX_WORKERS
from .utils import wilcoxon_test, plot_error_hists
import json


class Researcher:
    def __init__(self, client, model, lab, survey, temperature=RESEARCH_TEMPERATURE, max_tokens=RESEARCH_MAX_TOKENS):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.author = BaseBot(self.client, 
                              model=model, 
                              temperature=temperature,
                              max_tokens=max_tokens)

        prompts_path = os.path.join(
            os.path.dirname(__file__),
            "data",
            "prompts",
            "researcher",
            "default.json",
        )
        with open(prompts_path, "r", encoding="utf-8") as f:
            self.metaprompts = {prompt["name"]: prompt["content"] for prompt in json.load(f)}
        
        self.lab = lab 
        self.survey = survey

    @multithreaded(max_workers=RESEARCH_MAX_WORKERS)
    def generate_content(self, messages):
        content, thoughts =  self.author.get_response(
            messages=messages,
            format=True
            )
        return [{"content": content, "thoughts": thoughts}]

    @multithreaded(max_workers=MAX_WORKERS)
    def _evaluate_dialogue(self, user_id, system, schema, user_prompt = None, convo_column="conversation"):
        user_params = self.lab.generate_user_params_from_df(user_id)
        user = User(self.client, user_params, 
                    self.lab.model["user"], 
                    self.lab.temperature["user"], 
                    self.lab.max_tokens["user"])
        
        assert convo_column in user_params, "User params must include a conversation history."

        if user_prompt is None:
            prompt = self.metaprompts.get(
                "dialogue_evaluation_user_prompt",
                (
                    "Evaluate the following participant conversation against the provided schema.\n\n"
                    "Schema:\n{schema_json}\n\nConversation:\n{conversation_json}"
                ),
            )
        else:
            prompt = user.format_personalization(user_prompt)
        
        prompt = prompt.format(
            schema_json=json.dumps(schema, ensure_ascii=False, indent=2),
            conversation_json=json.dumps(user_params[convo_column], ensure_ascii=False)
        )

        user_result = {
            "id": user_id, 
            "prompt": prompt, 
            "evaluation_model": self.model,
            }
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ]
        user_result["eval_messages"] = messages
        content = self.author.get_response(messages=messages, json=True, format=False)
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start >= 0 and end > start:
                parsed = json.loads(content[start:end+1])
            else:
                print("Failed to parse JSON response.")
                parsed = {}

        user_result["parsed"] = parsed

        return [user_result]
    
    def evaluate_dialogue(self, sampled_users, system, schema, user_prompt=None, convo_column="conversation"):
        results = self._evaluate_dialogue(sampled_users, non_iterables={"system": system, "schema": schema, "user_prompt": user_prompt, "convo_column": convo_column}) 
        return results

    def format_bot_system_texts(self, bot_system_texts, metaprompt_name):
        metaprompt = self.metaprompts.get(metaprompt_name, "").strip()
        formatted = {}
        for key, text in bot_system_texts.items():
            if metaprompt and text:
                formatted[key] = f"{text.rstrip()}\n\n{metaprompt}"
            elif metaprompt:
                formatted[key] = metaprompt
            else:
                formatted[key] = text
        return formatted

    def run_survey(self, sampled_users, user_system_texts, scales=None, context_mode="none"):
        formatted_user_system_texts = self.format_bot_system_texts(
            user_system_texts,
            "user_system_text_formatting",
        )
        return self.lab.run_survey(
            sampled_users,
            formatted_user_system_texts,
            self.survey,
            scales=scales,
            context_mode=context_mode,
        )

    def format_results(self, results, scales=None):
        return self.lab.format_results(results, self.survey, scales=scales)

    def format_messages(self, system, prompt):
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

    def get_user_answers(self, row, survey, scale_name):
        scale = survey.scales[scale_name]
        answers = {}
        for item_key, item_label in scale["items"].items():
            if item_key in row:
                answers[item_label] = row[item_key]
        return answers

    def _get_participant_examples(self, sampled_users, max_examples=3):
        examples = []
        for user_id in list(sampled_users)[:max_examples]:
            user_params = self.lab.generate_user_params_from_df(user_id)
            examples.append(
                {
                    key: value
                    for key, value in user_params.items()
                    if key not in {"messages", "conversation"}
                }
            )
        return json.dumps(examples, ensure_ascii=False, indent=2)

    def suggest_initial_prompts(self, n_prompts, sampled_users, prompt_objective, base_user_prompt):
        system = self.metaprompts["initial_prompt_suggestion_system"]
        prompt = (
            f"<prompt_objective>\n{prompt_objective}\n</prompt_objective>\n\n"
            f"<prompt_template>\n{base_user_prompt}\n</prompt_template>\n\n"
            f"<participant_examples>\n{self._get_participant_examples(sampled_users)}\n</participant_examples>\n\n"
            f"{self.metaprompts.get('prompt_formatting', '').strip()}"
        ).strip()
        messages = [self.format_messages(system, prompt) for _ in range(n_prompts)]
        generated = self.generate_content(messages)

        prompts = {"baseline": base_user_prompt}
        for idx, result in enumerate(generated, start=1):
            prompts[f"candidate_{idx}"] = result["content"]
        return prompts

    def _build_review_prompt(self, row, scale):
        conversation = row.get("conversation")
        if conversation is None and "id" in row:
            user_params = self.lab.generate_user_params_from_df(row["id"])
            conversation = user_params.get("conversation", user_params.get("messages", []))

        prompt = (
            f"<conversation>\n{json.dumps(conversation or [], ensure_ascii=False, indent=2)}\n</conversation>\n\n"
            f"<correct_participant_answers>\n"
            f"{json.dumps(self.get_user_answers(row, self.survey, scale), ensure_ascii=False, indent=2)}\n"
            f"</correct_participant_answers>"
        )
        return prompt

    def get_reviews(self, sampled_users, scale, results_df):
        review_rows = results_df[results_df["id"].astype(str).isin([str(user_id) for user_id in sampled_users])]
        messages = [
            self.format_messages(
                self.metaprompts["individual_review_system"],
                self._build_review_prompt(row, scale),
            )
            for _, row in review_rows.iterrows()
        ]
        generated = self.generate_content(messages)
        reviews = []
        for row, result in zip(review_rows.to_dict("records"), generated):
            reviews.append(
                {
                    "id": row["id"],
                    "scale": scale,
                    "review": result["content"],
                    "messages": self.format_messages(
                        self.metaprompts["individual_review_system"],
                        self._build_review_prompt(row, scale),
                    ),
                }
            )
        return reviews

    def generate_reviews(self, sampled_users, scale, results_df):
        return self.get_reviews(sampled_users, scale, results_df)

    def revise(self, n_prompts, reviews, prompt_objective, current_system_message):
        suggestions = "\n".join(review["review"] for review in reviews)
        system = self.metaprompts["revise_prompt_suggestion_system"]
        prompt = (
            f"<prompt_objective>\n{prompt_objective}\n</prompt_objective>\n\n"
            f"<draft_prompt>\n{current_system_message}\n</draft_prompt>\n\n"
            f"<suggestions>\n{suggestions}\n</suggestions>\n\n"
            f"{self.metaprompts.get('prompt_formatting', '').strip()}"
        ).strip()
        messages = [self.format_messages(system, prompt) for _ in range(n_prompts)]
        generated = self.generate_content(messages)

        prompts = {"baseline": current_system_message}
        for idx, result in enumerate(generated, start=1):
            prompts[f"revision_{idx}"] = result["content"]
        return prompts

    def test_prompts(self, results, scales=None, baseline="baseline", chatbot=False):
        system = "survey_chatbot_model" if chatbot else "user_system"
        temp_df = self.format_results(results, scales=scales)
        print(len(temp_df))
        scales = self.survey.get_scales(scales)
 
        for scale in scales:
            print(f"Basic stats for {scale}_sim_error:")
            print(temp_df.groupby(system)[f"{scale}_sim_error"].describe())
            print()
            print(f"Basic stats for {scale}_score:")
            print(temp_df.groupby(system)[f"{scale}_score"].describe())
            print()

            for prompt, _ in temp_df.groupby(system):
                if prompt == baseline:
                    continue

                print("Testing", prompt, "vs", baseline, "on", scale+"_sim_error...")
                print()
                test_df = temp_df.dropna(subset=[scale+"_sim_error"])
                common_ids = set(test_df[test_df[system] == prompt].id).intersection(
                    set(test_df[test_df[system] == baseline].id)
                )
                
                test_df = test_df[test_df.id.isin(common_ids)]
                wilcoxon_test(test_df[test_df[system]==prompt][scale+"_sim_error"], 
                    test_df[test_df[system]==baseline][scale+"_sim_error"])

                if not chatbot:
                    print(f"Human error vs {baseline} error:")
                    wilcoxon_test(test_df[test_df["user_system"]==baseline][scale], 
                        test_df[test_df["user_system"]==baseline][scale+"_score"])

                    print(f"Human error vs {prompt} error:")
                    wilcoxon_test(test_df[test_df["user_system"]==prompt][scale], 
                        test_df[test_df["user_system"]==prompt][scale+"_score"])
                
    def plot_prompt_performance(self, results, scales=None):
        temp_df = self.format_results(results, scales=scales)
        scales = self.survey.get_scales(scales)

        for scale in scales:
            plot_error_hists(temp_df,scale)
