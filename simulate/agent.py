import time

from .utils import BOT_TEMPERATURE, BOT_MAX_TOKENS


EMPTY_RESPONSE_MAX_RETRIES = 2
EMPTY_RESPONSE_RETRY_DELAY_SECONDS = 0.5

class BaseBot:
    def __init__(self, client, model, temperature, max_tokens):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.system_message = ""
        self.user_params = {}
        self.messages = []
        self.memory = []

    def _get_response(
        self, messages, temperature=None, max_tokens=None, json=False, model=None
    ):

        temp = temperature if temperature is not None else self.temperature
        max_toks = max_tokens if max_tokens is not None else self.max_tokens
        response_format = {"type": "json_object"} if json else None
        model = model if model else self.model

        request_kwargs = {
            "model": model,
            "messages": messages,
            "response_format": response_format,
        }
        if temp is not None:
            request_kwargs["temperature"] = float(temp)
        if max_toks is not None:
            request_kwargs["max_tokens"] = max_toks

        total_attempts = EMPTY_RESPONSE_MAX_RETRIES + 1
        for attempt in range(total_attempts):
            response = self.client.chat.completions.create(**request_kwargs)
            content = response.choices[0].message.content
            if not self._is_empty_response(content):
                return content

            if attempt < total_attempts - 1:
                print(
                    f"Empty response from model {model}; retrying "
                    f"({attempt + 1}/{EMPTY_RESPONSE_MAX_RETRIES})."
                )
                time.sleep(EMPTY_RESPONSE_RETRY_DELAY_SECONDS * (attempt + 1))

        raise ValueError(
            f"Model {model} returned an empty response after {total_attempts} attempts."
        )

    def get_response(self, messages, temperature=None, max_tokens=None, json=False, model=None, format=True):
        response = self._get_response(messages, temperature, max_tokens, json, model)
        if format:
            return self.format_response(response)
        return response
    
    def format_response(self, response):
        if self._is_empty_response(response):
            raise ValueError("Model returned an empty response.")
        if "</think>" in response:
            thoughts = response.split("</think>")[0]
            response = response.replace(thoughts, "").replace("</think>", "")
        else:
            thoughts = ""
        return response.strip(), thoughts.replace("<think>", "").strip()

    @staticmethod
    def _is_empty_response(response):
        return response is None or (isinstance(response, str) and not response.strip())
    
    def update_messages(self, role, message):
        self.messages.append({"role": role, "content": message})
    
    def update_thoughts(self, role, message, thoughts):
        self.memory.append({"role": role, "content": message, "thoughts": thoughts})
    
    def update(self, role, message, thoughts):
        self.update_messages(role, message)
        self.update_thoughts(role, message, thoughts)
    
    def update_memory(self, memory):
        for mem in memory:
            self.update(mem["role"], mem["content"], mem["thoughts"])
            
    def clear(self):
        if len(self.messages) > 0:
            self.messages = [self.messages[0]]
            self.memory = [self.memory[0]]
        else:
            self.messages = []
            self.memory = []

    def format_personalization(self, text):
        try:
            for key, value in self.user_params.items():
                text = text.replace(f"[{key.upper()}]", str(value))
        except Exception as e:
            print(text)
            print(f"Failed to format personalization: {e}")
        return text


class Chatbot(BaseBot):
    def __init__(self, client, user_params, model, temperature=BOT_TEMPERATURE, max_tokens=BOT_MAX_TOKENS):
        super().__init__(client, model, temperature, max_tokens)
        self.user_params = user_params

    def format_messages(self, personalization_text):
        self.system_message = self.format_personalization(personalization_text or "")
        if self.system_message:
            self.messages = [{"role": "system", "content": self.system_message}]
            self.memory = [{"role": "system", "content": self.system_message, "thoughts": ""}]
        else:
            self.messages = []
            self.memory = []


class User(BaseBot):
    def __init__(self, client, user_params, model, temperature=BOT_TEMPERATURE, max_tokens=BOT_MAX_TOKENS):
        super().__init__(client, model, temperature, max_tokens)
        self.user_params = user_params

    def format_messages(self, user_system_text, gen_user_info=False):
        user_system_text = self.format_personalization(user_system_text)

        if gen_user_info:
            user_info, _ = self.get_response(
                [
                    {
                        "role": "user",
                        "content": user_system_text
                        + "\n Write at least two sentences about yourself. You can write about your job, hobbies, living arrangements or any other information you think might be relevant.",
                    }
                ]
            )
            self.user_params["user_info"] = user_info

            user_system_text = (
                user_system_text
                + "\n\nThe user you are simulating has these circumstances: [USER_INFO]".replace(
                    "[USER_INFO]", self.user_params["user_info"]
                )
            )

        self.system_message = user_system_text
        self.messages.append({"role": "system", "content": self.system_message})
        self.memory = [{"role": "system", "content": self.system_message, "thoughts": ""}] 

    def start_convo(self, user_start_qs):
        self.update("user", user_start_qs, "")
        return self.get_response(self.messages)
