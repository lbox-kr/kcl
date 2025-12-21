import json
import os
from pathlib import Path

from google import genai
from google.genai import types
from google.oauth2 import service_account


class GeminiModel:

    def __init__(
        self,
        model_name: str,
        thinking_budget: int = -1,
    ):
        self.__set_client()

        if not model_name:
            raise ValueError("Model name must be provided.")

        self.model_name = model_name
        self.thinking_budget = thinking_budget

        self.usage_history = []

    def __set_client(self):
        self.credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", None)
        if not self.credentials:
            raise ValueError(
                "GOOGLE_APPLICATION_CREDENTIALS environment variable is not set."
            )
        credentials = Path(self.credentials)
        json_credential = json.loads(credentials.read_text())
        credentials = service_account.Credentials.from_service_account_file(
            credentials,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )

        self.client = genai.Client(
            vertexai=True,
            credentials=credentials,
            project=json_credential["project_id"],
            location="us-central1",
        )

    def generate(self, prompt: str, cache=None):
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_budget=self.thinking_budget
            ),
            safety_settings=[
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
            ],
        )
        if cache is not None:
            config.cached_content = cache.name

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[prompt],
            config=config,
        )

        return response.text

    def count_tokens(self, text: str) -> int | None:
        response = self.client.models.count_tokens(
            model=self.model_name, contents=[text]
        )
        return response.total_tokens


if __name__ == "__main__":

    my_gemini = GeminiModel(
        model_name="gemini-2.5-flash", thinking_budget=1024
    )

    response = my_gemini.generate("What is the capital of France?")
    print(response)
