import requests
from loguru import logger


class LocalModel:

    def __init__(
        self,
        model_name: str,
        port: int = 8000,
    ):
        self.model_name = model_name
        self.port = port
        self.__set_client()

        if not model_name:
            raise ValueError("Model name must be provided.")

        self.max_tokens = 16_384

    def __set_client(self):
        self.url = f"http://localhost:{self.port}/v1/chat/completions"

    def generate(self, prompt: str):

        message = [
            {"role": "user", "content": prompt},
        ]

        headers = {"Content-Type": "application/json"}

        payload = {
            "model": self.model_name,
            "messages": message,
            "stream": False,
            "top_p": 0.95,
            "temperature": 0.6,
        }

        try:
            response = requests.post(self.url, headers=headers, json=payload)
            response = response.json()

            response = response["choices"][0]["message"]["content"].strip()

        except Exception as e:
            logger.error(f"Error in generating response: {e}")
            response = ""

        return response


if __name__ == "__main__":

    model_kwargs = {"system_prompt": "You are a good boy"}

    my_model = LocalModel(model_name="google/gemma-3-27b-it", **model_kwargs)
    response = my_model.generate("What is the capital of France?")
    print(response)
