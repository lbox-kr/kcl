from openai import OpenAI


class OAIModel:

    def __init__(
        self,
        model_name: str,
        thinking_budget: str = "None",
        system_prompt: str = None,
    ):
        self.__set_client()

        if not model_name:
            raise ValueError("Model name must be provided.")

        self.model_name = model_name

        self.reasoning = None
        if thinking_budget != "None":
            self.reasoning = {"effort": thinking_budget}

        self.system_prompt = (
            system_prompt
            if system_prompt
            else ("You are a helpful assistant.")
        )

        self.usage_history = []

    def __set_client(self):

        self.client = OpenAI()

    def generate(self, prompt: str):

        response = self.client.responses.create(
            model=self.model_name,
            input=[
                {"role": "developer", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            reasoning=self.reasoning,
        )

        return response.output_text


if __name__ == "__main__":

    my_model = OAIModel(
        model_name="o4-mini-2025-04-16", thinking_budget="medium"
    )
    output = my_model.generate("What is the capital of France?")
    print(output)
