import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


class ClaudeModel:

    def __init__(
        self,
        model_name: str,
        thinking_budget: int = 8192,
    ):
        self.__set_client()

        if not model_name:
            raise ValueError("Model name must be provided.")

        self.model_name = model_name
        self.thinking_budget = thinking_budget

        self.usage_history = []

    def __set_client(self):

        self.client = boto3.client(
            "bedrock-runtime",
            region_name="us-east-2",
            config=Config(
                read_timeout=3600,
                connect_timeout=900,
            ),
        )

    def generate(self, prompt: str):

        conversation = [
            {
                "role": "user",
                "content": [{"text": prompt}],
            }
        ]

        reasoning_config = None
        if self.thinking_budget > 0:
            reasoning_config = {
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": self.thinking_budget,
                }
            }
        if reasoning_config is None:
            response = self.client.converse(
                modelId=self.model_name,
                messages=conversation,
            )

        else:
            response = self.client.converse(
                modelId=self.model_name,
                messages=conversation,
                inferenceConfig={"maxTokens": 128_000},
                additionalModelRequestFields=reasoning_config,
            )

        if self.thinking_budget:
            response_text = response["output"]["message"]["content"][1]["text"]
        else:
            response_text = response["output"]["message"]["content"][0]["text"]

        return response_text


if __name__ == "__main__":

    model = ClaudeModel(
        model_name="us.anthropic.claude-sonnet-4-20250514-v1:0"
    )
    try:
        response = model.generate("What is the capital of France?")
        print(response)
    except ClientError as e:
        print(f"An error occurred: {e}")
    except ValueError as ve:
        print(f"Value error: {ve}")
