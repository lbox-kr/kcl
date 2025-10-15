try:
    from .claude import ClaudeModel

    _err_claude = None
except Exception as e:
    ClaudeModel = None
    _err_claude = e

try:
    from .gemini import GeminiModel

    _err_gemini = None
except Exception as e:
    GeminiModel = None
    _err_gemini = e

try:
    from .oai import OAIModel

    _err_oai = None
except Exception as e:
    OAIModel = None
    _err_oai = e


API_MODEL_REGISTRY = {
    "us.anthropic.claude-3-7-sonnet-20250219-v1:0": ClaudeModel,
    "us.anthropic.claude-sonnet-4-20250514-v1:0": ClaudeModel,
    "us.anthropic.claude-opus-4-20250514-v1:0": ClaudeModel,
    "gemini-2.5-flash": GeminiModel,
    "gemini-2.5-pro": GeminiModel,
    "gpt-4.1-2025-04-14": OAIModel,
    "o3-2025-04-16": OAIModel,
    "o4-mini-2025-04-16": OAIModel,
    "gpt-5-2025-08-07": OAIModel,
    "gpt-5-mini-2025-08-07": OAIModel,
}


def get_model(model_name, **kwargs):

    key = model_name.lower()

    if key not in API_MODEL_REGISTRY:
        raise ValueError(f"Unsupported model name: {model_name}")

    return API_MODEL_REGISTRY[key](model_name, **kwargs)


__all__ = ["get_model"]
