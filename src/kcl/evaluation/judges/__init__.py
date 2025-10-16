from .kcl_essay_judge import KCLEssayEval
from .kcl_mcqa_judge import KCLMCQAEval

_TASK_REGISTRY = {
    "kcl_essay": KCLEssayEval,
    "kcl_mcqa": KCLMCQAEval,
}


def get_judge(name, **kwargs):
    key = name.lower()
    try:
        cls = _TASK_REGISTRY[key]
    except KeyError:
        raise ValueError(
            f"Unknown task name: {name!r}. Available: {sorted(set(_TASK_REGISTRY))}"
        ) from None
    return cls(**kwargs)


__all__ = [
    "get_judge",
]
