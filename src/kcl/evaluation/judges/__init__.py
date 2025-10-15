from .kbl import KBLEval
from .kcl import KCLEvalTasksRubricBased

_TASK_REGISTRY = {
    "kcl": KCLEvalTasksRubricBased,
    "kcl_oracle_rag": KCLEvalTasksRubricBased,
    "kbl": KBLEval,
    "kbl_oracle_rag": KBLEval,
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
    "KCLEvalTasksRubricBased",
    "get_judge",
]
