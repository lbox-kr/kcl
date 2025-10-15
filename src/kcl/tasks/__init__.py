from .kcl_essay import KCLEssay
from .kcl_mcqa import KCLMCQA

_TASK_REGISTRY = {
    "kcl_essay": KCLEssay,
    "kcl_mcqa": KCLMCQA,
}


def get_loader(name, **kwargs):
    key = name.lower()
    try:
        cls = _TASK_REGISTRY[key]
    except KeyError:
        raise ValueError(
            f"Unknown task name: {name!r}. Available: {sorted(set(_TASK_REGISTRY))}"
        ) from None
    return cls(**kwargs)


__all__ = [
    "get_loader",
]
