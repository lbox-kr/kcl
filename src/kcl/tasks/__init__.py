from .kbl import KBL
from .kbl_oracle_rag import KBLOracleRAG
from .kcl import KCL
from .kcl_oracle_rag import KCLOracleRAG

_TASK_REGISTRY = {
    "kbl": KBL,
    "kbl_oracle_rag": KBLOracleRAG,
    "kcl": KCL,
    "kcl_oracle_rag": KCLOracleRAG,
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
    "KBL",
    "KBLOracleRAG",
    "KCL",
    "KCLOracleRAG",
    "get_loader",
]
