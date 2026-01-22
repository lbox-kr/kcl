import json
import logging
import random
import time
from collections import defaultdict
from functools import partial
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from joblib import Parallel, delayed
from loguru import logger
from omegaconf import DictConfig
from tenacity import retry, stop_after_attempt, wait_fixed
from tqdm.auto import tqdm

from kcl.models import get_model
from kcl.tasks import get_loader

MAX_RETRY = 5
RETRY_WAIT_SEC = 10

tqdm = partial(tqdm, dynamic_ncols=True)


def retry_log(retry_state):
    logger.warning(
        f"[Retry {retry_state.attempt_number}/{MAX_RETRY}] {retry_state.outcome.exception()}",
    )
    base = RETRY_WAIT_SEC * (2**retry_state.attempt_number)
    jitter = random.uniform(0, RETRY_WAIT_SEC)
    time.sleep(base + jitter)


def process(model, sample, task_name):
    try:
        out_text = generate_sample(model, sample)
        success = True
        error_msg = None
    except Exception as exc:
        out_text = ""
        success = False
        error_msg = str(exc)

    out = sample.copy()
    out["model_output"] = out_text
    if not success:
        out["error"] = error_msg

    return task_name, out


@retry(
    stop=stop_after_attempt(MAX_RETRY),
    wait=wait_fixed(RETRY_WAIT_SEC),
    before_sleep=retry_log,
    reraise=True,
)
def generate_sample(model, sample):
    return model.generate(sample["input_text"])


@hydra.main(version_base=None, config_path=None, config_name=None)
def main(cfg: DictConfig):

    logging.getLogger("httpx").propagate = cfg.verbose
    logging.getLogger("google_genai.models").propagate = cfg.verbose

    model_kwargs = cfg.get("model_kwargs", {})
    model = get_model(cfg.model_name, **model_kwargs)

    loader = get_loader(cfg.tasks, **cfg.tasks_kwargs)
    task = loader.load()

    flat_samples = [(task._info.config_name, sample) for sample in task]

    run_dir = Path(HydraConfig.get().runtime.output_dir)
    save_root = run_dir / "results"
    cfg_name = HydraConfig.get().job.config_name

    n_jobs = cfg.get("n_jobs", 1)
    if n_jobs > 1:
        results = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(process)(model, s, t)
            for t, s in tqdm(
                flat_samples,
                desc="Processing samples",
                total=len(flat_samples),
            )
        )
    else:
        results = [
            process(model, s, t)
            for t, s in tqdm(
                flat_samples,
                desc="Processing samples",
                total=len(flat_samples),
            )
        ]

    if hasattr(model, "cleanup"):
        logger.info("Cleaning up model resources...")
        model.cleanup()

    by_task = defaultdict(list)
    for task_name, out in results:
        by_task[task_name].append(out)

    for task_name, samples in by_task.items():
        task_dir = save_root / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        out_file = task_dir / f"{cfg_name}.json"
        out_file.write_text(json.dumps(samples, ensure_ascii=False, indent=4))

        logger.info(f"Saved {task_name}: {len(samples)} → {out_file}")


if __name__ == "__main__":
    main()
