import json
import logging
from collections import defaultdict
from pathlib import Path

import hydra
import yaml
from hydra.core.hydra_config import HydraConfig
from joblib import Parallel, delayed
from loguru import logger
from omegaconf import DictConfig
from tqdm.auto import tqdm

from kcl.evaluation.judges import get_judge


@hydra.main(version_base=None, config_path=None, config_name=None)
def main(cfg: DictConfig):

    if not cfg.verbose:
        logging.getLogger("httpx").propagate = False
        logging.getLogger("google_genai.models").propagate = False

    input_dir = Path(cfg.get("input_dir", ""))
    if not input_dir:
        raise ValueError("input_dir must be specified in the config")

    model_dir = input_dir.parent
    task_dir = model_dir.parent
    inference_config_path = input_dir / ".hydra" / "config.yaml"

    inference_config = yaml.safe_load(inference_config_path.read_text())
    tasks = inference_config.get("tasks")
    inference_model_name = inference_config.get("model_name")

    if model_dir.name != inference_model_name.split("/")[-1]:
        raise ValueError(
            f"Model directory name '{model_dir.name}' does not match inference model name '{inference_model_name}'"
        )
    if task_dir.name != tasks:
        raise ValueError(
            f"Task directory name '{task_dir.name}' does not match tasks '{tasks}'"
        )

    run_dir = Path(HydraConfig.get().runtime.output_dir)
    (run_dir / "inference_configs.yaml").write_text(
        yaml.safe_dump(inference_config, allow_unicode=True)
    )

    save_root_dir = run_dir / inference_model_name / tasks / "results"

    inference_results_dir = input_dir / "results"

    sub_task_dirs = [d for d in inference_results_dir.iterdir() if d.is_dir()]
    inference_results_flattened = []
    for sub_task_dir in sub_task_dirs:
        sub_task_name = sub_task_dir.name
        results_jsons = list(sub_task_dir.glob("*.json"))
        for result_json in results_jsons:
            inference_result = json.loads(result_json.read_text())
            for item in inference_result:
                inference_results_flattened.append([sub_task_name, item])

    judge = get_judge(tasks, **cfg["judge_model"])
    eval_results = Parallel(n_jobs=cfg.get("n_jobs", 1), backend="threading")(
        delayed(judge)(sample)
        for _, sample in tqdm(inference_results_flattened, desc="Evaluating")
    )

    save_root_dir.mkdir(parents=True, exist_ok=True)

    final_results: dict[str, list[dict]] = defaultdict(list)
    for (sub_task_name, _), eval_result in zip(
        inference_results_flattened, eval_results
    ):
        final_results[sub_task_name].append(eval_result)

    for sub_task_name, samples in final_results.items():
        with open(
            save_root_dir / f"{sub_task_name}.json", "w", encoding="utf-8"
        ) as f:
            json.dump(samples, f, ensure_ascii=False, indent=4)

        logger.info(
            f"Saved evaluation results for {sub_task_name} → {sub_task_name}.json"
        )

    score_summation = {
        k: sum([item["normalized_score_sum"] for item in final_results[k]])
        for k in sorted(final_results.keys())
    }

    score_md_table = ""
    score_md_table += "| Task Name | Score |\n"
    score_md_table += "| --- | --- |\n"
    score_md_table += "\n".join(
        [
            f"| {task_name} | {score:.2f} |"
            for task_name, score in score_summation.items()
        ]
    )
    (save_root_dir / "a_score_summary.md").write_text(score_md_table)


if __name__ == "__main__":
    main()
