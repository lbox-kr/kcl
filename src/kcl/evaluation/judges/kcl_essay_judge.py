from functools import partial

from google.genai import types
from tqdm.auto import tqdm

from kcl.evaluation.utils.text_utils import parse_json_from_raw_string
from kcl.models import get_model

tqdm = partial(tqdm, dynamic_ncols=True)


class KCLEssayEval:

    def __init__(self, **cfg):
        self.model = get_model(cfg["model_name"], **cfg["kwargs"])
        self.score_per_rubric = cfg["score_per_rubric"]
        self.prompt_templates = cfg["prompt_templates"]

    def judge(self, item):
        _item = item.copy()
        rubrics = item["rubrics"]
        grades = {}

        input_text_prefix = ""
        for r_id, rubric in enumerate(rubrics):
            if r_id == 0:
                instruction = f"{self.prompt_templates['instruction']}\n"

                input_text_prefix += (
                    self.prompt_templates["model_answer_template"].format(
                        answer=item["model_output"],
                        score_per_rubric=self.score_per_rubric,
                    )
                    + "\n"
                )

                n_cache_toks = self.model.count_tokens(input_text_prefix)
                if n_cache_toks < 1024:
                    cache = None

                else:
                    cache = self.model.client.caches.create(
                        model=self.model.model_name,
                        config=types.CreateCachedContentConfig(
                            display_name="judge_prompt_and_model_answer",
                            system_instruction=instruction,
                            contents=[input_text_prefix],
                        ),
                    )

            criterion = f"{rubric} (점수: {self.score_per_rubric})"

            if cache is not None:
                input_text = (
                    self.prompt_templates["rubric_template"]
                    .format(rubrics_with_score=criterion)
                    .strip()
                )
                judge_output = self.model.generate(
                    prompt=input_text,
                    cache=cache,
                )

            else:
                input_text = (
                    instruction
                    + input_text_prefix
                    + self.prompt_templates["rubric_template"]
                    .format(rubrics_with_score=criterion)
                    .strip()
                )
                judge_output = self.model.generate(
                    prompt=input_text,
                )

            grade, success = self._parse_answer_judge(
                judge_output,
                item["score"],
                len(rubrics),
                self.score_per_rubric,
                criterion,
                rubric_id=r_id,
            )
            grades[str(r_id)] = {
                "grade": grade,
                "success": success,
            }

        _item["judge_input"] = input_text
        _item["grades"] = grades

        _item["normalized_score_sum"] = sum(
            v["grade"]["문항점수로정규화된점수"]
            for v in grades.values()
            if v["success"]
        )

        return _item

    def __call__(self, item):
        return self.judge(item)

    def _parse_answer_judge(
        self,
        grade_raw_json_str,
        original_full_score,
        n_rubrics,
        score_per_rubric,
        rubric_with_score,
        rubric_id,
    ):
        if grade_raw_json_str is None:
            parsing_success = False
        else:
            grade, parsing_success = parse_json_from_raw_string(
                grade_raw_json_str
            )
            if parsing_success:
                grade["척도"] = rubric_with_score
                grade["문항점수로정규화된점수"] = (
                    grade["점수"]
                    / score_per_rubric
                    * original_full_score
                    / n_rubrics
                )
                grade["문항점수로정규화된만점점수"] = (
                    original_full_score / n_rubrics
                )
        if not parsing_success:
            grade = {
                "평가척도번호": rubric_id,
                "점수": None,
                "근거": None,
                "문항점수로정규화된점수": None,
                "문항점수로정규화된만점점수": original_full_score / n_rubrics,
            }
        return grade, parsing_success
