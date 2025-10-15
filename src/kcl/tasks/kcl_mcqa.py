import json

from datasets import load_dataset


class KCLMCQA:

    def __init__(
        self,
        with_precedents=False,
    ) -> None:
        self.with_precedents = with_precedents

    def _concat_columns(self, example):

        input_text = ""

        if self.with_precedents:

            input_text += "다음의 [참고 판례]를 참조하여 [문제]에 대한 답을 선택지 중에서 고르세요.\n\n"
            input_text += "[참고 판례]: \n"

            for content in example["supporting_precedents"]:
                content_dict = json.loads(content)
                for case_name, case_content in content_dict.items():
                    input_text += "\n".join([case_name, case_content])

                input_text += "\n\n"

        input_text += "[문제]: \n"

        input_text += "다음 각 선택지를 읽고 A, B, C, D, E 중 하나를 선택하여 '답변: A' 와 같이 단답식으로 답해 주세요.\n\n"
        input_text += f'A. {example["A"]}\n\n'
        input_text += f'B. {example["B"]}\n\n'
        input_text += f'C. {example["C"]}\n\n'
        input_text += f'D. {example["D"]}\n\n'
        input_text += f'E. {example["E"]}\n\n'

        return {"input_text": input_text.strip()}

    def load(self):
        task = {
            "system_prompt": """
당신은 법률 전문가 입니다. 당신은 변호사 시험 사례형 문제의 답안을 작성해 줍니다. 주어진 법률 질문에 대하여 명확한 법적 근거(법령, 대법원 판례 등)와 함께 작성해 주세요.
""".strip()
        }

        ds = load_dataset("lbox/kcl", "kcl_mcqa", split="test")
        ds = ds.map(self._concat_columns, load_from_cache_file=False)
        task["ds"] = ds

        return task

    def __call__(self):
        return self.load()


if __name__ == "__main__":
    kcl_mcqa = KCLMCQA(with_precedents=True)
    task = kcl_mcqa()
    print(task["ds"][0]["input_text"])
