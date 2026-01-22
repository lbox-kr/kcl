import json

from datasets import load_dataset


class KCLMCQA:

    def __init__(
        self,
        with_precedents=False,
    ) -> None:
        self.with_precedents = with_precedents

    def _concat_columns(self, example):

        input_text = "다음은 변호사 시험 선택형 문제입니다.\n\n"
        input_text += f'문제: "{example["question"]}"\n\n'

        input_text += "선택지:\n"
        input_text += f'A. "{example["A"]}"\n'
        input_text += f'B. "{example["B"]}"\n'
        input_text += f'C. "{example["C"]}"\n'
        input_text += f'D. "{example["D"]}"\n'
        input_text += f'E. "{example["E"]}"\n\n'

        input_text += (
            "위의 문제와 각 선택지를 읽고 A, B, C, D, E 중 최종 답변을 출력하세요. "
            '최종 답변은 가장 마지막에 "정답은 X입니다." 와 같이 답해 주세요.\n'
        )
    
        if self.with_precedents:
            input_text += "[참고판례]:\n"

            for content in example["supporting_precedents"]:
                content_dict = json.loads(content)
                for case_name, case_content in content_dict.items():
                    input_text += "\n".join([case_name, case_content])
                input_text += "\n\n"

        

        return {"input_text": input_text.strip()}

    def load(self):

        ds = load_dataset("lbox/kcl", "kcl_mcqa", split="test")
        ds = ds.map(self._concat_columns, load_from_cache_file=False)

        return ds

    def __call__(self):
        return self.load()


if __name__ == "__main__":
    kcl_mcqa = KCLMCQA(with_precedents=True)
    task = kcl_mcqa()
    print(task["ds"][0]["input_text"])
