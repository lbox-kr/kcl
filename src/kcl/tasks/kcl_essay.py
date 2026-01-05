import json

from datasets import load_dataset


class KCLEssay:
    def __init__(
        self,
        with_precedents=False,
    ) -> None:
        self.with_precedents = with_precedents
        self.long_precedents = [
            "변호사시험 08회 공법 제1문의 1",
            "변호사시험 10회 형사법 제2문 1.",
            "변호사시험 11회 형사법 제1문 3. (나)",
            "변호사시험 11회 형사법 제2문 1. (가)",
            "변호사시험 12회 민사법 제3문의 1 1. 가.",
            "변호사시험 12회 민사법 제3문의 1 1. 나.",
            "변호사시험 12회 형사법 제2문 1. 가.",
            "변호사시험 13회 민사법 제1문의 5 1.",
            "변호사시험 13회 민사법 제3문 5.",
            "변호사시험 14회 형사법 제1문 2.",
            "변호사시험 14회 형사법 제2문 1.",
        ]

    def _concat_columns(self, example):
        input_text = "다음은 변호사 시험 사례형 문제입니다.\n\n"
        input_text += f'문제: "{example["question"]}"\n\n'
        if self.with_precedents:

            input_text += "[참고판례]:\n"

            for content in example["supporting_precedents"]:
                content_dict = json.loads(content)
                for case_name, case_content in content_dict.items():
                    if example["meta"] in self.long_precedents:
                        input_text += "\n".join(
                            [case_name, case_content.split("\n\n사건")[0]]
                        )
                    else:
                        input_text += "\n".join([case_name, case_content])

                input_text += "\n\n"

        return {"input_text": input_text.strip()}

    def load(self):

        ds = load_dataset("lbox/kcl", "kcl_essay", split="test")
        ds = ds.map(self._concat_columns, load_from_cache_file=False)

        return ds

    def __call__(self):
        return self.load()


if __name__ == "__main__":
    kcl_essay = KCLEssay(with_precedents=True)
    task = kcl_essay()
    print(task["ds"][0]["input_text"])
