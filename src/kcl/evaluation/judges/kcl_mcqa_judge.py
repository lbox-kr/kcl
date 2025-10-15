import re
from functools import partial
from typing import Any, Dict, List, Optional

from tqdm.auto import tqdm

tqdm = partial(tqdm, dynamic_ncols=True)


def _build_choice_re(choices: str) -> str:

    upper = "".join(sorted(set(c for c in choices.upper())))
    lower = upper.lower()
    return f"[{upper}{lower}]"


class KCLMCQAEval:

    def __init__(
        self,
        *,
        choices: str = "ABCDE",
        allow_fallback: bool = True,
        debug: bool = False,
    ):

        self.choices = choices
        self.allow_fallback = allow_fallback
        self.debug = debug

        choice_re = _build_choice_re(self.choices)

        self._primary_re = re.compile(
            rf"정답은\s*[\(\[\"'“”‘’『「]?\s*({choice_re})\s*[\)\]\"'“”‘’』」]?\s*입니다[\.!\u3002]*",
            flags=re.IGNORECASE,
        )

        secondary_patterns = [
            rf"(?:최종\s*정답|최종정답)\s*[:：]?\s*[\(\[\"'“”‘’『「]?\s*({choice_re})\s*[\)\]\"'“”‘’』」]?",
            rf"정답\s*[:：]?\s*[\(\[\"'“”‘’『「]?\s*({choice_re})\s*[\)\]\"'“”‘’』」]?",
            rf"[\"'“”‘’]?답변\s*[:：]?\s*[\(\[\"'“”‘’『「]?\s*({choice_re})\s*[\)\]\"'“”‘’』」]?",
            rf"응답\s*[:：]?\s*[\(\[\"'“”‘’『「]?\s*({choice_re})\s*[\)\]\"'“”‘’』」]?",
            rf"(?:Final\s*Answer|Answer)\s*[:：]?\s*[\(\[]?\s*({choice_re})\s*[\)\]]?",
            rf"\"answer\"\s*:\s*\"({choice_re})\"",
            rf"'answer'\s*:\s*'({choice_re})'",
            rf"(?:최종\s*정답|최종정답)\s*[:：]?\s*[\(\[\"'“”‘’『「]?\s*({choice_re})\s*[\)\]\"'“”‘’』」]?",
            rf"정답\s*[:：]?\s*[\(\[\"'“”‘’『「]?\s*({choice_re})\s*[\)\]\"'“”‘’』」]?",
            rf"(?:Final\s*Answer|Answer)\s*[:：]?\s*[\(\[]?\s*({choice_re})\s*[\)\]]?",
            rf"\"answer\"\s*:\s*\"({choice_re})\"",
            rf"'answer'\s*:\s*'({choice_re})'",
        ]
        self._secondary_res = [
            re.compile(p, flags=re.IGNORECASE) for p in secondary_patterns
        ]

        self._fallback_re = re.compile(
            rf"\b({choice_re})\b(?!.*\b{choice_re}\b)",
            flags=re.IGNORECASE | re.DOTALL,
        )

    def _extract_primary(self, text: str) -> Optional[str]:
        m = self._primary_re.search(text)
        return m.group(1) if m else None

    def _extract_secondary(self, text: str) -> Optional[str]:
        for rgx in self._secondary_res:
            m = rgx.search(text)
            if m:
                return m.group(1)
        return None

    def _extract_fallback(self, text: str) -> Optional[str]:
        if not self.allow_fallback:
            return None
        m = self._fallback_re.search(text)
        return m.group(1) if m else None

    def _extract_answer(self, text: str) -> Optional[str]:

        if not text:
            return None

        for extractor in (
            self._extract_primary,
            self._extract_secondary,
            self._extract_fallback,
        ):
            ans = extractor(text)
            if ans:
                return ans.upper()
        return None

    def judge(self, item):

        gt_raw = item.get("gt", "")
        model_output = item.get("model_output") or ""

        gt = str(gt_raw).strip().upper()
        pred = self._extract_answer(str(model_output))

        right = pred is not None and gt == pred

        if self.debug:
            print(f"pred={pred!r} gt={gt!r}")

        item["right"] = right
        item["normalized_score_sum"] = int(right)
        return item

    def __call__(self, item):
        return self.judge(item)


if __name__ == "__main__":

    def run_cases(
        evaluator: KCLMCQAEval,
        cases: List[Dict[str, Any]],
        title: str,
    ) -> None:
        print(f"\n=== {title} ===")
        n_right = 0
        for i, item in enumerate(cases, 1):
            out = evaluator.judge(dict(item))
            n_right += int(out["right"])
            print(
                f"[{i:02d}] gt={item['gt']} | right={out['right']} | normalized={out['normalized_score_sum']} | "
                f"snippet={str(item.get('model_output'))[:80]!r}"
            )
        print(f"Total: {n_right}/{len(cases)} correct")

    base_cases = [
        {"gt": "A", "model_output": "정답은 (A)입니다."},
        {"gt": "B", "model_output": "정답은  B  입니다!"},
        {"gt": "C", "model_output": "정답은『C』입니다。"},
        {"gt": "D", "model_output": "최종정답: D"},
        {"gt": "E", "model_output": "정답: (e)"},
        {"gt": "C", "model_output": "Final Answer: c"},
        {"gt": "B", "model_output": '"answer": "B"'},
        {"gt": "A", "model_output": "'answer': 'a'"},
        {"gt": "A", "model_output": "답변: A"},
        {"gt": "D", "model_output": "“답변 :  ( d ) ” 라고 정리하겠습니다."},
        {"gt": "E", "model_output": "응답: e"},
        {
            "gt": "C",
            "model_output": (
                "선지:\nA. 사과\nB. 바나나\nC. 체리\n해설: 여러 가능성을 검토한 끝에 C 가 가장 적합합니다"
            ),
        },
        {
            "gt": "B",
            "model_output": (
                "선지: A / B / C / D / E\n결론적으로 정답은 증거2(B)에 의해 뒷받침됩니다."
            ),
        },
        {"gt": "A", "model_output": "정답을 직접적으로 제시하지 않았습니다."},
        {"gt": "C", "model_output": None},
        {
            "gt": "C",
            "model_output": "Because we compare options, the correct is (C).",
        },
        {"gt": "A", "model_output": "정답은 X가 아니라 A입니다."},
    ]

    eval_default = KCLMCQAEval(
        choices="ABCDE", allow_fallback=True, debug=True
    )
    run_cases(
        eval_default,
        base_cases,
        "Default (choices=ABCDE, allow_fallback=True)",
    )

    no_fallback_cases = [
        {
            "gt": "C",
            "model_output": (
                "선지:\nA. 사과\nB. 바나나\nC. 체리\n해설: 여러 가능성을 검토한 끝에 C 가 가장 적합합니다"
            ),
        },
        {
            "gt": "B",
            "model_output": (
                "선지: A / B / C / D / E\n결론적으로 정답은 증거2(B)에 의해 뒷받침됩니다."
            ),
        },
    ]
    eval_no_fb = KCLMCQAEval(choices="ABCDE", allow_fallback=False, debug=True)
    run_cases(
        eval_no_fb,
        no_fallback_cases,
        "No Fallback (choices=ABCDE, allow_fallback=False)",
    )

    four_choice_cases = [
        {"gt": "D", "model_output": "정답: D"},
        {"gt": "E", "model_output": "정답: E"},
        {
            "gt": "B",
            "model_output": "A. 항목1\nB. 항목2\n설명 후 결론은 B 가 타당.",
        },
    ]
    eval_four = KCLMCQAEval(choices="ABCD", allow_fallback=True, debug=True)
    run_cases(
        eval_four,
        four_choice_cases,
        "Four Choices (choices=ABCD, allow_fallback=True)",
    )

    must_pass = [
        {"gt": "A", "model_output": "정답은 (A)입니다."},
        {"gt": "B", "model_output": '"answer": "B"'},
        {"gt": "A", "model_output": "'답변: A"},
        {"gt": "A", "model_output": "답변: A"},
        {"gt": "C", "model_output": "Final Answer: c"},
        {
            "gt": "C",
            "model_output": "답안 B 와 답안 C 를 비교하면 Final Answer: c",
        },
    ]
    for it in must_pass:
        checked = eval_default.judge(dict(it))
        assert checked["right"] is True, f"Must-pass case failed: {it}"
    print("\nAssertions passed for must-pass cases.")
