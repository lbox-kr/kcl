import re
from functools import partial

from tqdm.auto import tqdm

tqdm = partial(tqdm, dynamic_ncols=True)


class KBLEval:

    def __init__(self, **cfg):
        pass

    def judge(self, item):
        gt = item["gt"]
        model_output = item["model_output"]

        match = re.search(r"[a-zA-Z]", model_output)
        if match:
            first_alpha = match.group(0)
        else:
            first_alpha = ""

        item["right"] = gt.lower() == first_alpha.lower()
        item["normalized_score_sum"] = int(item["right"])

        return item

    def __call__(self, item):
        return self.judge(item)
