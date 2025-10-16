import json
import re

from jsonfixer import fix_quotes
from loguru import logger


def parse_json_from_raw_string(raw_json_str):
    success = False

    json_codeblock = fix_quotes(
        raw_json_str,
        parse_code=True,
        replace_smart=True,
    )

    try:
        parsed_json = json.loads(json_codeblock)
        if "점수" in parsed_json:

            if parsed_json["점수"].__class__ in [float, int]:
                parsed_json["점수"] = float(parsed_json["점수"])

            else:
                raise ValueError("점수 is not a number")

            success = True

            return parsed_json, success

        else:
            raise KeyError("No 점수 field found")

    except Exception as e:
        logger.error(f"Json have an Error: {e} in {json_codeblock}")

    score = extract_score_from_json_string(json_codeblock)
    if score is not None:
        parsed_json = {
            "점수": score,
            "근거": raw_json_str,
        }
        success = True
        return parsed_json, success

    parsed_json = {
        "점수": 0,
        "근거": raw_json_str,
    }

    return parsed_json, success


def extract_score_from_json_string(json_string):
    try:
        data = json.loads(json_string)

        if "점수" in data:
            return float(data["점수"])
        else:
            return None

    except Exception as e:
        logger.info(f"Failed to parse JSON: {e}")
        score_match = re.search(r'"점수"\s*:\s*(\d+(?:\.\d+)?)', json_string)
        if score_match:
            try:
                return float(score_match.group(1))
            except Exception as e:
                logger.error(f"Failed to convert score to float: {e}")
                return None
        return None
