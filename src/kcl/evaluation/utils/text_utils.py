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
        if "item_score" in parsed_json:

            if parsed_json["item_score"].__class__ in [float, int]:
                parsed_json["item_score"] = float(parsed_json["item_score"])

            else:
                raise ValueError("item_score is not a number")

            success = True

            return parsed_json, success

        else:
            raise KeyError("No item_score field found")

    except Exception as e:
        logger.error(f"Json have an Error: {e} in {json_codeblock}")

    score = extract_score_from_json_string(json_codeblock)
    if score is not None:
        parsed_json = {
            "item_score": score,
            "reason": raw_json_str,
        }
        success = True
        return parsed_json, success

    parsed_json = {
        "item_score": 0,
        "reason": raw_json_str,
    }

    return parsed_json, success


def extract_score_from_json_string(json_string):
    try:
        data = json.loads(json_string)

        if "item_score" in data:
            return float(data["item_score"])
        else:
            return None

    except Exception as e:
        logger.info(f"Failed to parse JSON: {e}")
        score_match = re.search(r'"item_score"\s*:\s*(\d+(?:\.\d+)?)', json_string)
        if score_match:
            try:
                return float(score_match.group(1))
            except Exception as e:
                logger.error(f"Failed to convert score to float: {e}")
                return None
        return None
