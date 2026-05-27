def filter_emotion_result(result: dict, min_confidence: float = 0.5) -> dict:
    if not result.get("face_detected"):
        return None

    if result.get("confidence", 0) < min_confidence:
        return None

    emotions = result.get("emotions", {})
    if emotions.get("neutral", 0) > 95:
        return None

    return result


def filter_gaze_result(result: dict) -> dict:
    if not result.get("face_detected"):
        return None

    avg_ratio = result.get("avg_ratio", 0.5)
    if avg_ratio < 0.1 or avg_ratio > 0.9:
        return None

    return result