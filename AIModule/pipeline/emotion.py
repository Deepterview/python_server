import numpy as np
from collections import deque
from deepface import DeepFace


class EmotionAnalyzer:
    def __init__(self, smoothing_window=5):
        self.history = deque(maxlen=smoothing_window)

    def analyze_emotion(self, frame: np.ndarray) -> dict:
        try:
            result = DeepFace.analyze(
                img_path=frame,
                actions=["emotion"],
                enforce_detection=False,
                silent=True
            )
            if isinstance(result, list):
                result = result[0]

            emotions = result["emotion"]
            self.history.append(emotions)

            smoothed = {}
            for key in emotions:
                smoothed[key] = round(
                    np.mean([h[key] for h in self.history]), 2
                )

            dominant = max(smoothed, key=smoothed.get)
            confidence = result.get("face_confidence", 0.0)

            return {
                "dominant_emotion": dominant,
                "emotions": smoothed,
                "confidence": round(confidence, 3),
                "face_detected": confidence > 0.5
            }

        except Exception as e:
            return {
                "dominant_emotion": "unknown",
                "emotions": {},
                "confidence": 0.0,
                "face_detected": False,
                "error": str(e)
            }