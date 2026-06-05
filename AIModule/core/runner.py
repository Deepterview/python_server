import os
import json
import cv2
import mediapipe as mp
from collections import Counter
from pipeline.extractor import extract_audio
from pipeline.audio import analyze_audio_librosa, transcribe_audio_whisper
from pipeline.audio_enhance import enhance_audio
from pipeline.video_enhance import enhance_frame
from pipeline.gaze import analyze_gaze
from pipeline.emotion import EmotionAnalyzer
from pipeline.result_filter import filter_emotion_result, filter_gaze_result


def _compute_nonverbal_summary(frame_results: list) -> dict:
    if not frame_results:
        return {
            "eye_contact_score": 0.0,
            "dominant_emotion": "unknown",
            "emotion_distribution_json": "{}",
            "smile_ratio": 0.0,
            "anxiety_score": 0.0,
            "confidence_score": 0.0,
            "head_stability_score": None,
        }

    n = len(frame_results)

    # 정면 응시 비율
    center_count = sum(1 for f in frame_results if f["gaze_direction"] == "center")
    eye_contact_score = round(center_count / n * 100, 2)

    # 전체 구간 최빈 감정
    dominant_emotion = Counter(f["dominant_emotion"] for f in frame_results).most_common(1)[0][0]

    # 감정별 평균 분포
    all_emotions = [f["emotions"] for f in frame_results if f.get("emotions")]
    if all_emotions:
        emotion_distribution = {
            k: round(sum(e.get(k, 0) for e in all_emotions) / len(all_emotions), 2)
            for k in all_emotions[0]
        }
    else:
        emotion_distribution = {}
    emotion_distribution_json = json.dumps(emotion_distribution)

    # 미소 비율
    smile_ratio = round(
        sum(1 for f in frame_results if f["dominant_emotion"] == "happy") / n * 100, 2
    )

    # 불안 점수: fear + sad + disgust 평균
    anxiety_score = round(
        sum(
            sum(f.get("emotions", {}).get(e, 0) for e in ["fear", "sad", "disgust"])
            for f in frame_results
        ) / n,
        2,
    )

    # 자신감 점수: 시선(40%) + 낮은불안(30%) + 얼굴감지신뢰도(30%)
    avg_face_confidence = sum(f.get("confidence", 0) for f in frame_results) / n
    confidence_score = round(
        (eye_contact_score * 0.4)
        + (max(0.0, 100.0 - anxiety_score) * 0.3)
        + (avg_face_confidence * 100 * 0.3),
        2,
    )

    return {
        "eye_contact_score": eye_contact_score,
        "dominant_emotion": dominant_emotion,
        "emotion_distribution_json": emotion_distribution_json,
        "smile_ratio": smile_ratio,
        "anxiety_score": anxiety_score,
        "confidence_score": confidence_score,
        "head_stability_score": None,
    }


def run_pipeline(video_path: str, output_dir: str = "output", frame_interval: int = 5) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    audio_path = extract_audio(video_path, output_dir)

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    emotion_analyzer = EmotionAnalyzer()

    frame_results = []
    frame_index = 0
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % frame_interval == 0:
            enhanced = enhance_frame(frame)
            gaze = analyze_gaze(enhanced, face_mesh)
            emotion = emotion_analyzer.analyze_emotion(enhanced)

            emotion_filtered = filter_emotion_result(emotion)
            gaze_filtered = filter_gaze_result(gaze)

            if emotion_filtered and gaze_filtered:
                frame_results.append({
                    "frame_index": frame_index,
                    "gaze_direction": gaze_filtered["gaze_direction"],
                    "dominant_emotion": emotion_filtered["dominant_emotion"],
                    "emotions": emotion_filtered.get("emotions", {}),
                    "confidence": emotion_filtered["confidence"],
                })

        frame_index += 1

    cap.release()
    face_mesh.close()

    enhanced_audio_path = enhance_audio(audio_path, os.path.join(output_dir, "enhanced_audio.wav"))
    audio_features = analyze_audio_librosa(enhanced_audio_path)
    transcription = transcribe_audio_whisper(enhanced_audio_path)

    return {
        "audio": {
            "tempo_bpm": audio_features.get("tempo_bpm"),
            "pitch_mean_hz": audio_features.get("pitch_mean_hz"),
            "pitch_std_hz": audio_features.get("pitch_std_hz"),
        },
        "transcription": {
            "text": transcription.get("text"),
            "language": transcription.get("language"),
            "segments": transcription.get("segments", []),
        },
        "frame_count": frame_index,
        "gaze_frames": frame_results,
        "nonverbal_summary": _compute_nonverbal_summary(frame_results),
    }


if __name__ == "__main__":
    result = run_pipeline(
        video_path="Analyze_Test.mp4",
        output_dir="output",
        frame_interval=5,
    )
    print(result)