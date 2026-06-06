import os
import subprocess

def _has_audio_stream(video_path: str) -> bool:
    """ffprobe로 영상에 오디오 스트림이 있는지 확인"""
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=codec_type",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True)
    return "audio" in result.stdout

def _generate_silent_audio(output_path: str, duration_sec: float = 5.0):
    """오디오가 없는 영상을 위한 무음 WAV 파일 생성"""
    print(f"[음성 추출] 오디오 스트림 없음 → 무음 WAV 생성 ({duration_sec}초)")
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"anullsrc=r=16000:cl=mono",
        "-t", str(duration_sec),
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"무음 WAV 생성 실패: {result.stderr}")

def _get_video_duration(video_path: str) -> float:
    """ffprobe로 영상 길이(초) 반환"""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except (ValueError, AttributeError):
        return 5.0


def extract_audio(video_path: str, output_dir: str):
    #영상에서 음성 데이터를 추출하는 함수
    """
    video_path : 파일 경로
    output_dir : 음성 파일 저장 디렉토리
    frame_interval = N 프레임마다 1장 추출  
    """

    audio_path = os.path.join(output_dir, "audio.wav")

    if not _has_audio_stream(video_path):
        duration = _get_video_duration(video_path)
        _generate_silent_audio(audio_path, duration)
        print(f"[음성 추출] 무음 WAV 완료 → {audio_path}")
        return audio_path


    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",                  # 영상 제외
        "-acodec", "pcm_s16le", # WAV 포맷
        "-ar", "16000",         # 16kHz (Whisper 권장)
        "-ac", "1",             # 모노
        audio_path
    ]

    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"음성 추출 실패: {result.stderr}")
 
    print(f"[음성 추출] 완료 → {audio_path}")
    return audio_path

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    video_path = os.path.join(parent_dir, "Analyze_Test.mp4")
    output_dir = os.path.join(parent_dir, "output")
    extract_audio(video_path, output_dir)