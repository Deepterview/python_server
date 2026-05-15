
from fastapi import APIRouter, BackgroundTasks
from core.runner import run_pipeline, send_to_spring

router = APIRouter()

#라우터 엔드포인트에 따라 구현
@router.post("/analyze")
async def analyze(payload: dict, background_tasks: BackgroundTasks):
    # 즉시 응답 반환
    background_tasks.add_task(run_and_callback, payload)
    return {"status": "processing"}  # Spring은 여기서 바로 받음

async def run_and_callback(payload: dict):
    # 백그라운드에서 분석 후 콜백
    result = run_pipeline(payload["video_path"])
    await send_to_spring(payload["callback_url"], result)