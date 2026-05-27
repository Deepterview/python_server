import uuid
import httpx
from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel

router = APIRouter()

# 메모리 내 job 상태 저장 (추후 Redis로 교체 가능)
jobs: dict[str, str] = {}


class AnalyzeRequest(BaseModel):
    video_path: str
    interview_id: str
    callback_url: str  # Spring Boot 콜백 엔드포인트


class AnalyzeResponse(BaseModel):
    job_id: str
    status: str


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_video(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs[job_id] = "processing"
    background_tasks.add_task(_run_and_callback, job_id, request)
    return AnalyzeResponse(job_id=job_id, status="processing")


@router.get("/analyze/{job_id}", response_model=AnalyzeResponse)
async def get_job_status(job_id: str):
    status = jobs.get(job_id, "not_found")
    return AnalyzeResponse(job_id=job_id, status=status)


async def _run_and_callback(job_id: str, request: AnalyzeRequest):
    from core.runner import run_pipeline

    try:
        result = run_pipeline(request.video_path)
        jobs[job_id] = "done"
        payload = {
            "job_id": job_id,
            "interview_id": request.interview_id,
            "status": "done",
            "result": result,
        }
    except Exception as e:
        jobs[job_id] = "error"
        payload = {
            "job_id": job_id,
            "interview_id": request.interview_id,
            "status": "error",
            "error": str(e),
        }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(request.callback_url, json=payload)
    except Exception as e:
        print(f"[콜백 전송 실패] {e}")
