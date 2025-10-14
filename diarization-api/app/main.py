from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import uuid
import shutil
import asyncio
from audio_pipeline import AudioPipeline

app = FastAPI(title="Diarization API")

# Simple in-memory job store; for production replace with persistent store
jobs = {}

WORK_ROOT = Path("/app/work")
WORK_ROOT.mkdir(parents=True, exist_ok=True)


@app.post("/process")
async def process_audio(file: UploadFile = File(...), num_speakers: int = Form(2), skip_llm: bool = Form(False)):
    # Save uploaded file to a unique working folder
    job_id = str(uuid.uuid4())
    job_dir = WORK_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    audio_path = job_dir / file.filename

    with audio_path.open("wb") as f:
        content = await file.read()
        f.write(content)

    # Kick off processing in background
    pipeline = AudioPipeline(work_dir=str(job_dir), num_speakers=num_speakers)

    async def run_pipeline():
        try:
            result = pipeline.process_audio(str(audio_path), skip_llm=skip_llm)
            jobs[job_id] = {"status": "done", "result": result}
        except Exception as e:
            jobs[job_id] = {"status": "error", "error": str(e)}

    jobs[job_id] = {"status": "running"}
    asyncio.create_task(run_pipeline())

    return JSONResponse({"job_id": job_id, "status_endpoint": f"/status/{job_id}"})


@app.get("/status/{job_id}")
def status(job_id: str):
    info = jobs.get(job_id)
    if not info:
        raise HTTPException(status_code=404, detail="Job not found")
    return info


@app.get("/")
def read_root():
    return {"status": "ok", "message": "Diarization API up"}
