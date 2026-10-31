import os
import shutil
import subprocess
import tempfile
import time
import uuid

import cv2
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO

from tracking import track_video

MODEL_PATH = os.getenv("MODEL_PATH", "weights/best.pt")
OUTPUT_DIR = "outputs"
KEEP_RESULTS_FOR = 60 * 60  # seconds
# processing every frame is slow on a CPU, 2 means every other frame
VIDEO_STEP = int(os.getenv("VIDEO_STEP", "2"))

VIDEO_TYPES = {".mp4", ".avi", ".mov", ".mkv"}
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

app = FastAPI()
model = YOLO(MODEL_PATH)

os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/predict")
def predict(file: UploadFile):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in VIDEO_TYPES | IMAGE_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    clear_old_results()

    # each upload gets its own folder so two people using the app don't overwrite each other
    job_id = uuid.uuid4().hex[:12]
    job_dir = os.path.join(OUTPUT_DIR, job_id)
    os.makedirs(job_dir)

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        upload_path = tmp.name

    try:
        if ext in IMAGE_TYPES:
            return detect_image(upload_path, job_dir)
        return detect_video(upload_path, job_dir)
    finally:
        os.remove(upload_path)


def detect_image(path, job_dir):
    result = model.predict(path, conf=0.5, imgsz=800, verbose=False)[0]
    out_path = os.path.join(job_dir, "detected.jpg")
    cv2.imwrite(out_path, result.plot())

    counts = {}
    for cls in result.boxes.cls.int().tolist():
        name = model.names[cls]
        counts[name] = counts.get(name, 0) + 1

    return {"type": "image", "output_path": "/" + out_path.replace(os.sep, "/"), "counts": counts}


def detect_video(path, job_dir):
    # the tracker keeps its state on the model object, so each video gets its own copy
    raw_video, csv_path, summary = track_video(YOLO(MODEL_PATH), path, job_dir, step=VIDEO_STEP)

    # OpenCV writes mp4v, which most browsers won't play, so convert to H.264
    web_video = os.path.join(job_dir, "tracked_web.mp4")
    if to_h264(raw_video, web_video):
        os.remove(raw_video)
    else:
        web_video = raw_video

    return {
        "type": "video",
        "output_path": "/" + web_video.replace(os.sep, "/"),
        "csv_path": "/" + csv_path.replace(os.sep, "/"),
        "summary": summary,
    }


def to_h264(src, dst):
    try:
        done = subprocess.run(
            ["ffmpeg", "-y", "-i", src, "-vcodec", "libx264", "-crf", "23", "-preset", "veryfast",
             "-pix_fmt", "yuv420p", dst],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return False
    return done.returncode == 0 and os.path.exists(dst)


def clear_old_results():
    now = time.time()
    for name in os.listdir(OUTPUT_DIR):
        folder = os.path.join(OUTPUT_DIR, name)
        if os.path.isdir(folder) and now - os.path.getmtime(folder) > KEEP_RESULTS_FOR:
            shutil.rmtree(folder, ignore_errors=True)
