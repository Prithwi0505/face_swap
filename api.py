#!/usr/bin/env python3
"""
FastAPI backend for the roop face-swapping pipeline.
Designed for Google Colab: loads models once, processes requests in-process.

Usage (Colab):
    !uvicorn api:app --host 0.0.0.0 --port 7860
"""

import os
import sys
import time
import uuid
import shutil
import warnings
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List

from database import (
    init_db,
    create_request,
    mark_in_progress,
    mark_completed,
    mark_failed,
    get_request,
    get_all_requests,
)
import cv2

# Patch OpenCV to prevent headless crashes from underlying libraries closing windows
cv2.destroyAllWindows = lambda *args, **kwargs: None
cv2.imshow = lambda *args, **kwargs: None

# Performance: single thread doubles CUDA performance
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import onnxruntime
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

import roop.globals
from roop.core import (
    encode_execution_providers,
    decode_execution_providers,
    suggest_execution_providers,
    suggest_execution_threads,
    limit_resources,
    update_status,
)
from roop.predictor import predict_video, predict_image
from roop.processors.frame.core import get_frame_processors_modules
from roop.utilities import (
    has_image_extension,
    is_image,
    is_video,
    detect_fps,
    create_video,
    extract_frames,
    get_temp_frame_paths,
    restore_audio,
    create_temp,
    move_temp,
    clean_temp,
    normalize_output_path,
)

warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')

# ---------------------------------------------------------------------------
# Directories & Files
# ---------------------------------------------------------------------------
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")


def _ensure_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Configure roop globals with sensible defaults for API / Colab usage
# ---------------------------------------------------------------------------
def _configure_globals() -> None:
    """Set roop global defaults suitable for headless API usage on Colab."""
    roop.globals.headless = True
    roop.globals.keep_fps = True
    roop.globals.keep_frames = False
    roop.globals.skip_audio = False
    roop.globals.many_faces = False
    roop.globals.reference_face_position = 0
    roop.globals.reference_frame_number = 0
    roop.globals.similar_face_distance = 0.85
    roop.globals.temp_frame_format = 'png'
    roop.globals.temp_frame_quality = 0
    roop.globals.output_video_encoder = 'libx264'
    roop.globals.output_video_quality = 1
    roop.globals.max_memory = None
    roop.globals.execution_threads = suggest_execution_threads()

    # Auto-detect best execution provider (CUDA if available, else CPU)
    available = encode_execution_providers(onnxruntime.get_available_providers())
    if 'cuda' in available:
        roop.globals.execution_providers = decode_execution_providers(['cuda'])
    else:
        roop.globals.execution_providers = decode_execution_providers(['cpu'])

    roop.globals.frame_processors = ['face_swapper', 'face_enhancer']


# ---------------------------------------------------------------------------
# Pre-load models so first request is fast
# ---------------------------------------------------------------------------
def _preload_models() -> None:
    """Download model weights if missing and warm-load the processor modules."""
    modules = get_frame_processors_modules(roop.globals.frame_processors)
    for module in modules:
        module.pre_check()  # downloads weights if not present
    print("[API] Models pre-checked and ready.")


# ---------------------------------------------------------------------------
# Core face-swap pipeline (Images)
# ---------------------------------------------------------------------------
def run_face_swap_image_pipeline(source_path: str, target_path: str, output_path: str) -> str:
    """
    Execute the face-swap pipeline in-process for images.
    Returns the path to the output image on success, raises on failure.
    """
    roop.globals.source_path = source_path
    roop.globals.target_path = target_path
    roop.globals.output_path = output_path

    # Validate processors can start
    frame_processor_modules = get_frame_processors_modules(roop.globals.frame_processors)
    for processor in frame_processor_modules:
        if not processor.pre_start():
            raise RuntimeError(f"Processor {getattr(processor, 'NAME', 'unknown')} pre_start failed. "
                               "Check that the source image contains a detectable face.")

    # NSFW check on target image
    if is_image(target_path):
        if predict_image(target_path):
            raise ValueError("NSFW content detected in the target image. Processing refused.")
    else:
        raise ValueError("Target file is not a valid image.")

    shutil.copy2(target_path, output_path)

    # Process frame
    for processor in frame_processor_modules:
        update_status(f'Processing...', getattr(processor, 'NAME', 'PROCESSOR'))
        processor.process_image(source_path, output_path, output_path)
        processor.post_process()

    if not is_image(output_path):
        raise RuntimeError("Processing to image failed!")

    update_status('Processing complete!')
    return output_path


# ---------------------------------------------------------------------------
# Core face-swap pipeline (mirrors roop.core.start for video targets)
# ---------------------------------------------------------------------------
def run_face_swap_pipeline(source_path: str, target_path: str, output_path: str) -> str:
    """
    Execute the full face-swap pipeline in-process.
    Returns the path to the output video on success, raises on failure.
    """
    # Store paths in globals (processors read from there)
    roop.globals.source_path = source_path
    roop.globals.target_path = target_path
    roop.globals.output_path = output_path

    # Validate processors can start
    frame_processor_modules = get_frame_processors_modules(roop.globals.frame_processors)
    for processor in frame_processor_modules:
        if not processor.pre_start():
            raise RuntimeError(f"Processor {getattr(processor, 'NAME', 'unknown')} pre_start failed. "
                               "Check that the source image contains a detectable face.")

    # NSFW check on target video
    if is_video(target_path):
        if predict_video(target_path):
            raise ValueError("NSFW content detected in the target video. Processing refused.")
    else:
        raise ValueError("Target file is not a valid video.")

    # --- Frame extraction ---
    update_status('Creating temporary resources...')
    create_temp(target_path)

    if roop.globals.keep_fps:
        fps = detect_fps(target_path)
        update_status(f'Extracting frames with {fps} FPS...')
        extract_frames(target_path, fps)
    else:
        fps = 30.0
        update_status('Extracting frames with 30 FPS...')
        extract_frames(target_path)

    # --- Process each frame ---
    temp_frame_paths = get_temp_frame_paths(target_path)
    if not temp_frame_paths:
        raise RuntimeError("No frames were extracted from the video.")

    for processor in frame_processor_modules:
        update_status(f'Processing frames...', getattr(processor, 'NAME', 'PROCESSOR'))
        processor.process_video(source_path, temp_frame_paths)
        processor.post_process()

    # --- Reassemble video ---
    if roop.globals.keep_fps:
        fps = detect_fps(target_path)
        update_status(f'Creating video with {fps} FPS...')
        create_video(target_path, fps)
    else:
        update_status('Creating video with 30 FPS...')
        create_video(target_path)

    # --- Restore audio ---
    if roop.globals.skip_audio:
        move_temp(target_path, output_path)
        update_status('Skipping audio...')
    else:
        update_status('Restoring audio...')
        restore_audio(target_path, output_path)

    # --- Cleanup temp frames ---
    update_status('Cleaning temporary resources...')
    clean_temp(target_path)

    if not os.path.isfile(output_path):
        raise RuntimeError("Pipeline completed but no output file was produced.")

    update_status('Processing complete!')
    return output_path


# ---------------------------------------------------------------------------
# FastAPI app with lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: configure globals, create dirs, init DB, pre-load models."""
    _configure_globals()
    _ensure_dirs()
    init_db()          # <-- initialise SQLite on startup
    limit_resources()
    _preload_models()
    print("[API] Face-swap API is ready.")
    yield
    # Shutdown: nothing special needed
    print("[API] Shutting down.")


app = FastAPI(
    title="Roop Face-Swap API",
    description="Face-swap API powered by InsightFace + GFPGAN. Upload a source face image and a target video to receive the swapped result.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    """Health check — confirms the API is running."""
    return {
        "status": "healthy",
        "execution_providers": roop.globals.execution_providers,
        "frame_processors": roop.globals.frame_processors,
    }


@app.get("/request/{request_id}")
async def get_request_status(request_id: str):
    """
    Fetch the stored record for a specific request ID.

    Returns:
        - **request_id**: UUID of the request
        - **status**: `not_completed` | `in_progress` | `completed` | `failed`
        - **timestamp**: seconds taken to process (null until completed/failed)
        - **request_type**: `image` or `video`
        - **created_at**: ISO-8601 UTC time the request arrived
    """
    record = get_request(request_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Request ID '{request_id}' not found.")
    return JSONResponse(content=record)


@app.get("/requests")
async def list_requests():
    """
    List all face-swap requests ever made, newest first.
    Useful for auditing or monitoring batch jobs.
    """
    records = get_all_requests()
    return JSONResponse(content={"total": len(records), "requests": records})


@app.get("/download/{request_id}")
async def download_result(request_id: str):
    """
    Download the output file (image or video) for a completed face-swap request.

    Use the `request_id` returned by `/swap-face` or `/swap-face-image` to fetch
    the processed file. The file is served as-is from the outputs directory.
    """
    record = get_request(request_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Request ID '{request_id}' not found.")
    if record["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Request is not completed yet. Current status: '{record['status']}'"
        )

    # Find the output file — it is named {short_id}_result.*
    short_id = request_id[:8]
    matches = list(OUTPUT_DIR.glob(f"{short_id}_result.*"))
    if not matches:
        raise HTTPException(status_code=404, detail="Output file not found on server.")

    result_path = str(matches[0])
    ext = matches[0].suffix.lower()

    if ext in (".mp4", ".mov", ".avi", ".mkv"):
        media_type = "video/mp4"
        filename = f"swapped_REQ-{short_id}{ext}"
    elif ext == ".png":
        media_type = "image/png"
        filename = f"swapped_REQ-{short_id}.png"
    else:
        media_type = "image/jpeg"
        filename = f"swapped_REQ-{short_id}.jpg"

    return FileResponse(path=result_path, filename=filename, media_type=media_type)


@app.post("/swap-face-image")
async def swap_face_image(
    source_image: UploadFile = File(..., description="Source face image (jpg/png)"),
    target_image: UploadFile = File(..., description="Target image (jpg/png)"),
    frame_processors: List[str] = Query(
        default=["face_swapper", "face_enhancer"],
        description="Frame processors to apply. Options: face_swapper, face_enhancer",
    ),
    many_faces: bool = Query(default=False, description="Swap all faces in the image"),
):
    """
    Perform face swapping on an image.

    Upload a **source face image** and a **target image**.
    The face in the target image will be replaced with the face from the source image.
    Returns a JSON record with request metadata and a download URL for the result.
    """
    request_id = str(uuid.uuid4())
    short_id = request_id[:8]
    source_filename = f"{short_id}_source_{source_image.filename}"
    target_filename = f"{short_id}_target_{target_image.filename}"

    source_path = str(UPLOAD_DIR / source_filename)
    target_path = str(UPLOAD_DIR / target_filename)

    target_name, target_ext = os.path.splitext(target_image.filename)
    output_filename = f"{short_id}_result{target_ext or '.jpg'}"
    output_path = str(OUTPUT_DIR / output_filename)

    print(f"[API] swap-face-image request_id={request_id}")

    # --- DB: record arrival ---
    create_request(request_id, request_type="image")
    start_time = time.monotonic()

    try:
        with open(source_path, "wb") as f:
            shutil.copyfileobj(source_image.file, f)
        with open(target_path, "wb") as f:
            shutil.copyfileobj(target_image.file, f)

        if not is_image(source_path):
            raise HTTPException(status_code=400, detail="Source file is not a valid image. Use jpg/png.")
        if not is_image(target_path):
            raise HTTPException(status_code=400, detail="Target file is not a valid image. Use jpg/png.")

        roop.globals.frame_processors = frame_processors
        roop.globals.many_faces = many_faces
        # Safe defaults
        roop.globals.keep_fps = True
        roop.globals.skip_audio = False

        from roop.processors.frame.core import FRAME_PROCESSORS_MODULES
        import roop.processors.frame.core as proc_core
        proc_core.FRAME_PROCESSORS_MODULES = []

        from roop.face_reference import clear_face_reference
        clear_face_reference()

        # --- DB: mark in_progress before heavy work ---
        mark_in_progress(request_id)

        result_path = run_face_swap_image_pipeline(source_path, target_path, output_path)

        elapsed = time.monotonic() - start_time
        # --- DB: mark completed with elapsed time ---
        mark_completed(request_id, elapsed)

        db_record = get_request(request_id)

        # Return JSON with DB record + a download URL for the result file
        return JSONResponse(content={
            **db_record,
            "download_url": f"/download/{request_id}",
            "message": "Face swap completed. Use the download_url to fetch your result file.",
        })

    except HTTPException:
        elapsed = time.monotonic() - start_time
        mark_failed(request_id, elapsed)
        raise
    except ValueError as e:
        elapsed = time.monotonic() - start_time
        mark_failed(request_id, elapsed)
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        elapsed = time.monotonic() - start_time
        mark_failed(request_id, elapsed)
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        elapsed = time.monotonic() - start_time
        mark_failed(request_id, elapsed)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    finally:
        for path in [source_path, target_path]:
            if os.path.isfile(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


@app.post("/swap-face")
async def swap_face(
    source_image: UploadFile = File(..., description="Source face image (jpg/png)"),
    target_video: UploadFile = File(..., description="Target video (mp4, 1-2 seconds)"),
    frame_processors: List[str] = Query(
        default=["face_swapper", "face_enhancer"],
        description="Frame processors to apply. Options: face_swapper, face_enhancer",
    ),
    many_faces: bool = Query(default=False, description="Swap all faces in the video"),
    keep_fps: bool = Query(default=True, description="Keep original video FPS"),
    skip_audio: bool = Query(default=False, description="Skip audio in output"),
    output_video_quality: int = Query(default=1, ge=0, le=100, description="Output video quality (0-100)"),
):
    """
    Perform face swapping.

    Upload a **source face image** and a **target video** (1-2 seconds).
    The face in the video will be replaced with the face from the image.
    Returns a JSON record with request metadata and a download URL for the result.
    """
    # Generate unique ID for this request to avoid file collisions
    request_id = str(uuid.uuid4())
    short_id = request_id[:8]
    source_filename = f"{short_id}_source_{source_image.filename}"
    target_filename = f"{short_id}_target_{target_video.filename}"

    source_path = str(UPLOAD_DIR / source_filename)
    target_path = str(UPLOAD_DIR / target_filename)

    # Determine output filename
    target_name, target_ext = os.path.splitext(target_video.filename)
    output_filename = f"{short_id}_result{target_ext or '.mp4'}"
    output_path = str(OUTPUT_DIR / output_filename)

    print(f"[API] swap-face request_id={request_id}")

    # --- DB: record arrival ---
    create_request(request_id, request_type="video")
    start_time = time.monotonic()

    try:
        # --- Save uploaded files ---
        with open(source_path, "wb") as f:
            shutil.copyfileobj(source_image.file, f)
        with open(target_path, "wb") as f:
            shutil.copyfileobj(target_video.file, f)

        # --- Validate uploads ---
        if not is_image(source_path):
            raise HTTPException(status_code=400, detail="Source file is not a valid image. Use jpg/png.")
        if not is_video(target_path):
            raise HTTPException(status_code=400, detail="Target file is not a valid video. Use mp4.")

        # --- Set per-request options on globals ---
        roop.globals.frame_processors = frame_processors
        roop.globals.many_faces = many_faces
        roop.globals.keep_fps = keep_fps
        roop.globals.skip_audio = skip_audio
        roop.globals.output_video_quality = output_video_quality

        # Reset processor module cache so new processor list takes effect
        from roop.processors.frame.core import FRAME_PROCESSORS_MODULES
        import roop.processors.frame.core as proc_core
        proc_core.FRAME_PROCESSORS_MODULES = []

        # Reset face reference for fresh processing
        from roop.face_reference import clear_face_reference
        clear_face_reference()

        # --- DB: mark in_progress before heavy work ---
        mark_in_progress(request_id)

        # --- Run the pipeline ---
        result_path = run_face_swap_pipeline(source_path, target_path, output_path)

        elapsed = time.monotonic() - start_time
        # --- DB: mark completed with elapsed time ---
        mark_completed(request_id, elapsed)

        db_record = get_request(request_id)

        # Return JSON with DB record + a download URL for the result file
        return JSONResponse(content={
            **db_record,
            "download_url": f"/download/{request_id}",
            "message": "Face swap completed. Use the download_url to fetch your result file.",
        })

    except HTTPException:
        elapsed = time.monotonic() - start_time
        mark_failed(request_id, elapsed)
        raise  # re-raise FastAPI exceptions as-is
    except ValueError as e:
        elapsed = time.monotonic() - start_time
        mark_failed(request_id, elapsed)
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        elapsed = time.monotonic() - start_time
        mark_failed(request_id, elapsed)
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        elapsed = time.monotonic() - start_time
        mark_failed(request_id, elapsed)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    finally:
        # Cleanup uploaded files (output is kept until served)
        for path in [source_path, target_path]:
            if os.path.isfile(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# Direct run (for local dev — in Colab use: !uvicorn api:app ...)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
