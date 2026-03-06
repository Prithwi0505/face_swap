#!/usr/bin/env python3
"""
FastAPI backend for the roop face-swapping pipeline.
Designed for Google Colab: loads models once, processes requests in-process.

Usage (Colab):
    !uvicorn api:app --host 0.0.0.0 --port 7860
"""

import os
import sys
import uuid
import shutil
import warnings
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List
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
from roop.predictor import predict_video
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
# Directories
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
    """Startup: configure globals, create dirs, pre-load models."""
    _configure_globals()
    _ensure_dirs()
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
    Returns the processed video file.
    """
    # Generate unique ID for this request to avoid file collisions
    request_id = str(uuid.uuid4())[:8]
    source_filename = f"{request_id}_source_{source_image.filename}"
    target_filename = f"{request_id}_target_{target_video.filename}"

    source_path = str(UPLOAD_DIR / source_filename)
    target_path = str(UPLOAD_DIR / target_filename)

    # Determine output filename
    target_name, target_ext = os.path.splitext(target_video.filename)
    output_filename = f"{request_id}_result{target_ext or '.mp4'}"
    output_path = str(OUTPUT_DIR / output_filename)

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

        # --- Run the pipeline ---
        result_path = run_face_swap_pipeline(source_path, target_path, output_path)

        # --- Return the result video ---
        return FileResponse(
            path=result_path,
            filename=f"swapped_{target_video.filename}",
            media_type="video/mp4",
        )

    except HTTPException:
        raise  # re-raise FastAPI exceptions as-is
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
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
