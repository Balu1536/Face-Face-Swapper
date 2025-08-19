# =========================
# app.py  (production-ready, safer)
# =========================
import os

# Streamlit server tweaks (safe on HF Spaces / containers)
os.environ["STREAMLIT_SERVER_ENABLECORS"] = "false"
os.environ["STREAMLIT_SERVER_ENABLEWEBSOCKETCOMPRESSION"] = "false"

import streamlit as st
import numpy as np
import cv2
import tempfile
import traceback

# -------------------------
# VERY EARLY: initialize session state
# -------------------------
# This prevents the "SessionInfo before it was initialized" glitch on some boots
for key, default in {
    "uploaded_image": None,
    "uploaded_video": None,
    "output_video": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# -------------------------
# GPU check (optional torch import)
# -------------------------
def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        # If torch isn't installed, just say no CUDA
        return False

# -----------------------------------
# Page & Sidebar (controls for speed)
# -----------------------------------
st.set_page_config(page_title="Face Swapper", layout="centered")
st.title("🎭 Savvy Long Video Swapper")

st.sidebar.title("⚙️ Settings")

# Downscale to speed up detection & swapping
proc_res = st.sidebar.selectbox(
    "Processing Resolution",
    ["Original", "720p", "480p"],
    index=1,
    help="Frames are resized before detection/swap. Lower = faster."
)

# Skip frames to hit a lower effective FPS
fps_cap = st.sidebar.selectbox(
    "Target FPS",
    ["Original", "24", "15"],
    index=0,
    help="Lower target FPS drops frames during processing for speed."
)

# Keep the original output resolution even if we process smaller
keep_original_res = st.sidebar.checkbox(
    "Keep original output resolution",
    value=False,
    help="If enabled, processed frames are upscaled back to the input size."
)

# Limit faces per frame (helps speed on crowded scenes)
max_faces = st.sidebar.slider(
    "Max faces per frame", min_value=1, max_value=8, value=4,
    help="At most this many faces will be swapped per frame."
)

# -------------------------
# Model loading (cached)
# -------------------------
@st.cache_resource(show_spinner=True)
def load_models():
    """
    Load InsightFace detectors and the inswapper model once.
    Auto-select GPU if available, else CPU.
    Be tolerant of insightface versions (providers kwarg may not exist).
    """
    import insightface
    from insightface.app import FaceAnalysis

    # Desired providers for ORT
    wants_cuda = _has_cuda()
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if wants_cuda else ["CPUExecutionProvider"]

    # Face detector/landmarks (retinaface + arcface in buffalo_l)
    ctx_id = 0 if wants_cuda else -1
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))

    # Face swapper (inswapper_128)
    # Some insightface versions accept providers=..., some don't.
    swapper = None
    try:
        swapper = insightface.model_zoo.get_model(
            "inswapper_128.onnx",
            download=True,
            download_zip=False,
            providers=providers
        )
    except TypeError:
        # Fallback path: older insightface without providers kwarg
        swapper = insightface.model_zoo.get_model(
            "inswapper_128.onnx",
            download=True,
            download_zip=False
        )
    except Exception as e:
        # Last resort: surface a helpful error
        raise RuntimeError(f"Failed to load inswapper_128.onnx: {e}")

    return app, swapper, providers, ctx_id

# Initialize models
with st.spinner("Loading models…"):
    try:
        app, swapper, providers, ctx_id = load_models()
    except Exception as e:
        st.error("❌ Model loading failed. See logs for details.")
        raise

st.caption(
    f"Device: {'GPU (CUDA)' if ctx_id == 0 else 'CPU'} • ORT Providers: {', '.join(providers)}"
)

# -------------------------
# Helpers
# -------------------------
def _target_size_for_height(width, height, target_h):
    if target_h <= 0 or height == 0:
        return width, height
    scale = target_h / float(height)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    return new_w, new_h

def _get_proc_size_choice(orig_w, orig_h, choice):
    if choice == "720p":
        return _target_size_for_height(orig_w, orig_h, 720)
    if choice == "480p":
        return _target_size_for_height(orig_w, orig_h, 480)
    return orig_w, orig_h

def _parse_fps_cap(original_fps, cap_choice):
    # Handle bad/zero FPS from container decoders
    if not original_fps or original_fps <= 0:
        original_fps = 25.0
    if cap_choice == "Original":
        return max(1.0, original_fps), 1  # write_fps, frame_step
    try:
        tgt = float(cap_choice)
        tgt = max(1.0, tgt)
        step = max(1, int(round(original_fps / tgt)))
        write_fps = max(1.0, original_fps / step)
        return write_fps, step
    except Exception:
        return max(1.0, original_fps), 1

def _safe_imdecode(file_bytes):
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

# -------------------------------------
# Core: face swap over an input video
# -------------------------------------
def swap_faces_in_video(
    image_bgr: np.ndarray,
    video_path: str,
    proc_res: str,
    fps_cap: str,
    keep_original_res: bool,
    max_faces: int,
    progress
):
    # Validate source image
    try:
        source_faces = app.get(image_bgr)
    except Exception as e:
        st.error(f"❌ FaceAnalysis failed on source image: {e}")
        return None

    if not source_faces:
        st.error("❌ No face detected in the source image.")
        return None

    # Use the largest detected face if there are multiple
    source_face = max(
        source_faces,
        key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[1]-f.bbox[3])  # absolute area doesn't depend on sign but keep positive
        if hasattr(f, "bbox") else 0
    )
    # (safer area) re-compute properly
    source_face = max(
        source_faces,
        key=lambda f: max(1, int((f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1])))
    )

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("❌ Could not open the uploaded video. Try re-encoding to MP4/H.264.")
        return None

    # Read properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if orig_fps <= 0 or np.isnan(orig_fps):
        orig_fps = 25.0

    # Decide processing size & FPS behavior
    proc_w, proc_h = _get_proc_size_choice(orig_w, orig_h, proc_res)
    write_fps, frame_step = _parse_fps_cap(orig_fps, fps_cap)
    out_w, out_h = (orig_w, orig_h) if keep_original_res else (proc_w, proc_h)

    # Prepare output writer
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_out:
        output_path = tmp_out.name

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, write_fps, (out_w, out_h))
    if not out.isOpened():
        cap.release()
        st.error(
            "❌ Failed to open VideoWriter. "
            "Try setting Processing Resolution to 480p or Target FPS to 24."
        )
        return None

    st.info(
        f"Input: {orig_w}×{orig_h} @ {orig_fps:.2f} fps | "
        f"Processing: {proc_w}×{proc_h} | Writing: {out_w}×{out_h} @ {write_fps:.2f} fps | "
        f"Frame step: {frame_step} (1 = process every frame) | "
        f"Max faces/frame: {max_faces}"
    )

    # Process loop
    read_idx = 0
    processed_frames = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # FPS cap by skipping frames
            if frame_step > 1 and (re_
