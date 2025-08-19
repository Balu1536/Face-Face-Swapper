# =========================
# app.py  (production-ready)
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

# Lazy imports for heavy libs inside cached loaders to avoid early session init issues
def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

# -----------------------------------
# Page & Sidebar (controls for speed)
# -----------------------------------
st.set_page_config(page_title="Face Swapper", layout="centered")
st.title("🎭 Face Swapper in Video")

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
    """
    # Defer heavy imports until Streamlit session is ready
    import insightface
    from insightface.app import FaceAnalysis

    # Providers for ONNX Runtime (insightface uses ORT under the hood)
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if _has_cuda() else ["CPUExecutionProvider"]

    # Face detector/landmarks (retinaface + arcface in buffalo_l)
    app = FaceAnalysis(name="buffalo_l")
    # ctx_id: 0 (GPU) or -1 (CPU)
    ctx_id = 0 if _has_cuda() else -1
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))

    # Face swapper (inswapper_128)
    # Let insightface download the model if not present
    swapper = insightface.model_zoo.get_model(
        "inswapper_128.onnx",
        download=True,
        download_zip=False,
        providers=providers
    )

    return app, swapper, providers, ctx_id

# Initialize models
with st.spinner("Loading models…"):
    app, swapper, providers, ctx_id = load_models()

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
        return original_fps, 1  # write_fps, frame_step
    try:
        tgt = float(cap_choice)
        step = max(1, int(round(original_fps / tgt)))
        write_fps = original_fps / step
        return write_fps, step
    except Exception:
        return original_fps, 1

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
    source_faces = app.get(image_bgr)
    if not source_faces:
        st.error("❌ No face detected in the source image.")
        return None

    # Use the largest detected face if there are multiple
    source_face = max(
        source_faces,
        key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1])
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
    orig_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0

    # Decide processing size & FPS behavior
    proc_w, proc_h = _get_proc_size_choice(orig_w, orig_h, proc_res)
    write_fps, frame_step = _parse_fps_cap(orig_fps, fps_cap)
    out_w, out_h = (orig_w, orig_h) if keep_original_res else (proc_w, proc_h)

    # Prepare output writer
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_out:
        output_path = tmp_out.name

    # `mp4v` is widely compatible on Spaces/desktop browsers
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, write_fps, (out_w, out_h))
    if not out.isOpened():
        cap.release()
        st.error("❌ Failed to open VideoWriter. Try a different resolution/FPS setting.")
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
            if frame_step > 1 and (read_idx % frame_step != 0):
                read_idx += 1
                if frame_count > 0:
                    progress.progress(min(1.0, read_idx / frame_count))
                continue

            # Resize for processing
            if (proc_w, proc_h) != (orig_w, orig_h):
                proc_frame = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
            else:
                proc_frame = frame

            try:
                # Detect faces on processed frame
                target_faces = app.get(proc_frame)

                if target_faces:
                    # Optionally limit faces to largest N for speed
                    target_faces = sorted(
                        target_faces,
                        key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]),
                        reverse=True
                    )[:max_faces]

                # Swap into a working buffer
                result_frame = proc_frame.copy()
                for tface in target_faces:
                    # Two-call fallback for different insightface versions
                    try:
                        result_frame = swapper.get(
                            proc_frame, tface, source_face, paste_back=True
                        )
                    except Exception:
                        result_frame = swapper.get(
                            result_frame, tface, source_face, paste_back=True
                        )

                # Upscale back to original if requested
                if keep_original_res and (proc_w, proc_h) != (orig_w, orig_h):
                    result_frame = cv2.resize(result_frame, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

                out.write(result_frame)

            except Exception as e:
                # Log & write fallback frame (processed size or original size)
                print(f"[WARN] Frame {read_idx} failed: {e}")
                traceback.print_exc()
                fallback = proc_frame
                if keep_original_res and (proc_w, proc_h) != (orig_w, orig_h):
                    fallback = cv2.resize(proc_frame, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
                out.write(fallback)

            read_idx += 1
            processed_frames += 1

            # Update progress
            if frame_count > 0:
                progress.progress(min(1.0, read_idx / frame_count))
            elif processed_frames % 30 == 0:
                # Fallback progress for unknown frame counts
                progress.progress(min(1.0, (processed_frames % 300) / 300.0))

    finally:
        cap.release()
        out.release()

    return output_path

# -------------------------
# UI: Uploads & Preview
# -------------------------
st.write("Upload a **source face image** and a **target video**, preview them, tweak speed options, then start swapping.")

image_file = st.file_uploader("Upload Source Image", type=["jpg", "jpeg", "png"])
video_file = st.file_uploader("Upload Target Video", type=["mp4", "mov", "mkv", "avi"])

# Previews
if image_file:
    st.subheader("📷 Source Image Preview")
    st.image(image_file, caption="Source Image", use_column_width=True)

if video_file:
    st.subheader("🎬 Target Video Preview")
    st.video(video_file)

# -------------------------
# Run button
# -------------------------
if st.button("🚀 Start Face Swap"):
    if not image_file or not video_file:
        st.error("⚠️ Please upload both a source image and a target video.")
    else:
        # Read uploads safely (do not consume file pointer used by preview)
        try:
            image_bytes = image_file.getvalue()
            source_image = _safe_imdecode(image_bytes)
            if source_image is None:
                st.error("❌ Failed to decode source image. Please use a valid JPG/PNG.")
                st.stop()
        except Exception:
            st.error("❌ Failed to read the source image bytes.")
            st.stop()

        try:
            video_bytes = video_file.getvalue()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
                tmp_video.write(video_bytes)
                tmp_video_path = tmp_video.name
        except Exception:
            st.error("❌ Failed to save the uploaded video to a temp file.")
            st.stop()

        with st.spinner("Processing video… This can take a while ⏳"):
            progress_bar = st.progress(0)
            output_video_path = swap_faces_in_video(
                source_image,
                tmp_video_path,
                proc_res=proc_res,
                fps_cap=fps_cap,
                keep_original_res=keep_original_res,
                max_faces=max_faces,
                progress=progress_bar
            )

        if output_video_path:
            st.success("✅ Face swapping completed!")

            st.subheader("📺 Output Video Preview")
            st.video(output_video_path)

            # Download button
            try:
                with open(output_video_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Processed Video",
                        data=f,
                        file_name="output_swapped_video.mp4",
                        mime="video/mp4"
                    )
            except Exception:
                st.warning("⚠️ Could not open the output file for download.")

        # Cleanup temp input video; keep output so it can be downloaded
        try:
            os.remove(tmp_video_path)
        except Exception:
            pass

# -------------
# Diagnostics
# -------------
with st.expander("🩺 Diagnostics"):
    st.write(
        "- If you see **SessionInfo** errors: this app defers heavy imports via `@st.cache_resource` "
        "so Streamlit initializes first. If errors persist, restart the runtime.\n"
        "- If output is jumpy/stutters: lower **Target FPS** or choose **480p** processing.\n"
        "- If video fails to open: re-encode your input to **MP4 (H.264, AAC)**."
    )