import os
os.environ["STREAMLIT_SERVER_ENABLECORS"] = "false"
os.environ["STREAMLIT_SERVER_ENABLEWEBSOCKETCOMPRESSION"] = "false"

import streamlit as st
import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis
import tempfile

st.set_page_config(page_title="Face Swapper", layout="centered")

# ------------------------------
# Sidebar options
# ------------------------------
st.sidebar.title("⚙️ Settings")

# Device
device_option = st.sidebar.radio("Choose Device", ["CPU", "GPU"], index=0)
ctx_id = 0 if device_option == "GPU" else -1

# Processing resolution (downscale for speed)
proc_res = st.sidebar.selectbox(
    "Processing Resolution",
    ["Original", "720p", "480p"],
    index=1,  # default 720p for T4 speed
    help="Frames are resized for detection/swap. Lower = faster."
)

# FPS cap (drop frames for speed)
fps_cap = st.sidebar.selectbox(
    "Target FPS",
    ["Original", "24", "15"],
    index=0,
    help="Lower FPS = fewer frames processed = faster."
)

# Option: keep original output resolution
keep_original_res = st.sidebar.checkbox(
    "Keep original output resolution", value=False,
    help="If enabled, we upscale processed frames back to original size."
)

# ------------------------------
# Load models
# ------------------------------
@st.cache_resource
def load_models(ctx_id):
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))

    swapper = insightface.model_zoo.get_model(
        'inswapper_128.onnx', download=False, download_zip=False
    )
    return app, swapper

app, swapper = load_models(ctx_id)

# ------------------------------
# Helpers
# ------------------------------
def _target_size_for_height(width, height, target_h):
    """Keep aspect ratio; compute new (w,h) for a given target height."""
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
    return orig_w, orig_h  # Original

def _parse_fps_cap(original_fps, cap_choice):
    if not original_fps or original_fps <= 0:
        original_fps = 25.0
    if cap_choice == "Original":
        return original_fps, 1  # write_fps, frame_step
    try:
        tgt = float(cap_choice)
        if tgt <= 0:
            return original_fps, 1
        # frame_step = how many frames to skip while reading
        step = max(1, int(round(original_fps / tgt)))
        write_fps = original_fps / step
        return write_fps, step
    except:
        return original_fps, 1

# ------------------------------
# Face swapping function
# ------------------------------
def swap_faces_in_video(image, video_path, progress):
    # Detect source face(s)
    source_faces = app.get(image)
    if not source_faces:
        st.error("❌ No face detected in the source image.")
        return None
    # Use largest source face if multiple
    source_face = max(source_faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))

    # Prepare output file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_out:
        output_path = tmp_out.name

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("❌ Could not open video file. Try re-encoding to MP4/H.264.")
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps   = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0

    # Decide processing size and fps behavior
    proc_w, proc_h = _get_proc_size_choice(orig_w, orig_h, proc_res)
    write_fps, frame_step = _parse_fps_cap(orig_fps, fps_cap)

    # Choose output size
    out_w, out_h = (orig_w, orig_h) if keep_original_res else (proc_w, proc_h)

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, write_fps, (out_w, out_h))
    if not out.isOpened():
        cap.release()
        st.error("❌ Failed to open VideoWriter. Try a different resolution/fps.")
        return None

    # Status
    st.info(
        f"Input: {orig_w}x{orig_h} @ {orig_fps:.2f} fps | "
        f"Processing: {proc_w}x{proc_h} | Writing: {out_w}x{out_h} @ {write_fps:.2f} fps | "
        f"Frame step: {frame_step} (1=every frame)"
    )

    i = 0     # read frame index
    wcount = 0  # written frames counter (for progress if frame_count==0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Skip frames for FPS cap
        if frame_step > 1 and (i % frame_step != 0):
            i += 1
            # Still update progress if total frame count is known
            if frame_count > 0:
                progress.progress(min(1.0, i / frame_count))
            continue

        # Resize for processing if needed
        proc_frame = frame
        if (proc_w, proc_h) != (orig_w, orig_h):
            proc_frame = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)

        try:
            # Detect faces on processed frame
            target_faces = app.get(proc_frame)
            result_frame = proc_frame.copy()

            # Swap each detected face
            for target_face in target_faces:
                try:
                    result_frame = swapper.get(
                        proc_frame, target_face, source_face, paste_back=True
                    )
                except Exception:
                    # fallback to paste on intermediate buffer
                    result_frame = swapper.get(
                        result_frame, target_face, source_face, paste_back=True
                    )

            # Upscale back to original if requested
            if keep_original_res and (proc_w, proc_h) != (orig_w, orig_h):
                result_frame = cv2.resize(result_frame, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
                out.write(result_frame)
            else:
                out.write(result_frame)

        except Exception as e:
            # Log and fall back: write original frame (resized to output size if needed)
            print(f"⚠️ Frame {i} skipped due to error: {e}")
            fallback = proc_frame
            if keep_original_res and (proc_w, proc_h) != (orig_w, orig_h):
                fallback = cv2.resize(proc_frame, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
            out.write(fallback)

        i += 1
        wcount += 1
        # Progress
        if frame_count > 0:
            progress.progress(min(1.0, i / max(1, frame_count)))
        else:
            # crude fallback when frame_count is unknown
            if wcount % 30 == 0:
                progress.progress(min(1.0, (wcount % 300) / 300.0))

    cap.release()
    out.release()
    return output_path

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🎭 Face Swapper in Video")
st.write("Upload a **source image** and a **target video**, preview them, set speed options, then swap faces.")

# Upload files
image_file = st.file_uploader("Upload Source Image", type=["jpg", "jpeg", "png"])
video_file = st.file_uploader("Upload Target Video", type=["mp4", "avi", "mov", "mkv"])

# Preview uploaded files
if image_file:
    st.subheader("📷 Source Image Preview")
    st.image(image_file, caption="Source Image", use_column_width=True)

if video_file:
    st.subheader("🎬 Target Video Preview")
    st.video(video_file)

# Button to process
if st.button("🚀 Start Face Swap"):
    if image_file and video_file:
        # Use .getvalue() (safe for preview + processing)
        image_bytes = image_file.getvalue()
        source_image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

        video_bytes = video_file.getvalue()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
            tmp_video.write(video_bytes)
            tmp_video_path = tmp_video.name

        with st.spinner("Processing video... This may take a while ⏳"):
            progress_bar = st.progress(0)
            output_video_path = swap_faces_in_video(source_image, tmp_video_path, progress_bar)

        if output_video_path:
            st.success("✅ Face swapping completed!")
            st.subheader("📺 Output Video Preview")
            st.video(output_video_path)

            with open(output_video_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download Processed Video",
                    data=f,
                    file_name="output_swapped_video.mp4",
                    mime="video/mp4"
                )

            # Cleanup temp input (keep output so it can be downloaded)
            try:
                os.remove(tmp_video_path)
            except Exception:
                pass
    else:
        st.error("⚠️ Please upload both a source image and a video.")
