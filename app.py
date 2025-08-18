import streamlit as st
import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis
import tempfile
import os

st.set_page_config(page_title="Face Swapper", layout="centered")

# ------------------------------
# Sidebar options
# ------------------------------
st.sidebar.title("⚙️ Settings")

# CPU / GPU selection
device_option = st.sidebar.radio("Choose Device", ["CPU", "GPU"], index=0)
ctx_id = 0 if device_option == "GPU" else -1

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
# Face swapping function
# ------------------------------
def swap_faces_in_video(image, video, progress):
    source_faces = app.get(image)
    if len(source_faces) == 0:
        st.error("❌ No face detected in the source image.")
        return None

    source_face = source_faces[0]

    # Temporary output file (MP4 for better compatibility)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_out:
        output_path = tmp_out.name

    cap = cv2.VideoCapture(video)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # MP4 writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        target_faces = app.get(frame)
        result_frame = frame.copy()

        for target_face in target_faces:
            # depending on insightface version, frame or result_frame may be needed
            try:
                result_frame = swapper.get(frame, target_face, source_face, paste_back=True)
            except:
                result_frame = swapper.get(result_frame, target_face, source_face, paste_back=True)

        out.write(result_frame)

        i += 1
        if frame_count > 0:
            progress.progress(min(1.0, i / frame_count))

    cap.release()
    out.release()
    return output_path

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🎭 Face Swapper in Video")
st.write("Upload a **source image** and a **target video**, preview them, then swap faces.")

# Upload files
image_file = st.file_uploader("Upload Source Image", type=["jpg", "jpeg", "png"])
video_file = st.file_uploader("Upload Target Video", type=["mp4", "avi"])

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
        source_image = cv2.imdecode(
            np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR
        )

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
            tmp_video.write(video_file.read())
            tmp_video_path = tmp_video.name

        with st.spinner("Processing video... Please wait ⏳"):
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

            # Cleanup
            os.remove(tmp_video_path)
            # Optionally delete output file after preview
            # os.remove(output_video_path)
    else:
        st.error("⚠️ Please upload both a source image and a video.")
