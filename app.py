import streamlit as st
import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis
import tempfile
import os

# Initialize face analysis and load model
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

# Load the face swapper model
swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=False, download_zip=False)

def swap_faces_in_video(image, video):
    """
    Swaps faces from a source image with faces detected in a video and returns the path to the output video file.
    
    image: Source image (as an array)
    video: Path to the input video file
    """
    source_faces = app.get(image)

    if len(source_faces) == 0:
        st.error("No face detected in the source image.")
        return None

    source_face = source_faces[0]

    # Create a temporary file to save the output video
    output_path = tempfile.mktemp(suffix='.avi')

    # Open the video file
    cap = cv2.VideoCapture(video)

    # Get video properties for output
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Exit if the video is finished

        # Detect faces in the current frame
        target_faces = app.get(frame)

        # Create a copy of the frame for the result
        result_frame = frame.copy()

        # Swap faces for each detected face in the video frame
        for target_face in target_faces:
            result_frame = swapper.get(result_frame, target_face, source_face, paste_back=True)

        # Write the result frame to the output video
        out.write(result_frame)

    # Release resources
    cap.release()
    out.release()
    
    return output_path

# Streamlit UI
st.title("Face Swapper in Video")
st.write("Upload an image and a video to swap faces.")

# File uploader for the source image
image_file = st.file_uploader("Upload Source Image", type=["jpg", "jpeg", "png"])

# File uploader for the video
video_file = st.file_uploader("Upload Video", type=["mp4", "avi"])

if st.button("Swap Faces"):
    if image_file is not None and video_file is not None:
        # Read the source image
        source_image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Save the uploaded video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
            tmp_video.write(video_file.read())
            tmp_video_path = tmp_video.name

        # Show a spinner while processing
        with st.spinner("Processing video..."):
            output_video_path = swap_faces_in_video(source_image, tmp_video_path)

        if output_video_path:
            st.success("Face swapping completed!")
            st.video(output_video_path)
            os.remove(tmp_video_path)  # Clean up temporary video file
            os.remove(output_video_path)  # Clean up output video file if needed
    else:
        st.error("Please upload both an image and a video.")
