import streamlit as st
from deepface import DeepFace
import cv2
import numpy as np
from PIL import Image

# Streamlit UI
st.title("😃 Emotion Detection App")
st.write("Upload an image to detect the emotion.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load Image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to OpenCV format
    img_cv = np.array(image)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    try:
        # Perform emotion analysis
        st.write("🔍 *Analyzing emotions...*")
        result = DeepFace.analyze(img_path=img_cv, actions=['emotion'], detector_backend='retinaface', enforce_detection=False)

        # Get dominant emotion
        dominant_emotion = result[0]['dominant_emotion']
        emotion_scores = result[0]['emotion']

        # Display result
        st.success(f"🎭 Dominant Emotion: *{dominant_emotion.capitalize()}*")
        
        # Show all emotion scores
        st.write("📊 *Emotion Scores:*")
        for emotion, score in emotion_scores.items():
            st.write(f"{emotion.capitalize()}: {score:.2f}%")

    except Exception as e:
        st.error("😔 Could not detect a face. Try another image.")
        st.write(f"Error: {str(e)}")