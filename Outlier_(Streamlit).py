import streamlit as st
import numpy as np
from PIL import Image
import cv2
import os
import random

from outlier_detection import check_image_is_outlier
from blur_function import anonymize_center

st.set_page_config(layout="centered")

st.title("Outlier detection and Blur")

if "anomaly_image" not in st.session_state:
    st.session_state.anomaly_image = None

# ----- RANDOM IMAGE LOADER -----

if st.button("Load random image"):
    folder = "image_data/image_data"
    files = [f for f in os.listdir(folder) if f.lower().endswith(".png")]
    if files:
        choice = random.choice(files)
        path = os.path.join(folder, choice)
        pil = Image.open(path).convert("RGB")
        st.session_state.random_image = pil
    else:
        st.session_state.random_image = None

# Determine active image
raw_image = st.file_uploader("Upload anomaly image", type=["png"])
st.caption("(Must be .png)")
st.warning('Blurring will only happen when anomaly is detected')
active_image = None

if raw_image is not None:
    active_image = Image.open(raw_image).convert("RGB")
elif "random_image" in st.session_state and st.session_state.random_image is not None:
    active_image = st.session_state.random_image

if active_image is not None:
    st.image(active_image, caption="Active image")

    if st.button("Detect anomaly"):
        np_image = np.array(active_image)
        cv_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)

        outlier, mse = check_image_is_outlier(
            np_image,
            "model_path/autoencoder_200.keras",
            "model_path/threshold.txt"
        )

        with open("model_path/threshold.txt", "r") as f:
            threshold = float(f.read().strip())

        st.write(f"Outlier: {outlier}")
        st.write(f"MSE: {mse}")
        st.write(f"Threshold: {threshold}")

        if outlier:
            st.session_state.anomaly_image = cv_image

if st.session_state.anomaly_image is not None:
    blur_factor = st.slider("Blur factor", min_value=1, max_value=51, value=25, step=2)
    region_ratio = st.slider("Center region size", min_value=0.1, max_value=1.0, value=0.8, step=0.05)

    blurred = anonymize_center(
        st.session_state.anomaly_image,
        region_ratio=region_ratio,
        factor=blur_factor
    )

    blurred_rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
    final_pil = Image.fromarray(blurred_rgb)

    st.image(final_pil, caption="Blurred anomaly")

    st.download_button(
        label="Download blurred image",
        data=final_pil.tobytes(),
        file_name="blurred.png",
        mime="image/png"
    )
