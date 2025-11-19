import streamlit as st
import matplotlib.pyplot as plt
from PIL.ImageOps import grayscale
from denoising_autoencoder import *

st.set_page_config(layout="centered")
st.title("De-noising with autoencoder")

if "raw_image" not in st.session_state:
    st.session_state.raw_image = None

if "noise_image" not in st.session_state:
    st.session_state.noise_image = None

if st.button("Load random image"):
    st.session_state.raw_image = load_random_image_from_folder('undistorted_images')
    st.session_state.noise_image = None

if st.session_state.raw_image is not None:
    st.image(st.session_state.raw_image, caption="Original")

    if st.button("Apply noise on image"):
        st.session_state.noise_image = add_noise(st.session_state.raw_image)

if st.session_state.noise_image is not None:
    st.image(st.session_state.noise_image, caption="Noisy")

    if st.button("De-noise image with autoencoder model"):
        denoised = denoise_single_image(
            'model_path/denoising_autoencoder.keras',
            st.session_state.noise_image
        )
        st.image(denoised, caption="Denoised")
