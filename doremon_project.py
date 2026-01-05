import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(page_title="Doraemon RGB Visualization", layout="wide")

st.title("🎨 Doraemon Image – RGB Channel Visualization")

# Load image from local file
Doraemon_file = r"C:\Users\ANITHA\Downloads\doraemon.png"

try:
    Doraemon = Image.open(Doraemon_file).convert("RGB")
except Exception as e:
    st.error(f"Error loading image: {e}")
    st.stop()

Doraemon_np = np.array(Doraemon)

# Split RGB channels
R, G, B = Doraemon_np[:, :, 0], Doraemon_np[:, :, 1], Doraemon_np[:, :, 2]

# Create images emphasizing each channel
red_img = np.zeros_like(Doraemon_np)
green_img = np.zeros_like(Doraemon_np)
blue_img = np.zeros_like(Doraemon_np)

red_img[:, :, 0] = R
green_img[:, :, 1] = G
blue_img[:, :, 2] = B

# ---- Display Images ----
st.subheader("Original and RGB Channel Images")

col1, col2 = st.columns(2)

with col1:
    st.image(Doraemon_np, caption="Original Image", use_container_width=True)
    st.image(red_img, caption="Red Channel Emphasis", use_container_width=True)

with col2:
    st.image(green_img, caption="Green Channel Emphasis", use_container_width=True)
    st.image(blue_img, caption="Blue Channel Emphasis", use_container_width=True)

# ---- Grayscale with Colormap ----
st.subheader("🎭 Grayscale with Colormap")

Doraemon_gray = Doraemon.convert("L")
Doraemon_gray_np = np.array(Doraemon_gray)

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(Doraemon_gray_np, cmap="viridis")
ax.set_title("Colormapped Grayscale")
ax.axis("off")
plt.colorbar(im, ax=ax)

st.pyplot(fig)
