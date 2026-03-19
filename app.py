"""
Retinal Vessel Segmentation - My U-Net Streamlit App
====================================================

This is my personal Streamlit app for testing retinal vessel segmentation with U-Net.
I built it to match the exact visualization style from test.py - three aligned views side by side.

What it does:
- Upload fundus images (PNG/JPG)
- Runs U-Net inference instantly after first model load
- Shows original, prediction, and overlay - all perfectly aligned

Files you need in the same folder:
- files/unet_checkpoint.pth (my trained model)
- utils.py (has denormalize and load_model_checkpoint functions I use)

To run: streamlit run app.py

What I need installed:
streamlit>=1.28.0, torch>=2.0.0, numpy, pillow, matplotlib
"""

import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from utils import denormalize, load_model_checkpoint

@st.cache_resource
def load_model():
    """
    Load U-Net model checkpoint with device detection.

    Returns trained model in evaluation mode on optimal device (CUDA/CPU).
    Model cached for instant reuse after first load.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model_checkpoint("files/unet_checkpoint.pth", device)
    model.eval()
    return model, device

def preprocess_image(image):
    """
    Preprocess input image for U-Net inference.

    Steps:
    1. Resize to 512x512 (model input size)
    2. Normalize to [0,1] range
    3. Convert HWC -> CHW format
    4. Add batch dimension
    """
    image = image.resize((512, 512))
    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))  # HWC -> CHW
    img_array = torch.from_numpy(img_array).unsqueeze(0)  # Add batch dim
    return img_array

def create_aligned_plots(image_np, pred_np):
    """
    Create three perfectly aligned plots matching test.py visualization style.

    Returns three matplotlib figures with identical dimensions:
    1. Original Image - RGB fundus photograph
    2. Prediction - Grayscale vessel probability map
    3. Image+Prediction - Original + jet colormap overlay (alpha=0.5)

    All figures: 6x6 inches, 80 DPI for perfect column alignment.
    """
    fig_height = 6
    fig_width = 6
    dpi = 80

    # 1. Original Image (exact test.py style)
    fig1, ax1 = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    ax1.imshow(image_np)
    ax1.set_title('Original Image', fontsize=18, pad=10, fontweight='bold')
    ax1.axis('off')

    # 2. Prediction (grayscale, exact test.py style)
    fig2, ax2 = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    ax2.imshow(pred_np, cmap='gray')
    ax2.set_title('Prediction', fontsize=18, pad=10, fontweight='bold')
    ax2.axis('off')

    # 3. Image+Prediction (test.py jet overlay, exact style)
    fig3, ax3 = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    ax3.imshow(image_np)
    ax3.imshow(pred_np, cmap='jet', alpha=0.5)  # test.py original!
    ax3.set_title('Image+Prediction', fontsize=18, pad=10, fontweight='bold')
    ax3.axis('off')

    return fig1, fig2, fig3

# === STREAMLIT APP CONFIGURATION ===
st.set_page_config(
    page_title="Retinal Vessel Segmentation",
    page_icon=None,
    layout="wide"
)

# === APPLICATION HEADER ===
st.title("U-Net Retinal Vessel Segmentation")
st.markdown("Upload a retinal fundus image to visualize vessel segmentation predictions")

# === SIDEBAR: IMAGE UPLOAD ===
st.sidebar.title("Image Upload")
st.sidebar.markdown("Supported formats: PNG, JPG, JPEG")
uploaded_file = st.sidebar.file_uploader("Choose retinal image", type=['png', 'jpg', 'jpeg'])

# === MAIN PROCESSING LOGIC ===
if uploaded_file is not None:
    input_image = Image.open(uploaded_file)

    # Show processing status indicator
    with st.spinner("Processing retinal image and running vessel prediction..."):
        # Load model and preprocess
        model, device = load_model()
        processed_image = preprocess_image(input_image).to(device)

        # Run inference with sigmoid activation (test.py style)
        with torch.no_grad():
            pred = torch.sigmoid(model(processed_image))

        # Denormalize for visualization (test.py style)
        image_np = denormalize(processed_image[0].cpu())
        pred_np = denormalize(pred[0].cpu())

    # === THREE-COLUMN PERFECTLY ALIGNED DISPLAY ===
    figs = create_aligned_plots(image_np, pred_np)
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        #st.subheader("Original Image")
        fig1, ax1 = plt.subplots(figsize=(6, 6), dpi=80)
        ax1.imshow(np.array(input_image.resize((512, 512))))
        ax1.set_title('Original Image', fontsize=18, pad=10, fontweight='bold')
        ax1.axis('off')
        st.pyplot(fig1)
        plt.close(fig1)
        plt.close(figs[0])

    with col2:
        #st.subheader("Vessel Prediction")
        st.pyplot(figs[1])
        plt.close(figs[1])

    with col3:
        #st.subheader("Image + Prediction")
        st.pyplot(figs[2])
        plt.close(figs[2])

    # === INTERPRETATION GUIDE ===
    st.markdown("---")
    with st.expander("How to Interpret Results"):
        st.markdown("""
        **Original Image**  
        Retinal fundus photograph showing natural vessel appearance.

        **Prediction (Grayscale)**  
        U-Net model vessel segmentation result:
        - White areas show predicted vessels
        - Black areas show predicted background

        **Image + Prediction**  
        Original image with model prediction overlay:
        - Prediction result overlaid with 50% transparency
        - Combines original image anatomy with segmentation output

        **Usage**:  
        Compare vessel patterns across all three images for comprehensive assessment.
        """)

else:
    st.info("Please upload a retinal fundus image using the sidebar uploader.")

# === FOOTER ===
st.markdown("---")
st.caption("U-Net Retinal Vessel Segmentation Demo | Professional Medical Interface")
