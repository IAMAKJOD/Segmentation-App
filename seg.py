import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.color import label2rgb
import streamlit as st
from PIL import Image

# Define a function to apply segmentation techniques
def apply_segmentation(image):
    # Convert the image to grayscale
    gray = image.convert('L')

    # Apply thresholding to create a binary image
    thresholded = gray.point(lambda p: p > 127 and 255)

    # Convert the PIL image to a NumPy array
    image_np = np.array(image)

    # Convert the image to LAB color space
    lab_image = image.convert('LAB')

    # Apply Felzenszwalb's Graph-Based Superpixel Segmentation
    segments = slic(image_np, n_segments=200, compactness=10, sigma=0.8)
    graph_segmented = label2rgb(segments, image=image_np, kind='avg', bg_label=0)

    # Apply SLIC superpixel segmentation
    slic_segments = slic(image_np, n_segments=100, compactness=10)
    slic_segmented = label2rgb(slic_segments, image=image_np, kind='avg')

    return image, thresholded, graph_segmented, slic_segmented

# Streamlit app
st.title('Image Segmentation App')

uploaded_image = st.file_uploader('Upload an image', type=['jpg', 'png', 'jpeg'])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button('Apply Segmentation'):
        segmented_images = apply_segmentation(image)

        st.subheader('Segmentation Results')

        # Create four columns
        columns = st.columns(4)

        for i, segmented_image in enumerate(segmented_images):
            if i < len(columns):
                columns[i].image(segmented_image, caption=f'Technique {i + 1}', use_column_width=True)
