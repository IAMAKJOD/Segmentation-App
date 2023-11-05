import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.color import label2rgb
import streamlit as st

# Define a function to apply segmentation techniques
def apply_segmentation(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to create a binary image
    _, thresholded = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Reshape the image to a 2D array of pixels
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # Define the number of clusters (K)
    K = 3

    # Define the criteria for K-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # Perform K-means clustering
    _, labels, centers = cv2.kmeans(pixel_values, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert back to 8-bit values
    centers = np.uint8(centers)

    # Map the labels to center values
    kmeans_segmented = centers[labels.flatten()].reshape(image.shape)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, threshold1=30, threshold2=100)

    # Convert the image to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Apply Felzenszwalb's Graph-Based Superpixel Segmentation
    segments = cv2.ximgproc.segmentation.createGraphSegmentation()
    segments.setSigma(0.8)
    segments.setK(200)
    graph_segmented = segments.processImage(image)

    # Convert the image to RGB (scikit-image uses RGB format)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply SLIC superpixel segmentation
    slic_segments = slic(image_rgb, n_segments=100, compactness=10)
    slic_segmented = label2rgb(slic_segments, image=image_rgb, kind='avg')

    return image, thresholded, kmeans_segmented, edges, graph_segmented, slic_segmented

# Streamlit app
st.title('Image Segmentation App')

uploaded_image = st.file_uploader('Upload an image', type=['jpg', 'png', 'jpeg'])

if uploaded_image is not None:
    image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button('Apply Segmentation'):
        segmented_images = apply_segmentation(image)

        st.subheader('Segmentation Results')

        # Create six columns
        columns = st.columns(6)

        for i, segmented_image in enumerate(segmented_images):
            if i < len(columns):
                columns[i].image(segmented_image, caption=f'Technique {i + 1}', use_column_width=True)
