import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.color import label2rgb

# Title
st.title('Image Segmentation App')

# Upload an image
uploaded_image = st.file_uploader('Upload an image', type=['jpg', 'png', 'jpeg'])

if uploaded_image is not None:
    # Load the uploaded image
    image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), cv2.IMREAD_UNCHANGED)

    # Display the original image
    st.image(image, caption='Original Image', use_column_width=True)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to create a binary image
    _, thresholded = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Reshape the image to a 2D array of pixels
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # Define the number of clusters (K)
    K = st.slider('Number of Clusters (K)', min_value=2, max_value=10, value=3)

    # Perform K-means clustering
    _, labels, centers = cv2.kmeans(pixel_values, K, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2), 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert back to 8-bit values
    centers = np.uint8(centers)

    # Map the labels to center values
    kmeans_segmented = centers[labels.flatten()].reshape(image.shape)

    # Display K-Means Clustering result
    st.image(cv2.cvtColor(kmeans_segmented, cv2.COLOR_BGR2RGB), caption=f'K-Means Clustering (K={K})', use_column_width=True)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, threshold1=30, threshold2=100)

    # Display Canny Edge Detection result
    st.image(edges, caption='Canny Edge Detection', use_column_width=True)

    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Apply Felzenszwalb's Graph-Based Superpixel Segmentation
    segments = cv2.ximgproc.segmentation.createGraphSegmentation()
    segments.setSigma(0.8)
    segments.setK(200)
    graph_segmented = segments.processImage(image)

    st.image(graph_segmented, caption='Felzenszwalb's Graph-Based', use_column_width=True)

    # Apply SLIC superpixel segmentation
    slic_segments = slic(image, n_segments=100, compactness=10)
    slic_segmented = label2rgb(slic_segments, image=image, kind='avg')

    # Display SLIC Superpixel Segmentation result
    st.image(slic_segmented, caption='Graph-Based Segmentation (SLIC)', use_column_width=True)
