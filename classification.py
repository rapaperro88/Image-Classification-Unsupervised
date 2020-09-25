"""
User interface to load user images for classification
> streamlit run classification.py
"""

import streamlit as st
import numpy as np
import os
from utils import *

#################################################
############### Helper Functions ################
#################################################

# utils.py file
    
#################################################
################### Interface ###################
#################################################

#title
st.title("Classify your personal images (cell phone, old albums, etc)")
st.markdown("We propose an automatic classification")

folder = st.text_input('Enter a valid file path for folder containing images:')

try:
    images = compile_images(folder)
    st.success("Images Loaded")
    # plt.savefig(path)

except FileNotFoundError:
    st.error('File not found.')

# SELECTION OF : MODEL / ALGORITHM / NUMBER OF CLUSTERS 
# model
model_name = st.radio("Choose a pretrained model", ('resnet50', 'vgg16', 'vgg19'))
input_shape=(128,128,3)
if model_name == 'resnet50':
    model = keras.applications.resnet50.ResNet50(include_top=False, weights="imagenet", input_shape=input_shape)

elif model_name == 'vgg19':
    model = keras.applications.vgg19.VGG19(include_top=False, weights="imagenet", input_shape=input_shape)
# algorithm
algo = st.radio("Choose a clustering algorithm", ('KMeans', 'Gaussian Mixture'))
# number of clusters
nb_clusters = st.slider("How many categories do you think we can cluster your images", min_value=2, max_value=8, value=3, step=1)

# CLASSIFICATION ROUTINE
if st.button("Classify!"):
    model_output = flatten_output(model, images)
    st.success("Convolution Layers applied.")
    model_pca = create_fit_PCA(model_output)
    model_output_pca = model_pca.transform(model_output)
    st.success("Dimensionality reduction performed")
    if algo == "KMeans":
        K_fitted = create_train_kmeans(model_output_pca)
        clusters = K_fitted.predict(model_output_pca)
    elif algo == "Gaussian Mixture":
        GM_fitted = create_train_gmm(model_output_pca)
        clusters = GM_fitted.predict(model_output_pca)

    dictionnary, figs = group_and_plot(folder, clusters, nb_clusters, size=(56,56), plot=True)

    for i, fig in enumerate(figs):
        st.warning(f"Group {i}")
        st.pyplot(fig)