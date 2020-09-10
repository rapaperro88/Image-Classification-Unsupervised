# titanic_app.py
"""
User interface to load user images for classification
> streamlit run classification.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import os



#################################################
################### Interface ###################
#################################################

#title
st.title("Classify your personal images (cell phone, old albums, etc)")
st.markdown("We propose an automatic classification")

filename = st.text_input('Enter a valid file path:')
try:
    with open(filename) as input:
        st.text(input.read())
except FileNotFoundError:
    st.error('File not found.')
