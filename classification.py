# titanic_app.py
"""
User interface to load user images for classification
> streamlit run classification.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import time



#################################################
################### Interface ###################
#################################################

#title
st.title("Classify your personal images (cell phone, old albums, etc)")
st.markdown("We propose an automatic classification")

