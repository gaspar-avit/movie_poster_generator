## Alternative movie poster generator



import streamlit as st
import pandas as pd
import numpy as np
import json
import requests
import os

from streamlit import session_state as session
from datetime import time, datetime
from zipfile import ZipFile
from kaggle.api.kaggle_api_extended import KaggleApi
from sentence_transformers import SentenceTransformer
from diffusers import DiffusionPipeline


###############################
## ------- FUNCTIONS ------- ##
###############################

#@st.cache(persist=True, show_spinner=False, suppress_st_warning=True)
@st.experimental_memo(persist=True, show_spinner=False, suppress_st_warning=True)
def load_dataset():
    """
    Load Dataset from Kaggle
    -return: dataframe containing dataset
    """
    # Downloading Movies dataset
    api.dataset_download_file('rounakbanik/the-movies-dataset', 'movies_metadata.csv')

    # Extract data
    zf = ZipFile('movies_metadata.csv.zip')
    zf.extractall() 
    zf.close()

    # Create dataframe
    data = pd.read_csv('movies_metadata.csv', low_memory=False)

    return data

def query_summarization(text):
    """
    Get summarization from HuggingFace Inference API
    -param text: text to be summarized
    -return: summarized text
    """
    API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
    headers = {"Authorization": f"Bearer {st.secrets['hf_token']}"}
    payload = {"inputs": f"text",}
    
    response = requests.request("POST", API_URL, headers=headers, json=payload)
    return response.json()

def generate_poster(movie_data):
    """
    Function for recommending movies
    -param movie_data: metadata of movie selected by user
    -return: image of generated alternative poster
    """

    st.write(movie_data.overview)

    # Get summarization of movie synopsis
    synopsis_sum =  query_summarization(movie_data.overview)
    st.text(synopsis_sum)

    # Get image based on synopsis
    pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    image = pipe(prompt).images[0]
    st.image(image, caption=movie_data.title)

    return None #poster_image


###############################
## --- CONNECT TO KAGGLE --- ##
###############################

# Authenticate Kaggle account
os.environ['KAGGLE_USERNAME'] = st.secrets['username']
os.environ['KAGGLE_KEY'] = st.secrets['key']


api_token = {"username":st.secrets['username'],"key":st.secrets['key']}
with open('/home/appuser/.kaggle/kaggle.json', 'w') as file:
    json.dump(api_token, file)


# Activate Kaggle API
api = KaggleApi()

try:
    api.authenticate()
except:
    with open('/home/appuser/.kaggle/kaggle.json', 'w') as file:
        json.dump(api_token, file)
    api.authenticate() 



###############################
## --------- MAIN ---------- ##
###############################

image = None

# Create dataset
data = load_dataset()


st.title("""
Alternative Movie Poster Generator :film_frames:
This is a movie poster generator based on movie's synopsis :sunglasses:
Just select the title of a movie to generate an alternative poster.
 """)

st.text("")
st.text("")
st.text("")
st.text("")

session.selected_movie = st.selectbox(label="Select a movie to generate alternative poster", options=data.title)

st.text("")
st.text("")

buffer1, col1, buffer2 = st.columns([1.3, 1, 1])

is_clicked = col1.button(label="Generate poster!")


if is_clicked:
    image = generate_poster(data[data.title==session.selected_movie])

st.text("")
st.text("")
st.text("")
st.text("")

#if data is not None:
#    st.table(data)
