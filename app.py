## Alternative movie poster generator
## @author: Gaspar Avit Ferrero


import streamlit as st
import pandas as pd
import numpy as np
import json
import requests
import os
import io
import string
import random
from streamlit import session_state as session
from datetime import time, datetime
from zipfile import ZipFile
from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
from htbuilder.funcs import rgba, rgb
from PIL import Image



###############################
## --- GLOBAL VARIABLES ---- ##
###############################


PATH_JSON = '/home/user/.kaggle/kaggle.json'



# Environment variables to authenticate Kaggle account
os.environ['KAGGLE_USERNAME'] = st.secrets['username']
os.environ['KAGGLE_KEY'] = st.secrets['key']
os.environ['KAGGLE_CONFIG_DIR'] = PATH_JSON

from kaggle.api.kaggle_api_extended import KaggleApi



###############################
## ------- FUNCTIONS ------- ##
###############################

def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)


def layout(*args):

    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
     .stApp { bottom: 105px; }
    </style>
    """

    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        color="black",
        text_align="center",
        height="auto",
        opacity=1
    )

    style_hr = styles(
        display="block",
        margin=px(8, 8, "auto", "auto"),
        border_style="inset",
        border_width=px(2)
    )

    body = p()
    foot = div(
        style=style_div
    )(
        hr(
            style=style_hr
        ),
        body
    )

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)

    
def footer():
    myargs = [
        "Made with ❤️ by ",
        link("https://www.linkedin.com/in/gaspar-avit/", "Gaspar Avit"),
    ]
    layout(*myargs)

    
def authenticate_kaggle():
    # Connect to kaggle API

    # Save credentials to json file
    if not os.path.exists(PATH_JSON):
        api_token = {"username":st.secrets['username'],"key":st.secrets['key']}
        with open(PATH_JSON, 'w') as file:
            json.dump(api_token, file)

    # Activate Kaggle API
    global api
    api = KaggleApi()
    api.authenticate()


@st.experimental_memo(persist=True, show_spinner=False, suppress_st_warning=True, max_entries=1)
def load_dataset():
    """
    Load Dataset from Kaggle
    -return: dataframe containing dataset
    """

    ## --- Connect to kaggle API --- ##
    # Save credentials to json file
    if not os.path.exists(PATH_JSON):
        api_token = {"username":st.secrets['username'],"key":st.secrets['key']}
        with open(PATH_JSON, 'w') as file:
            json.dump(api_token, file)

    # Activate Kaggle API
    global api
    api = KaggleApi()
    api.authenticate()
    ## ----------------------------- ##

    # Downloading Movies dataset
    api.dataset_download_file('rounakbanik/the-movies-dataset', 'movies_metadata.csv')

    # Extract data
    zf = ZipFile('movies_metadata.csv.zip')
    zf.extractall() 
    zf.close()

    # Create dataframe
    data = pd.read_csv('movies_metadata.csv', low_memory=False)
    data['year'] = data["release_date"].map(lambda x: x.split('-')[0] if isinstance(x, str) else '0')
    data['title_year'] = data['title'] + ' (' + data['year'] + ')'

    return data


def query_summary(text):
    """
    Get summarization from HuggingFace Inference API
    -param text: text to be summarized
    -return: summarized text
    """
    API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
    headers = {"Authorization": f"Bearer {st.secrets['hf_token']}"}
    payload = {"inputs": f"{text}",}
    
    response = requests.request("POST", API_URL, headers=headers, json=payload).json()
    
    try:
        text = response[0].get('summary_text')
    except:
        text = response[0]
    return text


def query_generate(text, genres, year):
    """
    Get image from HuggingFace Inference API
    -param text: text to generate image
    -return: generated image
    """
    API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
    headers = {"Authorization": f"Bearer {st.secrets['hf_token']}"}
    text = 'A movie Poster based on the following synopsis: \"' + text + '\". Style: ' + genres + ', year ' + year + \
    '. Ignore ' + ''.join(random.choices(string.ascii_letters, k=10))
    payload = {"inputs": f"{text}",}
    
    response = requests.post(API_URL, headers=headers, json=payload)

    try:
        response_str = response.content.decode("utf-8")
        if 'error' in response_str:
            payload = {"inputs": f"{text}",
                       "options": {"wait_for_model": True},
            }
            response = requests.post(API_URL, headers=headers, json=payload)
    except:
        pass

    return response.content

@st.experimental_memo(persist=False, show_spinner=False, suppress_st_warning=True)
def generate_poster(movie_data):
    """
    Function for recommending movies
    -param movie_data: metadata of movie selected by user
    -return: image of generated alternative poster
    """

    # Get movie metadata
    genres = [i['name'] for i in eval(movie_data['genres'].values[0])]
    genres_string = ', '.join(genres)

    year = movie_data['year'].values[0]


    # Get summarization of movie synopsis
    st.text("")
    with st.spinner("Please wait while the synopsis is being summarized..."):
        synopsis_sum = query_summary(movie_data.overview.values[0])

    # Print summarized synopsis
    st.text("")
    synopsis_expander = st.expander("Show synopsis", expanded=False)
    with synopsis_expander:
        st.subheader("Summarized synopsis:")
        col1, col2 = st.columns([5, 1])
        with col1:
            st.write(synopsis_sum)
    st.text("")


    # Get image based on synopsis
    with st.spinner("Generating poster image... This could take a few minutes."):
        response_content = query_generate(synopsis_sum, genres_string, year)

    # Show image
    try: 
        image = Image.open(io.BytesIO(response_content))

        st.text("")
        st.text("")
        st.subheader("Resulting poster:")
        st.text("")
        col1, col2, col3 = st.columns([1, 5, 1])
        with col2:
            st.image(image, caption="Movie: \"" + movie_data.title.values[0] + "\"")
        del image

    except:
        col1, col2 = st.columns([5, 1])
        with col1:
            st.write(response_content)

    return response_content
# ------------------------------------------------------- #


###############################
## --------- MAIN ---------- ##
###############################


if __name__ == "__main__":


    # Initialize image variable
    poster = None

    ## --- Page config ------------ ##
    # Set page title
    st.title("""
    Movie Poster Generator :film_frames:

    #### This is a movie poster generator based on movie's synopsis :sunglasses:

    #### Just select the title of a movie to generate an alternative poster.
    """)

    # Set page footer
    footer()
    ## ---------------------------- ##


    ## Create dataset
    data = load_dataset()

    st.text("")
    st.text("")
    st.text("")
    st.text("")

    ## Select box with all the movies as choices
    session.selected_movie = st.selectbox(label="Select a movie to generate alternative poster", options=data.title_year)

    st.text("")
    st.text("")

    ## Create button to trigger poster generation
    buffer1, col1, buffer2 = st.columns([1.3, 1, 1])
    is_clicked = col1.button(label="Generate poster!")

    ## Clear cache between runs
    st.runtime.legacy_caching.clear_cache()
    generate_poster.clear()

    ## Generate poster
    if is_clicked:
        poster = generate_poster(data[data.title_year==session.selected_movie])
        generate_poster.clear()
        st.runtime.legacy_caching.clear_cache()


    _= """
    is_clicked_rerun = None
    if poster is not None:
        buffer1, col1, buffer2 = st.columns([1.3, 1, 1])
        is_clicked_rerun = col1.button(label="Rerun with same movie!")

    if is_clicked_rerun:
        poster = generate_poster(data[data.title_year==selected_movie])
    """
