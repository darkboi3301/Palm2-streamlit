#create a streamlit app that takes email content and returns a prediction of whether the email is spam or not
import streamlit as st
import pandas as pd
import numpy as np
import vertexai
from vertexai.language_models import TextGenerationModel
import os

st.set_page_config(page_title='Palm2 Streamlit',
    page_icon='💀',
    layout='centered', 
    initial_sidebar_state='auto'
    )


st.toast("All Requred Libraries Imported",
    icon='🎉'
    )


os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './privcreds.json'
vertexai.init(project="orbital-nova-365616", location="us-central1")
st.toast("Connected To Google Palm 2",icon='🎉')
st.title('Palm2-streamlit')
st.subheader('pls dont abuse my api key i hav ennough credits for only like 15k requests')
pre_input_text =st.text_area('Tuning prompt here')
input_text =st.text_area('Prompt here')

with st.expander('Advanced Options'):
    temperature = st.slider('Select temprature', 0.0, 1.0, 0.2)
    output_tokens = st.slider('Select max output tokens', 0, 1000, 200)
    model_v = st.selectbox('Select Palm model', ['text-bison', 'text-bison-32k', 'text-bison@001'])
    


parameters = {
    "candidate_count": 1,
    "max_output_tokens": output_tokens,
    "temperature": temperature,
    "top_p": 0.8,
    "top_k": 40
}
model = TextGenerationModel.from_pretrained(model_v)


scan_button = st.button('Scan')
if scan_button:
    prediction_text = pre_input_text + "\n" + input_text
    if prediction_text != '':
        with st.spinner('Getting response from Palm 2...'):
            response = model.predict(
            prediction_text,
            **parameters)
        st.write(response.text)
    else:
        st.write("Null response")