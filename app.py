import pandas as pd
import numpy as np
import streamlit as st
import joblib as jl
from backend import predict


@st.cache
def load():
    return jl.load('model/model.pkl')


st.title('I-94 Traffic Volume Predictor')
st.header('Enter information about the situation for which you\'d like to predict traffic:')
day = st.selectbox('Day of Week:', ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
time_of_day = st.selectbox('Time of Day:') ['Early Morning', 'Morning', 'Late Morning', 'Afternoon', 'Evening', 'Night']
month = st.selectbox('Month:', ['January', 'February', 'March', 'April', 'May', 'June',
                                'July', 'August', 'September', 'October', 'November', 'December'])
temperature = st.slider('Temperature (F):', -32, 120, 75)
cloud_density = st.slider('Cloud Density:', 0, 100, 20)
is_holiday = st.checkbox('Holiday:', False)
if st.button('Predict Traffic Volume'):
    model = load()
    volume = predict(model, time_of_day, day, month, is_holiday, temperature, cloud_density)
    st.success(f'The predicted traffic volume is {volume} vehicles per hour.')



# @st.cache
