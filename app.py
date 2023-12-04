import streamlit as st
import streamlit.web.cli as cli

import numpy as np
import pickle
import sklearn

loaded_model = pickle.load(open('model_and_scaler.pkl', 'rb'))

smoke_dict = {'never' : 0, 'former' : 1, 'smokes' : 2}
yes_no_dict = {'no' : 0, 'yes' : 1}

st.title('Stroke Likelihood Prediction	:brain:')
st.markdown('This app is powered by a logistic regression classifier optimized for recall.')
age = st.slider("Choose age", 0, 100)
bmi = st.slider("Choose BMI", 0, 100)
glucose = st.slider("Choose average glucose level in blood (mg/dL)", 0, 300)
hypertension = st.radio("Patient has hypertension", ['no', 'yes'])
married = st.radio("Married", ['no', 'yes'])
work = st.select_slider("Work type", ['never worked', 'children', 'self employed', 'private', 'govt job'])
smoke = st.radio("Choose smoking status", ['never', 'former', 'smokes'])
heart = st.radio("Patient has heart disease", ['no', 'yes'])

def predict():
    transformed_data = loaded_model['scaler'].transform([[age,bmi,glucose]])[0]
    rw = [[transformed_data[0], yes_no_dict[hypertension], yes_no_dict[heart], yes_no_dict[married], transformed_data[2], transformed_data[1], smoke_dict[smoke], int(work == 'govt job'), int(work == 'never worked'), int(work == 'private'), int(work == 'self employed'), int(work == 'children')]]
    prediction = loaded_model['model'].predict(rw)
    prob = loaded_model['model'].predict_proba(rw)[0]

    if prediction[0] == 1: 
        st.error(f'Patient at risk for stroke | Probability = {np.round(prob[1] * 100, 2)}%')
    else: 
        st.success(f'Patient is not at risk for stroke | Probability = {np.round(prob[0] * 100, 2)}%') 

predict()
