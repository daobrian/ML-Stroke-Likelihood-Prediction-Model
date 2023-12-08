# ML-Stroke-Likelihood-Prediction-Model

## Introduction
The development of this stroke prediction web app was driven by the increasing need for proactive health management. Stroke, a serious medical condition, often comes with long-term consequences. By leveraging predictive analytics, we aim to empower individuals to make informed decisions about their health and take preventive measures to reduce the risk of stroke.

We decided to engineer a web application to predict the likelihood of developing a stroke based on clinical and lifestyle features. Our web application is built to empower users by providing personalized predictions on the likelihood of a stroke. It leverages a logistic regression machine learning model trained on relevant health and lifestyle data. Our model is tuned and optimized for recall. Since this is a medical application, we use recall to evaluate model performance because we want to point out and prevent as many risky patients as possible even if we must sacrifice precision. We don't mind higher false positive rates as we are interested in warning and prevention as opposed to under reporting.

## About the Dataset
The dataset contains records from both ischemic and hemorrhagic strokes. This dataset is used to predict whether a patient is likely to experience a stroke based on the input parameters like gender, age, various diseases, and smoking status. Each row in the data provides relevant information about the patient. We are going to use this dataset to train a ML model that predicts the likelihood of a person having a stroke based on various clinical features.

## Front End
We used streamlit to run our Logistic Regression ML model. Our app uses the following features for prediction: age, BMI, average glucose level, hypertension status, marital status, work type, smoking status, and heart disease history. Each of these features plays a significant role in understanding an individual's overall health and lifestyle. Our model then evaluates the combination of input features and outputs a prediction of whether the patient is at risk for a stroke and provides the probability as a percentage. One of the key features of the application is the provision of real-time feedback. As users input their data, the application dynamically responds, offering insights into how each input contributes to the overall prediction. This real-time interaction creates an engaging and educational experience for users, fostering a deeper understanding of the factors influencing stroke risk and hopefully help them to take proactive steps towards improving their health.

## Dependencies for Notebook
* collections
* imblearn
* matplotlib
* numpy
* pandas
* pickle
* plotly
* seaborn
* sklearn
* warnings

## Dependencies for Web App
* numpy
* pickle
* sklearn
* streamlit

## To run the project
1) Make sure all dependencies are installed.
2) Verify that model_and_scaler.pkl is in the same directory as app.py
3) Use `streamlit run app.py` in the terminal to run the script and launch the app locally.
4) The app is also deployed here: [ml-stroke-likelihood-prediction-model.streamlit.app](ml-stroke-likelihood-prediction-model.streamlit.app)
   
## Collaborators
Aadvika Ahuja, Brian Dao, Srihita Ramini, Tony Tran
