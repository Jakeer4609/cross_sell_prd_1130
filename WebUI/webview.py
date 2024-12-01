# create a new folder
# copy the pickle file and train-dataset-file to that new folder

# web app development using streamlit
# load the necessary libraries

# try to install by  below command
# !pip install streamlit

import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.title("cross-sell-prediction App")

# read the dataset to fill the values in input options of each element
df = pd.read_csv('train.csv')

# create the input elements
# categorical columns
Gender = st.selectbox("Gender", pd.unique(df['Gender']))
Vehicle_Age = st.selectbox("Vehicle_Age", pd.unique(df['Vehicle_Age']))
Vehicle_Damage = st.selectbox("Vehicle_Damage", pd.unique(df['Vehicle_Damage']))

# non-categorical columns
Age = st.number_input("Age")
Driving_License = st.number_input("Driving_License")
Region_Code = st.number_input("Region_Code")
Previously_Insured = st.number_input("Previously_Insured")
Policy_Sales_Channel = st.number_input("Policy_Sales_Channel")
Vintage = st.number_input("Vintage")

# map the user inputs to respective column format
inputs = {
'Gender' : Gender,
'Age' : Age,
'Driving_License' : Driving_License,
'Region_Code' : Region_Code,
'Previously_Insured' : Previously_Insured,
'Vehicle_Age' : Vehicle_Age,
'Vehicle_Damage' : Vehicle_Damage,
'Policy_Sales_Channel' : Policy_Sales_Channel,
'Vintage' : Vintage

}

# load the model from the pickle file
model = joblib.load('pipeline_model.pkl')

# action for submit button
if st.button('Predict'):
    X_input = pd.DataFrame(inputs,index=[0])    
    prediction = model.predict(X_input)
    st.write("The predicted value is:")
    st.write(prediction)

#File upload experiment
st.subheader("Please upload a csv file for prediction")
upload_file = st.file_uploader("Choose a csv file", type=['csv'])

if upload_file is not None:
    df = pd.read_csv(upload_file)

    st.write("File uploaded successfully") 
    st.write(df.head(2))

    if st.button("Predict for the uploaded file"):
        df['Response'] = model.predict(df)
        st.write("Prediction completed")
        st.write(df.head(2))
        st.download_button(label="Download Prediction", 
                           data=df.to_csv(index=False), 
                           file_name="predictions.csv", mime="text/csv")


