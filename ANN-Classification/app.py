import tensorflow as tf
import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import pickle

#Loading model
model = load_model('my_model.keras')

#Loading pickle files
with open("label_encoding.pickle",'rb') as file:
    label_encode = pickle.load(file)

with open("one_hot_encoding.pickle",'rb') as file:
    one_hot_encode = pickle.load(file)

with open("Standard_Scaler.pickle",'rb') as file:
    standard_scaler = pickle.load(file)

#title
st.write("Customer Churn Prediction")

#input_data
CreditScore = st.number_input('Credit Score')
Gender = st.selectbox('Gender', label_encode.classes_)
Age = st.slider('Age', 18, 92)
Tenure = st.slider('Tenure', 0, 10)
Balance = st.number_input('Balance')
NumOfProducts = st.slider('Number of Products', 1, 4)
HasCrCard = st.selectbox('Has Credit Card', [0, 1])
IsActiveMember = st.selectbox('Is Active Member', [0, 1])
Geography = st.selectbox('Geography', one_hot_encode.categories_[0])



# Create DataFrame
# Create raw input
customer = pd.DataFrame({
    "CreditScore": [CreditScore],
    "Gender": [Gender],
    "Age": [Age],
    "Tenure": [Tenure],
    "Balance": [Balance],
    "NumOfProducts": [NumOfProducts],
    "HasCrCard": [HasCrCard],
    "IsActiveMember": [IsActiveMember],
    "EstimatedSalary": [EstimatedSalary],
    "Geography": [Geography]
})



# Encoding
customer['Gender'] = label_encode.transform(customer['Gender'])
geo_encoded = one_hot_encode.transform(customer[['Geography']]).toarray()
geo_df = pd.DataFrame(geo_encoded, columns=one_hot_encode.get_feature_names_out(['Geography']))

# Drop original Geography
customer = customer.drop(['Geography'], axis=1)

customer = pd.concat([customer,geo_df],axis = 1)

#scaling
scaled_data = standard_scaler.transform(customer)


# Predict
prediction = model.predict(scaled_data)[0][0]

# Output
if prediction > 0.5:
    st.error("Customer is LIKELY to leave the company")
else:
    st.success("Customer is UNLIKELY to leave the company")

st.write(f"Churn Probability: {prediction:.2%}")