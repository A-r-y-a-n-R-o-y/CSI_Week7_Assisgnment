import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load('iris_model.pkl')
classes = ['Setosa', 'Versicolor', 'Virginica']

st.title("🌸 Iris Flower Classifier")
st.write("Enter flower features to classify its species.")

# Input sliders
sl = st.slider('Sepal length (cm)', 4.0, 8.0, 5.1)
sw = st.slider('Sepal width (cm)', 2.0, 4.5, 3.5)
pl = st.slider('Petal length (cm)', 1.0, 7.0, 1.4)
pw = st.slider('Petal width (cm)', 0.1, 2.5, 0.2)

# Predict
input_data = np.array([[sl, sw, pl, pw]])
pred = model.predict(input_data)[0]
proba = model.predict_proba(input_data)[0]

st.subheader("Prediction")
st.write(f"The flower is **{classes[pred]}**")

st.subheader("Prediction Probabilities")
st.bar_chart(proba)
