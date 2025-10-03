import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("iris_best_model.pkl", "rb") as f:
    model = pickle.load(f)

# Title
st.title("🌸 Iris Flower Species Prediction")

st.write("Enter flower details below to predict its species:")

# User inputs
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)

# Predict button
if st.button("Predict"):
    # Prepare input for model
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Predict
    prediction = model.predict(features)[0]
    
    # Map label numbers to species names (if needed)
    label_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
    
    st.success(f"🌿 The predicted Iris species is: **{label_map[prediction]}**")
