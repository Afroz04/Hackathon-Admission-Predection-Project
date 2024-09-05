import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Function to load data and train the model
def load_data_and_train_model():
    # Load the CSV data
    data = pd.read_csv("/Users/afrozalam/Desktop/Admission Prediction/Admission_Predict_Ver1.1.csv")

    # Prepare the data
    X = data[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research']]
    y = data['Chance of Admit ']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a simple model (e.g., Linear Regression)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model

# Function to make predictions
def predict_admission(model, gre_score, toefl_score, university_rating, sop, lor, cgpa, research):
    input_data = np.array([[gre_score, toefl_score, university_rating, sop, lor, cgpa, research]])
    prediction = model.predict(input_data)
    return prediction[0]

# Streamlit app
def main():
    st.title("Graduate Admission Prediction App")

    # Train the model on startup
    model = load_data_and_train_model()

    # Input fields
    gre_score = st.number_input("GRE Score", min_value=0, max_value=340, value=300)
    toefl_score = st.number_input("TOEFL Score", min_value=0, max_value=120, value=100)
    university_rating = st.number_input("University Rating", min_value=1, max_value=5, value=3)
    sop = st.slider("Statement of Purpose (SOP) Strength", min_value=1.0, max_value=5.0, value=3.0)
    lor = st.slider("Letter of Recommendation (LOR) Strength", min_value=1.0, max_value=5.0, value=3.0)
    cgpa = st.number_input("Undergraduate CGPA", min_value=0.0, max_value=10.0, value=8.0)
    research = st.selectbox("Research Experience", [0, 1])

    # Button to make prediction
    if st.button("Predict Admission Chance"):
        prediction = predict_admission(model, gre_score, toefl_score, university_rating, sop, lor, cgpa, research)
        st.success(f"The predicted chance of admission is {prediction:.2f}")

if __name__ == '__main__':
    main()
