import joblib
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Function to load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv("Almond1.csv")  # Ensure correct path
    return data

# Load the pre-trained model with caching
@st.cache_resource
def load_model():
    try:
        model = joblib.load("almond.joblib")  # Ensure correct model path
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# Almond prediction function with input validation and error handling
def almond_prediction(input_data):
    try:
        model = load_model()
        if model is None:
            return "Model could not be loaded."
        
        # Convert input data to numpy array and reshape
        input_data_as_array = np.asarray(input_data, dtype=float)
        input_data_reshaped = input_data_as_array.reshape(1, -1)
        
        # Perform prediction
        prediction = model.predict(input_data_reshaped)
        return f"Predicted Almond weight: {prediction[0]}g"
    except Exception as e:
        return f"Prediction error: {e}"

# Define the prediction page
def prediction_page():
    st.title("Almond Weight Prediction")

    st.markdown("### Enter your details")
    Gender = st.text_input('Gender (Male=1, Female=0)')
    Height_cm = st.text_input('Height (cm)')
    Weight_kg = st.text_input('Weight (kg)')
    Cholesterol_Level = st.text_input('Cholesterol Level')
    BMI = st.text_input('BMI')
    Blood_Glucose_Level = st.text_input('Blood Glucose Level')
    Bone_Density = st.text_input('Bone Density')
    Vision_Sharpnes = st.text_input('Vision Sharpness')
    Hearing_Ability = st.text_input('Hearing Ability')
    Age = st.text_input('Age')

    if st.button('Click to Test Result'):
        inputs = [Gender, Height_cm, Weight_kg, Cholesterol_Level, BMI,
                  Blood_Glucose_Level, Bone_Density, Vision_Sharpnes, Hearing_Ability, Age]
        result = almond_prediction(inputs)
        st.success(result)

# Sidebar Navigation
st.sidebar.title("Navigation")
page_selection = st.sidebar.radio("Go to", ["Home", "Almond Classification Model", "Prediction"])

# Define pages
def home_page():
    st.title("Almond Weight Prediction App")
    st.image("almond_image.Jpeg")  # Replace with correct image path
    st.video("almond_video.mp4")  # Replace with correct video path
    st.write("""
    Welcome to the Almond Classification App! Navigate through different pages to explore almond classification.
    """)

def model_page():
    st.title("Almond Classification Model")
    st.sidebar.header("Model Hyperparameters")

    # Model hyperparameters
    n_estimators = st.sidebar.slider("Number of Estimators", 10, 200, 100)
    max_depth = st.sidebar.slider("Maximum Depth", 1, 20, 10)
    random_state = st.sidebar.number_input("Random State", value=42)

    # Preprocessing
    st.subheader("Preprocessing the Data")
    data = load_data()
    X = data.drop('Type', axis=1)
    y = data['Type']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # Train model
    if st.button("Train Model"):
        with st.spinner("Training..."):
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.success(f"Model Accuracy: {accuracy:.2f}")

            # Display Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)

# Display selected page
if page_selection == "Home":
    home_page()
elif page_selection == "Almond Classification Model":
    model_page()
elif page_selection == "Prediction":
    prediction_page()
