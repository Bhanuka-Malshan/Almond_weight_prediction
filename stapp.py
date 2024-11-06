import joblib
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px 

# Function to load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv("Almond1.csv")  # Replace with the path to your dataset
    return data

# Load data
data = load_data()

# Define the pages
def home_page():
    st.title("Almond Weight Prediction App")
    
    st.image("almond_image.Jpeg")  # Replace with your image path
    st.video("almond_video.mp4")  # Replace with your video path
    
    st.write("""
    Welcome to the Almond Classification App! Here, you can classify different types of almonds based on their features.
    Use the sidebar to navigate through the different pages and predict almond types using our machine learning model.
    """)
    df= pd.read_csv("almond.csv")
    st.subheader('BMI Distribution')
    fig, ax = plt.subplots()
    sns.histplot(df['BMI'], kde= True, bins=20, ax=ax)
    ax.set_title('BMI Ditribution of Individuals')
    ax.set_xlabel('BMI')
    ax.set_ylabel('Frequency')
    #ax.set_xticklabels(['No Heart Attack', 'Heart Attack'])
    st.pyplot(fig)
    st.write("")
    st.write("")
    st.write("")


    gender_counts = df['Gender'].value_counts().reset_index()
    gender_counts.columns = ['Gender', 'count']

    fig = px.pie(gender_counts, values='count', names='Gender', title='Gender Distribution of Individuals')
    st.plotly_chart(fig)



def model_page():
    # Title of the page
    st.title("Almond Classification Model")
    
    # Sidebar for user inputs
    st.sidebar.header("Model Hyperparameters")

    # Input widgets for model parameters
    n_estimators = st.sidebar.slider("Number of Estimators", 10, 200, 100, help="The number of trees in the forest.")
    max_depth = st.sidebar.slider("Maximum Depth", 1, 20, 10, help="The maximum depth of the tree.")
    random_state = st.sidebar.number_input("Random State", value=42, help="Random seed for reproducibility.")

    # Preprocessing
    st.subheader("Preprocessing the Data")
    X = data.drop('Type', axis=1)
    y = data['Type']

    # Display feature and target summaries
    with st.expander("Feature Summary"):
        st.write(X.describe())

    with st.expander("Target Summary"):
        st.write(y.value_counts())

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # Train the model
    st.subheader("Training the Model")
    train_button = st.button("Train Model")

    if train_button:
        with st.spinner('Training in progress...'):
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
            model.fit(X_train, y_train)

            # Predict on the test set
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            st.success(f"Model Trained! Accuracy: **{accuracy:.2f}**")

            # Display Confusion Matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)

    # Media file uploader (Image Upload)
    st.subheader("Upload an Image of an Almond")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

def prediction_page():
    st.title("Almond Weight Prediction")

    # Load the pre-trained model
    loaded_model = joblib.load(open('almond.joblib', 'rb'))

    def almond_prediction(input_data):
        try:
            # Convert input data to numpy array and reshape it
            input_data_as_numpy_array = np.asarray(input_data, dtype=float)  # Convert to float for compatibility
            input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
            # Predict using the loaded model
            prediction = loaded_model.predict(input_data_reshaped)
            return f"Predicted Almond weight: {prediction[0]}g"
        except ValueError as e:
            return "Error: Please ensure all inputs are valid numbers."

    def main():
        st.markdown("### Enter your details")
        # Collect inputs from the user
        Gender = st.text_input('Gender (Male=1 , Female=0)')
        Height_cm = st.text_input('Height (cm)')
        Weight_kg = st.text_input('Weight (kg)')
        Cholesterol_Level = st.text_input('Cholesterol Level')
        BMI = st.text_input('BMI')
        Blood_Glucose_Level = st.text_input('Blood Glucose Level')
        Bone_Density = st.text_input('Bone Density')
        Vision_Sharpnes = st.text_input('Vision Sharpnes')
        Hearing_Ability = st.text_input('Hearing Ability')
        Age = st.text_input('Age')

        # Button to trigger prediction
        if st.button('Click to Test Result'):
            with st.spinner('Please wait...'):
                try:
                    # Convert inputs to appropriate format for prediction
                    inputs = [
                        float(Gender), float(Height_cm), float(Weight_kg), float(Cholesterol_Level), 
                        float(BMI), float(Blood_Glucose_Level), float(Bone_Density), 
                        float(Vision_Sharpnes), float(Hearing_Ability), float(Age)
                    ]
                    diagnosis = almond_prediction(inputs)
                    st.success(diagnosis)
                except ValueError:
                    st.error("Please enter valid numeric values for all fields.")

    main()
