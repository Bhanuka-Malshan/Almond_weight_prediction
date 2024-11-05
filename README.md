# Almond Weight Prediction App

This is a web-based application developed using **Streamlit** for predicting the weight of almonds based on various input features. The application also includes a section for almond classification using a machine learning model, an interactive data exploration page, and a media file uploader.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Training](#model-training)

## Introduction
The **Almond Weight Prediction App** is designed for predicting almond weights using a pre-trained machine learning model. It also allows users to explore dataset distributions, train a custom classification model, and upload images for visualization.

## Features
- **Home Page**: Introduction, images, videos, and general information about the app.
- **Data Exploration**: Distribution plots (e.g., BMI distribution), gender distribution pie chart.
- **Model Training Page**: Allows users to train a **RandomForestClassifier** with customizable hyperparameters.
- **Prediction Page**: Accepts user input for various almond attributes and predicts the almond's weight.
- **Media Upload**: Option to upload images for display.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/almond-weight-prediction-app.git

2. Navigate to the project directory:
   cd almond-weight-prediction-app

3. Install the required Python packages:
   pip install -r requirements.txt
   Ensure you have joblib, pandas, streamlit, scikit-learn, matplotlib, seaborn, and plotly installed.

4. Run the Streamlit app:
   streamlit run app.py

## Usage
1. Start the app by running the command: streamlit run app.py.
2. Navigate through the sidebar to explore the different sections:
   Home: Overview and welcome content.
   Almond Classification Model: Customize model training with input sliders.
   Prediction: Input almond feature details and get a weight prediction.
   
3. Upload images for visualization on the relevant page.

## Dataset
The application uses a dataset named almond.csv that contains the following columns:
   Gender: 1 for male, 0 for female
   Height_cm
   Weight_kg
   Cholesterol_Level
   BMI
   Blood_Glucose_Level
   Bone_Density
   Vision_Sharpness
   Hearing_Ability
   Age
   almond_meal_weight_g (Target variable)
Ensure the almond.csv file is placed in the root directory for the app to load properly.

## Model Training
The model_page() function allows users to train a RandomForestClassifier with adjustable hyperparameters, such as:
1. n_estimators: Number of trees in the forest.
2. max_depth: Maximum depth of the trees.
3. random_state: Seed for reproducibility.
The model's performance is shown via an accuracy score and a confusion matrix plot.
