import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import streamlit as st

# Step 1: Loading Raw Data
def load_data(file_path):
    """Load the heart disease dataset."""
    data = pd.read_csv("heart_disease_raw_data.csv")  # Replace with dataset path
    return data

# Step 2: Data Cleaning & EDA
def clean_data(data):
    """Clean and preprocess the dataset."""
    # Check for missing values and fill with mean
    if data.isnull().sum().sum() > 0:
        data = data.fillna(data.mean())
    
    # Drop duplicate rows
    data = data.drop_duplicates()

    return data

def eda(data):
    """Perform Exploratory Data Analysis (EDA)."""
    st.write("### Dataset Overview")
    st.write(data.head())

    st.write("### Summary Statistics")
    st.write(data.describe())

    st.write("### Correlation Heatmap")
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Step 3: Feature Selection and Model Training
def train_model(data):
    """Train a predictive model on the dataset."""
    # Splitting data into features and labels
    X = data.iloc[:, :-1]  # Features
    y = data.iloc[:, -1]   # Target

    # Standardizing the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training a Random Forest Classifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Model evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"### Model Accuracy: {accuracy * 100:.2f}%")

    st.write("### Classification Report")
    st.text(classification_report(y_test, y_pred))

    return model, scaler

# Step 4: User Input and Prediction
def user_input_features():
    """Collect user input features for prediction."""
    age = st.number_input('Age', min_value=1, max_value=120, value=25)
    sex = st.selectbox('Sex (1 = Male, 0 = Female)', [1, 0])
    cp = st.number_input('Chest Pain Type (0-3)', min_value=0, max_value=3, value=1)
    trestbps = st.number_input('Resting Blood Pressure', min_value=50, max_value=200, value=120)
    chol = st.number_input('Cholesterol Level', min_value=100, max_value=600, value=200)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)', [1, 0])
    restecg = st.number_input('Resting ECG Results (0-2)', min_value=0, max_value=2, value=1)
    thalach = st.number_input('Maximum Heart Rate Achieved', min_value=50, max_value=250, value=150)
    exang = st.selectbox('Exercise Induced Angina (1 = Yes, 0 = No)', [1, 0])
    oldpeak = st.number_input('ST Depression Induced by Exercise', min_value=0.0, max_value=10.0, value=1.0)
    slope = st.number_input('Slope of the Peak Exercise ST Segment (0-2)', min_value=0, max_value=2, value=1)
    ca = st.number_input('Number of Major Vessels (0-3)', min_value=0, max_value=3, value=0)
    thal = st.number_input('Thalassemia (1-3)', min_value=1, max_value=3, value=2)

    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    features = pd.DataFrame(data, index=[0])
    return features

def make_prediction(model, scaler, input_data):
    """Make a prediction based on user input."""
    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_data_scaled)
    probabilities = model.predict_proba(input_data_scaled)

    # Debugging output
    st.write(f"Prediction: {prediction[0]}")
    st.write(f"Heart Disease Probability: {probabilities[0][1]:.2f}")

    # Extract probability of heart disease
    heart_disease_prob = probabilities[0][1]

    # Interpretation of Results
    if heart_disease_prob > 0.6:  # High probability threshold
        result = "You have high chances of heart disease. Please consult a doctor."
    elif 0.4 <= heart_disease_prob <= 0.6:  # Borderline case
        result = "You are on the borderline. Take precautionary measures and consult a healthcare professional."
    else:  # Low probability threshold
        result = "You are safe. Keep maintaining a healthy lifestyle."

    return result

# Step 5: Streamlit Web App
def main():
    st.title("Heart Disease Prediction App")

    # Load and clean data
    data = load_data('heart_disease.csv')  # Replace with your file path
    clean_data_df = clean_data(data)

    # Perform EDA
    eda(clean_data_df)

    # Train the model
    model, scaler = train_model(clean_data_df)

    # Collect user input
    st.sidebar.header('User Input Parameters')
    input_data = user_input_features()

    # Make predictions
    if st.button('Predict'):
        result = make_prediction(model, scaler, input_data)
        st.write("### Prediction Result")
        st.write(result)

if __name__ == '__main__':
    main()
