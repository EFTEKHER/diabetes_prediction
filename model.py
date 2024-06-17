import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import pickle

# Load the dataset
diabetes_dataset = pd.read_csv('diabetes.csv')

# Prepare the data
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

# Standardize the data
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X_standardized, Y, test_size=0.2, stratify=Y, random_state=2)

# Define models
models = {
    "Support Vector Machine": SVC(kernel='linear'),
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Random Forest": RandomForestClassifier()
}

# Train models and save them
for model_name, model in models.items():
    model.fit(X_train, Y_train)
    model_filename = f"models/{model_name.lower().replace(' ', '_')}_model.pkl"
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)

# Save the scaler
with open('models/scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)
