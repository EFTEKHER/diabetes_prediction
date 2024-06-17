from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load the scaler
with open('models/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Load models
models = {
    "Support Vector Machine": pickle.load(open('models/support_vector_machine_model.pkl', 'rb')),
    "Logistic Regression": pickle.load(open('models/logistic_regression_model.pkl', 'rb')),
    "Decision Tree": pickle.load(open('models/decision_tree_model.pkl', 'rb')),
    "Gradient Boosting": pickle.load(open('models/gradient_boosting_model.pkl', 'rb')),
    "Random Forest": pickle.load(open('models/random_forest_model.pkl', 'rb'))
}

# Metrics (add your pre-computed metrics here, or compute them similarly as in `model.py`)
model_metrics = {
    "Support Vector Machine": {"accuracy_test": 0.78, "recall_test": 0.75, "precision_test": 0.78, "f1_test": 0.76},
    "Logistic Regression": {"accuracy_test": 0.78, "recall_test": 0.75, "precision_test": 0.78, "f1_test": 0.76},
    "Decision Tree": {"accuracy_test": 0.73, "recall_test": 0.72, "precision_test": 0.74, "f1_test": 0.73},
    "Gradient Boosting": {"accuracy_test": 0.78, "recall_test": 0.75, "precision_test": 0.78, "f1_test": 0.76},
    "Random Forest": {"accuracy_test": 0.77, "recall_test": 0.75, "precision_test": 0.77, "f1_test": 0.76},
}

def make_prediction(model, scaler, input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    std_data = scaler.transform(input_data_reshaped)
    prediction = model.predict(std_data)
    return prediction[0]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_data = [
            int(request.form['Pregnancies']),
            int(request.form['Glucose']),
            int(request.form['BloodPressure']),
            int(request.form['SkinThickness']),
            int(request.form['Insulin']),
            float(request.form['BMI']),
            float(request.form['DiabetesPedigreeFunction']),
            int(request.form['Age'])
        ]
        model_name = request.form['model']
        selected_model = models[model_name]
        prediction = make_prediction(selected_model, scaler, input_data)
        result = 'Diabetic' if prediction == 1 else 'Not Diabetic'
        
        return render_template('index.html', result=result, model_metrics=model_metrics, selected_model=model_name)
    return render_template('index.html', model_metrics=model_metrics)

if __name__ == '__main__':
    app.run(debug=True)
