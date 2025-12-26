# app.py
from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("../model/iris_model.pkl")

# Home page
@app.route('/')
def home():
    return render_template("index.html")

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        features = [float(x) for x in request.form.values()]
        final_features = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(final_features)
        target_names = ['Setosa', 'Versicolor', 'Virginica']
        result = target_names[prediction[0]]
        
        return render_template("index.html", prediction_text=f"Predicted Iris Species: {result}")
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
