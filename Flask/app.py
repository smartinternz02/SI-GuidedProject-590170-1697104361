import numpy as np
import joblib
import pandas as pd
import os
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the model and encoder
model = joblib.load('C:/Users/Saireena/Desktop/miniproject/model.pkl')
encoder = joblib.load('C:/Users/Saireena/Desktop/miniproject/encoder.pkl')

@app.route('/')  # Route to display the home page
def home():
    return render_template('index.html')  # Rendering the home page

@app.route('/predict', methods=["POST"])  # Route to show the predictions in a web UI
def predict():
    if request.method == 'POST':
        try:
            input_features = [
                float(request.form['holiday']),
                float(request.form['temp']),
                float(request.form['rain']),
                float(request.form['snow']),
                float(request.form['weather']),
                float(request.form['year']),
                float(request.form['month']),
                float(request.form['day']),
                float(request.form['hours']),
                float(request.form['minutes']),
                float(request.form['seconds'])
            ]

            features_values = np.array(input_features).reshape(1, -1)
            prediction = model.predict(features_values)
            text = f"Estimated Traffic Volume is: {prediction[0]}"
            return render_template("index.html", prediction_text=text)
        except Exception as e:
            print(f"An error occurred during prediction: {e}")  # Print the specific error for debugging
            return "An error occurred during prediction."

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
