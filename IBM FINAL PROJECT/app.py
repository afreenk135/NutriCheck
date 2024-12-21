from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('nutricheck_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/calculator', methods=['GET', 'POST'])
def calculator():
    if request.method == 'POST':
        age = float(request.form['age'])
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        diet_quality = float(request.form['diet_quality'])
        
        # Prepare input for the model
        input_data = np.array([[age, height, weight, diet_quality]])
        
        # Make prediction
        risk_score = model.predict(input_data)[0]
        
        return render_template('results.html', risk_score=risk_score)
    return render_template('calculator.html')

if __name__ == '__main__':
    app.run(debug=True)
