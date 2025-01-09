from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the pre-trained model
model = pickle.load(open('Kidney.pkl', 'rb'))

# Default route to render the home page
@app.route('/')
def home():
    return render_template('home.html')


# Render the home page

# Route for the prediction form
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':  # If the form is submitted
        # Collecting user inputs
        sg = float(request.form['sg'])
        htn = float(request.form['htn'])
        hemo = float(request.form['hemo'])
        dm = float(request.form['dm'])
        al = float(request.form['al'])
        appet = float(request.form['appet'])
        rc = float(request.form['rc'])
        pc = float(request.form['pc'])

        # Prepare input for the model
        values = np.array([[sg, htn, hemo, dm, al, appet, rc, pc]])
        prediction = model.predict(values)

        # Render the result page with the prediction
        return render_template('result.html', prediction=prediction)
    else:  # If accessed via GET
        return render_template('index.html')  # Render the prediction form

if __name__ == "__main__":
    app.run(debug=True)
