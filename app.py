import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__) #Initialize the flask App
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(X) for X in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model_cat_loan.predict(final_features)

    output = round(prediction[0], 1)

    return render_template('index.html', prediction_text='Loan status is {}'.format(output))

if __name__ == "__main__":
    app.run()
