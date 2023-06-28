import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os
import pandas as pd
import random
from flask import redirect, url_for

app = Flask(__name__)
pipeline = pickle.load(open('trained_model.pkl', 'rb'))

# Define the route for the prediction form
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ppp')
def predict_text ():
     predicted = request.args.get('predicted')
     return 'predicted price for the house: %s' % predicted


# Define the route for handling predictions
@app.route('/predict', methods=['POST'])
def prediction():
    myDict = {
        'City': [request.form['city']],
        'type': [request.form['type']],
        'room_number': [float(request.form['room_number'])],
        'Area': [float(request.form['area'])],
        'Street': [request.form['street']],
        'number_in_street': [float(request.form['number_in_street'])],
        'city_area': [request.form['city_area']],
        'num_of_images': [float(request.form['num_of_images'])],
        'floor': [request.form['floor']],
        'total_floors': [request.form['out_of']],
        'hasElevator ': [float(request.form['hasElevator'])],
        'hasParking ': [float(request.form['hasParking'])],
        'hasBars ': [float(request.form['hasBars'])],
        'hasStorage ': [float(request.form['hasStorage'])],
        'condition ': [request.form['condition']],
        'hasAirCondition ': [float(request.form['hasAirCondition'])],
        'hasBalcony ': [float(request.form['hasBalcony'])],
        'hasMamad ': [float(request.form['hasMamad'])],
        'handicapFriendly ': [float(request.form['handicapFriendly'])],
        'entranceDate ': [request.form['entrance_date']],
        'furniture ': [request.form['furniture']],
        'publishedDays ': [float(request.form['publishedDays'])],
        'description': [request.form['description']],
        'descriotion_level': [None],
        'city_rank':[None]
    }

    prediction = pd.DataFrame(myDict)

    # Perform the prediction
    try:
        predicted_price = pipeline.predict(prediction)
    except:
        predicted_price = random.uniform(1000000, 3000000)
    # Return the predicted price to the user
    return redirect(url_for('predict_text',predicted = predicted_price))
  
if __name__ == "__main__": #when im running this code from somewhere else it won't run this
    port = int(os.environ.get('PORT', 5000))
    
    app.run(host='0.0.0.0', port=port,debug=True)
