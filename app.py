from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

# loading the saved model
model = pickle.load(open('sf.sav', 'rb'))

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST','GET'])
def predict():

    Item_Weight = float(request.form['Item Weight'])
    Item_Fat_Content = float(request.form['Item Fat Content'])
    Item_Visibility = float(request.form['Item Visibility'])
    Item_Type = float(request.form['Item Type'])
    Item_MRP = float(request.form['Item MRP'])
    Outlet_Establishment_Year = float(request.form['Outlet Establishment Year'])
    Outlet_Size = float(request.form['Outlet Size'])
    Outlet_Location_Type = float(request.form['Outlet Location Type'])
    Outlet_Type = float(request.form['Outlet Type'])

    X= np.array([[Item_Weight, Item_Fat_Content, Item_Visibility, Item_Type, Item_MRP, 
                  Outlet_Establishment_Year, Outlet_Size, Outlet_Location_Type, Outlet_Type]])


    prediction = model.predict(X)
    return render_template('index.html', prediction_text = 'Expected Item Outlet Sales: {}'.format(prediction))    

if __name__ == "__main__":
    app.run(debug=True)