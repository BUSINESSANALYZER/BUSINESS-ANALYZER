from flask import Flask, render_template, request
import xgboost as xgb
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# loading the saved model
model = xgb.Booster()
model.load_model('sf.json')

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    
    Item_Weight = float(request.form['Item Weight'])
    
    Item_Fat_Content = request.form['Item Fat Content']
    if (Item_Fat_Content == 'Low Fat'):
        Item_Fat_Content = 0,0
    else:
        Item_Fat_Content = 0,1
    Item_Fat_Content_1 = Item_Fat_Content
    
    Item_Visibility = float(request.form['Item Visibility: Range(0.6-1.8)'])
    
    Item_MRP = float(request.form['Item MRP'])
    
    
    Outlet_Identifier = request.form['Outlet Identifier']
    if (Outlet_Identifier == 'OUT010'):
        Outlet_Identifier = 0,0,0,0,0,0,0,0,0
    elif (Outlet_Identifier == 'OUT013'):
        Outlet_Identifier = 1,0,0,0,0,0,0,0,0 
    elif (Outlet_Identifier == 'OUT017'):
        Outlet_Identifier = 0,1,0,0,0,0,0,0,0
    elif (Outlet_Identifier == 'OUT018'):
        Outlet_Identifier = 0,0,1,0,0,0,0,0,0
    elif (Outlet_Identifier == 'OUT019'):
        Outlet_Identifier = 0,0,0,1,0,0,0,0,0
    elif (Outlet_Identifier == 'OUT027'):
        Outlet_Identifier = 0,0,0,0,1,0,0,0,0
    elif (Outlet_Identifier == 'OUT035'):
        Outlet_Identifier = 0,0,0,0,0,1,0,0,0
    elif (Outlet_Identifier == 'OUT045'):
        Outlet_Identifier = 0,0,0,0,0,0,1,0,0                        
    elif (Outlet_Identifier == 'OUT046'):
        Outlet_Identifier = 0,0,0,0,0,0,0,1,0       
    else:
        Outlet_Identifier = 0,0,0,0,0,0,0,0,1

    Outlet_1, Outlet_2, Outlet_3, Outlet_4, Outlet_5, Outlet_6, Outlet_7, Outlet_8, Outlet_9 = Outlet_Identifier
    
    
    Outlet_Establishment_Year = int(2013 - int(request.form['Outlet Establishment Year']))

    Outlet_Size = request.form['Outlet Size']
    if (Outlet_Size == 'Medium'):
        Outlet_Size = 1,0
    elif (Outlet_Size == 'Small'):
        Outlet_Size = 0,1
    else:
        Outlet_Size = 0,0
        
    Outlet_Size_1, Outlet_Size_2 = Outlet_Size

    Outlet_Location_Type = request.form['Outlet Location Type']
    if (Outlet_Location_Type == 'Tier 2'):
        Outlet_Location_Type = 1,0
    elif (Outlet_Location_Type == 'Tier 3'):
        Outlet_Location_Type = 0,1
    else:
        Outlet_Location_Type = 0,0

    Outlet_Location_Type_1, Outlet_Location_Type_2 = Outlet_Location_Type    

    Outlet_Type = request.form['Outlet Type']
    if (Outlet_Type == 'Supermarket Type1'):
        Outlet_Type = 1,0,0
    elif (Outlet_Type == 'Grocery Store'):
        Outlet_Type = 0,0,0
    elif (Outlet_Type == 'Supermarket Type3'):
        Outlet_Type = 0,0,1
    else:
        Outlet_Type = 0,1,0

    Outlet_Type_1, Outlet_Type_2, Outlet_Type_3 = Outlet_Type   



    Item_Type = request.form['Item Type']
    
    if(Item_Type == 'Snack Foods'):
        Item_Type = 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    elif (Item_Type == 'Seafood'):
        Item_Type = 1,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    elif (Item_Type == 'Breads'):
        Item_Type = 0,1,0,0,0,0,0,0,0,0,0,0,0,0,0
    elif (Item_Type == 'Canned'):
        Item_Type = 0,0,1,0,0,0,0,0,0,0,0,0,0,0,0
    elif (Item_Type == 'Dairy'):
        Item_Type = 0,0,0,1,0,0,0,0,0,0,0,0,0,0,0
    elif (Item_Type == 'Baking Goods'):
        Item_Type = 0,0,0,0,1,0,0,0,0,0,0,0,0,0,0
    elif (Item_Type == 'Breakfast'):
        Item_Type = 0,0,0,0,0,1,0,0,0,0,0,0,0,0,0
    elif (Item_Type == 'Fruits and Vegetables'):
        Item_Type = 0,0,0,0,0,0,1,0,0,0,0,0,0,0,0
    elif (Item_Type == 'Frozen Foods'):
        Item_Type = 0,0,0,0,0,0,0,1,0,0,0,0,0,0,0
    elif (Item_Type == 'Health and Hygiene'):
        Item_Type = 0,0,0,0,0,0,0,0,1,0,0,0,0,0,0
    elif (Item_Type == 'Meat'):
        Item_Type = 0,0,0,0,0,0,0,0,0,1,0,0,0,0,0
    elif (Item_Type == 'Starchy Foods'):
        Item_Type = 0,0,0,0,0,0,0,0,0,0,1,0,0,0,0
    elif (Item_Type == 'Soft Drinks'):
        Item_Type = 0,0,0,0,0,0,0,0,0,0,0,1,0,0,0
    elif (Item_Type == 'Hard Drinks'):
        Item_Type = 0,0,0,0,0,0,0,0,0,0,0,0,1,0,0
    elif (Item_Type == 'Household'):
        Item_Type = 0,0,0,0,0,0,0,0,0,0,0,0,0,1,0
    else:
        Item_Type = 0,0,0,0,0,0,0,0,0,0,0,0,0,0,1

    Item_Type_1, Item_Type_2, Item_Type_3, Item_Type_4, Item_Type_5, Item_Type_6, Item_Type_7, Item_Type_8, Item_Type_9, Item_Type_10, Item_Type_11, Item_Type_12, Item_Type_13, Item_Type_14, Item_Type_15 = Item_Type

    input_data = [Item_Weight, Item_Visibility, Item_MRP, Outlet_Establishment_Year, Item_Fat_Content_1, Outlet_Location_Type_1, Outlet_Location_Type_2, Outlet_Size_1, Outlet_Size_2, Outlet_Type_1, Outlet_Type_2, Outlet_Type_3, Item_Type_1, Item_Type_2, Item_Type_3, Item_Type_4, Item_Type_5, Item_Type_6, Item_Type_7, Item_Type_8, Item_Type_9, Item_Type_10, Item_Type_11, Item_Type_12, Item_Type_13, Item_Type_14, Item_Type_15, Outlet_1, Outlet_2, Outlet_3, Outlet_4, Outlet_5, Outlet_6, Outlet_7, Outlet_8, Outlet_9]
    features_value = [np.array(input_data)]

    features_name = [['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Establishment_Year', 
                     'Item_Fat_Content_1', 'Outlet_Location_Type_1', 'Outlet_Location_Type_2', 
                     'Outlet_Size_1', 'Outlet_Size_2', 'Outlet_Type_1', 'Outlet_Type_2', 'Outlet_Type_3', 
                     'Item_Type_1', 'Item_Type_2', 'Item_Type_3', 'Item_Type_4', 'Item_Type_5', 
                     'Item_Type_6', 'Item_Type_7', 'Item_Type_8', 'Item_Type_9', 'Item_Type_10', 
                     'Item_Type_11', 'Item_Type_12', 'Item_Type_13', 'Item_Type_14', 'Item_Type_15', 
                     'Outlet_1', 'Outlet_2', 'Outlet_3', 'Outlet_4', 'Outlet_5', 'Outlet_6', 'Outlet_7', 'Outlet_8', 'Outlet_9']]

    df = pd.DataFrame(features_value, columns=features_name)

    prediction = model.predict(df)
    output = round(prediction[0], 2)
    
    return render_template('index.html', prediction_text = 'Expected Item Outlet Sales: {}'.format(output))    
    
    
if __name__ == '__main__':
    app.run(debug=True)
