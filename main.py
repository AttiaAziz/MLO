from tokenize import String
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pickle
import sklearn
from fastapi import FastAPI, File, UploadFile
import uvicorn
import sys  
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI()

with open('../notebooks/best_model.pkl' , 'rb') as f:
    model = pickle.load(f) 




    
@app.post("/predict")
def return_predictions(amt: float,gender:float,zip:float,lat:float,long:float,merch_lat:float,merch_long:float,age:float,category:str,date:datetime.datetime):
   
    year = date.year
    month= date.month
    day = date.day
    # Extract hour,minute and second
    hour = date.hour
    minute =date.minute
    sec= date.second
    dataframe= pd.DataFrame()
    dataframe = dataframe.append({'amt': amt, 'gender': gender, 'zip': zip ,'lat':lat, 'long':long,
'merch_lat':merch_lat,
'merch_long':merch_long,
'year':year,
'month':month,
'day':day,
'hour':hour,
'sec': sec,
'age':age,
'category_entertainment':'category_entertainment'==category,
'category_food_dining':'category_food_dining'==category,
'category_gas_transport':'category_gas_transport'==category,
'category_grocery_net':'category_grocery_net'==category,
'category_grocery_pos':'category_grocery_pos'==category,
'category_health_fitness':'category_health_fitness'==category,
'category_home':'category_home'==category,
'category_kids_pets':'category_kids_pets'==category,
'category_misc_net':'category_misc_net'==category,
'category_misc_pos':'category_misc_pos'==category,
'category_personal_care':'category_personal_care'==category,
'category_shopping_net':'category_shopping_net'==category,
'category_shopping_pos':'category_shopping_pos'==category,
'category_travel':'category_travel'==category,
}, ignore_index=True)
    predictions = model.predict(dataframe)
    return {"predictions": predictions.tolist()}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)


