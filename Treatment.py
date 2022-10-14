import requests
import pandas as pd
import  streamlit as st
from plotly import graph_objs as go
URL="http://localhost:8000"

def getPredict(amt,zip,lat,long,merch_lat,merch_long,age,gender,category,date):
    params = {
    'amt': amt,
    'zip': zip,
    'lat': lat,
    'long': long,
    'merch_lat': merch_lat,
    'merch_long':merch_long,
    'age':age,
    'gender':gender,
    'category':category,
    'date':date
}
    res = requests.post(f"{URL}/predict",params=params)

    
    return res