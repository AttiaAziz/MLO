MLO – Credit Card Fraud Detection

This project demonstrates a complete end-to-end pipeline for credit card fraud detection, combining machine learning with a deployable API and dashboard.

Features:

Exploratory Data Analysis (EDA) on transaction data

Model training using Logistic Regression and Random Forest (with SMOTE for class balancing)

Fraud prediction served via FastAPI

Interactive front-end built with Streamlit

Deployment-ready and Docker-friendly structure

Project Structure:

Treatment.py: Data processing script

credit-card-fraud.ipynb: Main notebook with analysis and model training

credit-card-fraud.py: Equivalent Python script

main.py: FastAPI backend exposing /predict endpoint

fastAPI_Streamlit/: Streamlit dashboard and API client
