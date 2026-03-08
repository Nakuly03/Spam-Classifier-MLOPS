# Spam Classifier MLOps Project

## Overview

This project demonstrates a complete end-to-end Machine Learning Operations (MLOps) workflow for building, tracking, serving, and deploying a spam message classification system. The objective of the project is to classify SMS messages as either "Spam" or "Ham" (Not Spam) using machine learning techniques while following industry practices for building and deploying machine learning systems.

The project implements the full machine learning lifecycle including data preprocessing, feature engineering, model training, experiment tracking, API deployment, containerization, continuous integration, and web-based interaction with the trained model.

The system is designed to demonstrate how machine learning models can be developed and deployed in a production-ready environment.

---

## Problem Statement

Spam messages are unwanted messages sent in bulk to users. Detecting spam messages is important for improving communication systems and protecting users from scams or unwanted advertisements.

The goal of this project is to build a machine learning model that can automatically classify messages as spam or not spam based on the text content of the message.

---

## Project Objectives

The main objectives of this project are:

- Build a machine learning model to classify SMS messages.
- Create a modular machine learning pipeline for training and evaluation.
- Track experiments and model performance using MLflow.
- Serve the trained model through an API using FastAPI.
- Provide a user interface for interacting with the model using Streamlit.
- Containerize the entire application using Docker.
- Automate testing and building using GitHub Actions.
- Deploy the system on cloud platforms for public access.

---

## Machine Learning Experiment

Built a complete text classification pipeline on 5,572 SMS messages with an 8-stage preprocessing workflow including HTML/URL removal, punctuation cleaning, slang expansion, stopword removal, and Porter stemming.

Generated 3,000 Bag-of-Words features with bi-grams and benchmarked four classification algorithms (Naive Bayes, Logistic Regression, Support Vector Machine, Random Forest) using an 80–20 train-test split (1,115 test samples).

Achieved 97.8% accuracy and 0.91 F1-score for the spam class using Random Forest, outperforming Logistic Regression (96.9%) and remaining competitive with SVM (97.4%).

Reduced misclassification to 21 false negatives and 5 false positives through feature vector tuning, controlled preprocessing pipelines, and strict data-leakage prevention.

---


## System Architecture

The architecture of the project consists of several layers:

User Interface (Streamlit)  
→ API Layer (FastAPI)  
→ Machine Learning Model  
→ Experiment Tracking (MLflow)

Users interact with the application through a Streamlit web interface. The Streamlit application sends requests to the FastAPI server. The FastAPI server loads the trained machine learning model and returns predictions. MLflow is used to track model training experiments, parameters, and evaluation metrics.

---

## Project Structure

Spam-Classifier-MLOPS
│
├── .github/workflows
│ └── ci.yml
│
├── experiments
│ └── spam_classifier.ipynb
│
├── src
│ ├── data_ingestion.py
│ ├── pre_processing.py
│ ├── feature_engineering.py
│ ├── model_training.py
│ └── model_evaluation.py
│
├── app.py
├── streamlit_app.py
├── main.py
├── Dockerfile
├── requirements.txt
├── README.md
└── .gitignore
