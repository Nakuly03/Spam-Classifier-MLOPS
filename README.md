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
=======
# Spam Classifier MLOps

A production-ready SMS spam classification system demonstrating the full machine learning lifecycle — from experiment tracking and model training to containerized API deployment and CI automation.

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi)](http://localhost:8000/docs)
[![MLflow](https://img.shields.io/badge/Tracking-MLflow-0194E2?logo=mlflow)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Container-Docker-2496ED?logo=docker)](https://docker.com)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?logo=githubactions)](https://github.com/Nakuly03/Spam-Classifier-MLOPS/actions)

---

## Overview

This project classifies SMS messages as spam or ham (not spam) using a Random Forest classifier trained on 5,572 messages. The focus is not just on model accuracy but on building the infrastructure around the model — experiment tracking, a REST API, a web interface, containerization, and automated CI.

---

## Model Performance

| Model | Accuracy | F1 (Spam) |
|-------|----------|-----------|
| Random Forest | **97.8%** | **0.91** |
| SVM | 97.4% | 0.89 |
| Logistic Regression | 96.9% | 0.87 |
| Naive Bayes | 95.1% | 0.83 |

Evaluated on 1,115 test samples (80-20 split). Random Forest achieved 5 false positives and 21 false negatives on the test set.

---

## Architecture

```
User
  │
  ▼
Streamlit UI (streamlit_app.py)
  │
  ▼
FastAPI Server (app.py)
  │  Loads trained model artifact
  ▼
ML Pipeline
  ├── Text Preprocessing   (src/pre_processing.py)
  ├── Feature Engineering  (src/feature_engineering.py)
  └── Random Forest Model  (src/model_training.py)
  │
  ▼
MLflow Tracking Server
  └── Logs params, metrics, and model artifacts per run
```

---

## ML Pipeline

**Preprocessing (8 stages):**
- HTML and URL removal
- Punctuation cleaning
- Slang expansion
- Lowercasing
- Tokenization
- Stopword removal
- Porter stemming
- Whitespace normalization

**Feature Engineering:**
- Bag-of-Words with bigrams (3,000 features)
- Strict train-test split to prevent data leakage

**Experiment Tracking:**
- All runs logged to MLflow with parameters, metrics, and model artifacts
- Four algorithms benchmarked and compared in the MLflow UI
>>>>>>> 457d2a3 (docs: rewrite README)

---

## Project Structure

<<<<<<< HEAD
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
=======
```
Spam-Classifier-MLOPS/
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI pipeline
├── experiments/
│   └── spam_classifier.ipynb   # Exploratory analysis and benchmarking
├── src/
│   ├── data_ingestion.py       # Data loading and validation
│   ├── pre_processing.py       # 8-stage text cleaning pipeline
│   ├── feature_engineering.py  # BoW feature extraction
│   ├── model_training.py       # Model training with MLflow logging
│   └── model_evaluation.py     # Metrics and evaluation reporting
├── app.py                      # FastAPI prediction server
├── streamlit_app.py            # Streamlit web interface
├── main.py                     # Training pipeline entrypoint
├── Dockerfile                  # Container definition
├── requirements.txt
└── .gitignore
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/Nakuly03/Spam-Classifier-MLOPS
cd Spam-Classifier-MLOPS
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Train the model

```bash
python main.py
```

MLflow will log all experiment runs. View them at:
```bash
mlflow ui
# Open http://localhost:5000
```

### 3. Run the API

```bash
uvicorn app:app --reload
# Swagger docs at http://localhost:8000/docs
```

### 4. Run the Streamlit UI

```bash
streamlit run streamlit_app.py
```

### 5. Run with Docker

```bash
docker build -t spam-classifier .
docker run -p 8000:8000 spam-classifier
```

---

## API Reference

### POST /predict

Classify a message as spam or ham.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"message": "Congratulations! You have won a free iPhone. Click here to claim."}'
```

```json
{
  "message": "Congratulations! You have won a free iPhone. Click here to claim.",
  "prediction": "spam",
  "confidence": 0.94
}
```

### GET /health

```json
{"status": "ok"}
```

---

## CI Pipeline

GitHub Actions runs on every push to main:
- Install dependencies
- Run training pipeline
- Run tests

View runs at: https://github.com/Nakuly03/Spam-Classifier-MLOPS/actions

---

## Dataset

SMS Spam Collection Dataset — 5,572 messages (4,825 ham, 747 spam).
Source: UCI Machine Learning Repository.

---

## Tech Stack

Python · scikit-learn · FastAPI · Streamlit · MLflow · Docker · GitHub Actions · NLTK
>>>>>>> 457d2a3 (docs: rewrite README)
