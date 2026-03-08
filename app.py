from fastapi import FastAPI
import pickle
import logging


model = pickle.load(open("models/model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

app = FastAPI(title="Spam Classifier API")

@app.get("/")
def home():
    return {"message": "Spam Classifier API is running"}

@app.post("/predict")
def predict(message: str):

    transformed_message = vectorizer.transform([message])

    prediction = model.predict(transformed_message)[0]

    if prediction == 1:
        result = "spam"
    else:
        result = "ham"

    return {
        "message": message,
        "prediction": result
    }