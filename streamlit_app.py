import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("models/model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

st.title("Spam Message Classifier")

st.write("Enter a message to check if it is spam or not.")

message = st.text_area("Message")

if st.button("Predict"):

    if message.strip() == "":
        st.warning("Please enter a message.")
    else:

        transformed = vectorizer.transform([message])

        prediction = model.predict(transformed)[0]

        if prediction == 1:
            st.error(" Spam Message")
        else:
            st.success(" Not Spam (Ham)")