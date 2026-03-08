FROM python:3.10

WORKDIR /app


COPY . /app


RUN pip install --no-cache-dir -r requirements.txt


RUN python -m nltk.downloader stopwords


EXPOSE 8000
EXPOSE 8501


CMD bash -c "uvicorn app:app --host 0.0.0.0 --port 8000 & streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0"