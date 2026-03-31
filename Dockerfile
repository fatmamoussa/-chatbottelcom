FROM python:3.9-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download fr_core_news_sm

# Copier tout le code source
COPY datasets/modele_scratch/ .

# Créer le dossier datasets pour les CSV
RUN mkdir -p datasets modele_scratch mlflow_logs logs

EXPOSE 5005

CMD ["python", "api.py"]