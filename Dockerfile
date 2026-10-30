# Image de production : code + modèle validé par le quality gate.
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PROJECT_ROOT=/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY params.yaml .
COPY src ./src
# Le modèle doit exister : lancer `python -m credit_risk.train` avant le build (la CI le fait).
COPY models/model.joblib models/metadata.json ./models/

RUN useradd --create-home appuser && mkdir -p data reports && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "credit_risk.serve:app", "--host", "0.0.0.0", "--port", "8000"]
