# Dockerfile — ETA Predictor serving API
# Base image: slim Python to keep image size small
FROM python:3.11-slim

# Install only required system dependency
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first — Docker caches this layer
COPY requirements.txt .
RUN pip install --no-cache-dir fastapi uvicorn scikit-learn xgboost joblib pydantic numpy pandas

# Copy application code and saved artifacts
COPY serving/ ./serving/
COPY artifacts/ ./artifacts/

# Non-root user for security
RUN useradd -m mluser && chown -R mluser:mluser /app
USER mluser

EXPOSE 8000

CMD ["uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
