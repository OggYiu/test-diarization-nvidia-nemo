# Multi-stage Dockerfile for Diarization API
FROM python:3.10-slim AS base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    ffmpeg \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY app/requirements.txt .
COPY audio_pipeline.py .
COPY app/main.py .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for the FastAPI app
EXPOSE 8000

# Create working directory for runtime artifacts
ENV PYTHONUNBUFFERED=1
ENV WORK_DIR=/app/work
RUN mkdir -p ${WORK_DIR}

# Run using uvicorn
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]