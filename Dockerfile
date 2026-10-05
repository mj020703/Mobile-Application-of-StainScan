# Use a lightweight, official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=5000

# Install system dependencies (build-essential and libgomp1 are recommended for PyTorch)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the server file and model weights
COPY server.py .
COPY stainscan_model_v3_balanced.h5 .
COPY stain_model.h5 .
COPY stainscan_model.pth .

# Expose port (default 5000)
EXPOSE 5000

# Run the Flask app with Gunicorn. 
# We use 1 worker and multiple threads to minimize RAM usage (under 512MB for free tiers) 
# and prevent duplicate model loading into memory.
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 120 server:app"]
