# Start from a Python base image
FROM python:3.9-slim

# Start from a Miniconda base image
# FROM continuumio/miniconda3

# Set environment variables - Ensures python output is sent straight to the terminal without buffering
ENV PYTHONUNBUFFERED True

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt /app/

# Install PyTorch, torchvision, torchaudio, and CPU only using conda
# RUN conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 cpuonly -c pytorch

# Install gunicorn
RUN pip install gunicorn
RUN echo $PATH
RUN which gunicorn || echo "gunicorn not found"

# Install any dependencies (no deps too)
# RUN pip install --no-deps pyannote.audio
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Expose the port that Gunicorn will run on
EXPOSE 8000

# Start Gunicorn with a configuration suitable for production s
# CMD ["gunicorn", "--worker-class=gevent", "--workers=3", "--worker-connections=1000", "--bind=0.0.0.0:8000", "--timeout", "740", "app:app"]
CMD ["gunicorn", "--worker-class=gevent", "--workers=2", "--worker-connections=500", "--bind=0.0.0.0:8000", "--timeout", "1500", "--max-requests", "100", "app:app"]
