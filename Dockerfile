# Start from a Python base image
# FROM python:3.9-slim
FROM python:3.9

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

# Install any dependencies
RUN pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu
# RUN pip install torch==1.7.0+cpu torchvision==0.8.1+cpu torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install --no-deps pyannote.audio
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Expose the port that Gunicorn will run on
EXPOSE 8000

# Start Gunicorn with a configuration suitable for production
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "--timeout", "740", "app:app"]