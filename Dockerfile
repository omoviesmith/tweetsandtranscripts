# Start from a Python base image
FROM python:3.10-slim

# Set environment variables - Ensures python output is sent straight to the terminal without buffering
ENV PYTHONUNBUFFERED True

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt /app/

# Install system critical dependencies first
RUN pip install --no-cache-dir triton==2.0.0

# Install any dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Expose the port that Gunicorn will run on
EXPOSE 8000

# Start Gunicorn with a configuration suitable for production
# "-w" is the number of worker processes for handling requests
# "-b" is the bind address and port number
# "app:app" tells Gunicorn where the WSGI application object is
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "app:app"]