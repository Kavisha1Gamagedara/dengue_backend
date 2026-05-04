# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend files into the container at /app
COPY . .

# Make port 10000 available to the world outside this container
EXPOSE 10000

# Run gunicorn when the container launches
# Binding to 0.0.0.0:10000 which is standard for Render
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app", "--timeout", "90"]
