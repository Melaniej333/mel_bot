# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside the container
EXPOSE 80

# Define environment variable for Flask
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Run the app when the container launches
CMD ["flask", "run"]
