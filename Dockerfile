# Use a lightweight Python image
FROM python:3.9-slim

# Set working directory in the container
WORKDIR /app

# Copy the local files into the container
COPY . /app/
#COPY app.py /app/app.py

# Install required Python packages
RUN pip install -r requirements.txt

# Expose the application port
EXPOSE 5001

# Run the Flask application
CMD ["python", "/app/app.py"]