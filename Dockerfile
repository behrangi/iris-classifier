# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any dependencies
RUN pip install --no-cache-dir fastapi uvicorn scikit-learn

# Expose the port the app runs on
EXPOSE 8000

# Run the application
CMD ["uvicorn", "iris_app:app", "--host", "0.0.0.0", "--port", "8000"]