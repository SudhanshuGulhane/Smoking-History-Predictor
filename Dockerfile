# Use official lightweight Python image
FROM python:3.12.5

# Set working directory inside the container
WORKDIR /app

# Copy the entire current directory into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the API port
EXPOSE 4000

# Run the Flask app
CMD ["python", "app.py"]