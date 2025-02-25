FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Upgrade pip
RUN pip install --upgrade pip

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# CREATE A experiments DIRECTORY
RUN mkdir experiments

# Install Supervisord
RUN apt-get update && apt-get install -y supervisor

# Copy the current directory contents into the container
COPY . .

# Copy Supervisord configuration file
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose the FastAPI port
EXPOSE 8001

# Expose the MLflow UI port
EXPOSE 5000

# environment variables
ENV USE_SMALL_DATASET="True"

# Run Supervisord
CMD ["supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]