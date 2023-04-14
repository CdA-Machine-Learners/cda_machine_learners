# Use the official Python 3.9 image as the base image
FROM python:3.9

# Installing PyTorch takes a loooong time. Like more than 100 seconds, I could watch a fireship video and come back and it would still be installing.
# Docker caches each step in the Docker file, and only rebuilds the steps that have changed.
# So if we install PyTorch first, then change the requirements file, it will only have to rebuild the requirements file, not PyTorch.
RUN pip install torch==2.0.0 transformers==4.27.4

# Download the model files
ENV TRANSFORMERS_CACHE=/app/models
# Create a subdirectory for scripts
RUN mkdir -p /app/scripts
# Copy the download_models.py script
COPY src/download_models.py /app/scripts
# Download the GPT-2 model
RUN python /app/scripts/download_models.py


# Copy the requirements file into the container
COPY requirements.txt .

# Install any required packages
RUN pip install --no-cache-dir -r requirements.txt

# Add entrypoint scripts
ADD /docker/entrypoints /entrypoints
RUN chmod -R +x /entrypoints

# Set the working directory
WORKDIR /app
# Copy the rest of the application code
COPY src .