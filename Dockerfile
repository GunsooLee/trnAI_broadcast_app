# --- Dockerfile for Home Shopping Recommender ---
# Base lightweight python image (includes pip)
FROM python:3.11-slim

# Install system packages needed for psycopg2 and build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy project files and requirements first for layered caching
COPY requirements*.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app

# Expose Streamlit default port
EXPOSE 8501

# Default command: run Streamlit app
CMD ["streamlit", "run", "streamlit_app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
