# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch CPU version (faster and smaller)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy application files
COPY app.py .
COPY app_simple.py .
COPY test_setup.py .

# Copy documentation files (if they exist)
COPY README.md . 2>/dev/null || true
COPY SETUP_INSTRUCTIONS.md . 2>/dev/null || true

# Copy model file (if exists)
COPY lstm_model.pth . 2>/dev/null || true

# Expose Streamlit port
EXPOSE 8501

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Default command
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"] 