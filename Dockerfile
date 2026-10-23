FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install runtime dependencies + temporary build tools for compiling audio extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libportaudio2 \
    libasound2 \
    espeak-ng \
    curl \
    build-essential \
    portaudio19-dev \
    libasound2-dev \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU first (drops image from 7.7GB to ~1.4GB by omitting 5.3GB of NVIDIA CUDA runtime wheels)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download Whisper STT (461MB) and Silero VAD into image cache so the container starts in 0 seconds with zero runtime downloads
RUN python -c "import whisper; whisper.load_model('small'); import torch; torch.hub.load('snakers4/silero-vad', 'silero_vad', onnx=False)"

# Purge build-only tools to keep final image clean and small
RUN apt-get purge -y --auto-remove build-essential portaudio19-dev libasound2-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy application source
COPY . .

COPY ssl/fullchain.pem /app/fullchain.pem
COPY ssl/privkey.pem /app/privkey.pem


# Expose ports: 5008 (Streamlit Web UI), 8000 (FastAPI / WebSocket Server)
EXPOSE 5008

CMD ["sh", "-c", "if [ -s /app/fullchain.pem ] && [ -s /app/privkey.pem ]; then streamlit run app.py --server.address=0.0.0.0 --server.port=5008 --server.sslCertFile=/app/fullchain.pem --server.sslKeyFile=/app/privkey.pem --server.fileWatcherType=none; else streamlit run app.py --server.address=0.0.0.0 --server.port=5008 --server.fileWatcherType=none; fi"]
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5015", "--ssl-certfile", "/app/fullchain.pem", "--ssl-keyfile", "/app/privkey.pem"]
