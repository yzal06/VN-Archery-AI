FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies required for OpenCV / Ultralytics in headless containers
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first to keep image lightweight, then remaining requirements
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ ./src/
COPY web/ ./web/
COPY main_demo.py webcam_yolo_demo.py README.md ./

EXPOSE 8000

# Run baseline pipeline verification, then serve the Web ONNX app on port 8000
CMD ["sh", "-c", "python main_demo.py && echo '=== Web ONNX Server running at http://localhost:8000 ===' && python -m http.server 8000 --directory web --bind 0.0.0.0"]
