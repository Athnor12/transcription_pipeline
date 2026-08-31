FROM python:3.11-slim

# Installation de FFmpeg
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Installation de faster-whisper et des dépendances CUDA/cuDNN
RUN pip install --no-cache-dir faster-whisper nvidia-cublas-cu12 nvidia-cudnn-cu12

# Indiquer au système où trouver les bibliothèques CUDA installées par pip
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

COPY main.py .

CMD ["python", "-u", "main.py"]
