FROM rayproject/ray:2.35.0-py310-gpu

USER root

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/home/ray/anaconda3/bin:$PATH" \
    PYTHONPATH="/workspace:$PYTHONPATH"

WORKDIR /workspace

# Install minimal system requirements
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to maintain Docker cache layer
COPY requirements.txt /workspace/requirements.txt

# Install lightweight python dependencies
RUN pip install --no-cache-dir -r /workspace/requirements.txt
RUN pip install "click==8.2.1"
# Copy source repository
COPY . /workspace

# Install NexusCache package without failing on GPU environment checks during build
ENV NO_CUDA_EXT=1
RUN pip install --no-deps --no-build-isolation -e .

EXPOSE 6379 8265 10001 8000 9090

CMD ["python3", "-m", "nexuscache.server.main"]