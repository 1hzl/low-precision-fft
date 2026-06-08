# Independence test: simulate a clean user environment
# Tests the full flow: pip install → build → import → pytest
FROM python:3.10-slim

# System build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU (simulates user without CUDA GPU)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Create working directory
WORKDIR /test

# Copy test script
COPY test.sh /test/test.sh
RUN chmod +x /test/test.sh

ENTRYPOINT ["/bin/bash", "/test/test.sh"]
