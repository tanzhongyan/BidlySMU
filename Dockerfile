# =============================================================================
# BidlySMU Pipeline Docker Image (Lightweight)
# =============================================================================
# Base: Ubuntu 24.04 LTS (noble) - supported until 2029
# Python: 3.12 (security support)
# No browser required - uses Truba JSON API for calendar data
# =============================================================================

FROM ubuntu:24.04

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python
    python3.12 python3.12-venv python3-pip \
    # Build tools
    build-essential \
    # Network utilities
    wget curl ca-certificates \
    # Additional utilities
    jq \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create virtual environment
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY models/ ./models/

# Create required directories
RUN mkdir -p script_input script_output logs db_cache

# Environment variables
ENV PYTHONUTF8=1
ENV PYTHONIOENCODING=utf-8

# Entry point
COPY scripts/run_pipeline.sh .
RUN chmod +x run_pipeline.sh

ENTRYPOINT ["./run_pipeline.sh"]
