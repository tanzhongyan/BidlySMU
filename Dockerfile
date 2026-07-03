# =============================================================================
# BidlySMU Pipeline Docker Image
# =============================================================================
# Base: Ubuntu 24.04 LTS (noble) - supported until 2029
# Python: 3.12 (security support)
# Browser: Google Chrome + chromedriver for Selenium BOSS scraping
# =============================================================================

FROM ubuntu:24.04

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies: Python + Chrome + utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python
    python3.12 python3.12-venv python3-pip \
    # Build tools
    build-essential \
    # Network utilities
    wget curl ca-certificates gnupg \
    # Additional utilities
    jq \
    # Chrome dependencies (for Selenium headless)
    libglib2.0-0 libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 \
    libcups2 libdrm2 libdbus-1-3 libxkbcommon0 libxcomposite1 \
    libxdamage1 libxfixes3 libxrandr2 libgbm1 libpango-1.0-0 \
    libcairo2 libasound2t64 libx11-6 libxcb1 \
    fonts-liberation fonts-noto-color-emoji \
    && rm -rf /var/lib/apt/lists/*

# Install Google Chrome stable
RUN wget -q -O /tmp/chrome.deb https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb && \
    apt-get update && apt-get install -y --no-install-recommends /tmp/chrome.deb && \
    rm /tmp/chrome.deb && \
    rm -rf /var/lib/apt/lists/*

# Verify Chrome installed
RUN google-chrome --version

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

# Copy static input files needed at runtime
COPY script_input/classification_validation_results.csv \
     script_input/regression_median_validation_results.csv \
     script_input/regression_min_validation_results.csv \
     script_input/professor_lookup.csv \
     script_input/bidding_schedules.json \
     script_input/

# Environment variables
ENV PYTHONUTF8=1
ENV PYTHONIOENCODING=utf-8

# Entry point (strip CRLF from Windows Git, then make executable)
COPY scripts/run_pipeline.sh .
RUN sed -i 's/\r$//' run_pipeline.sh && chmod +x run_pipeline.sh

ENTRYPOINT ["./run_pipeline.sh"]
