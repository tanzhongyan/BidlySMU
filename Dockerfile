FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3-pip \
    build-essential \
    wget curl ca-certificates gnupg \
    jq \
    libglib2.0-0 libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 \
    libcups2 libdrm2 libdbus-1-3 libxkbcommon0 libxcomposite1 \
    libxdamage1 libxfixes3 libxrandr2 libgbm1 libpango-1.0-0 \
    libcairo2 libasound2t64 libx11-6 libxcb1 \
    fonts-liberation fonts-noto-color-emoji \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q -O /tmp/chrome.deb https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb && \
    apt-get update && apt-get install -y --no-install-recommends /tmp/chrome.deb && \
    rm /tmp/chrome.deb && \
    rm -rf /var/lib/apt/lists/*

RUN google-chrome --version

WORKDIR /app

RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY models/ ./models/

RUN mkdir -p script_input script_output logs db_cache

COPY data/ ./data/

COPY script_input/classification_validation_results.csv \
     script_input/regression_median_validation_results.csv \
     script_input/regression_min_validation_results.csv \
     script_input/

RUN cp data/professor_lookup.csv script_input/professor_lookup.csv

ENV PYTHONUTF8=1
ENV PYTHONIOENCODING=utf-8

COPY scripts/run_pipeline.sh .
RUN sed -i 's/\r$//' run_pipeline.sh && chmod +x run_pipeline.sh

RUN useradd --create-home appuser \
    && chown -R appuser:appuser /app \
    && chmod -R u+rwX /app

USER appuser

ENTRYPOINT ["./run_pipeline.sh"]
