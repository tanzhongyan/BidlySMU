# Lambda container image for monthly scheduler
# Uses Truba JSON API to fetch BOSS bidding events (no web scraping)
# Build from project root:
#   docker build -t <tag> -f lambda/monthly_scheduler/Dockerfile .

FROM public.ecr.aws/lambda/python:3.12

# Install Python dependencies
COPY lambda/monthly_scheduler/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create app directory structure
RUN mkdir -p /app/src/scraper /app/src/logging

# Copy source files needed by Lambda
COPY src/scraper/trumba_client.py /app/src/scraper/trumba_client.py
COPY src/logging/logger.py /app/src/logging/logger.py
COPY src/config.py /app/src/config.py

# Copy Lambda function
COPY lambda/monthly_scheduler/lambda_function.py ./lambda_function.py

# Set Lambda handler
CMD ["lambda_function.lambda_handler"]