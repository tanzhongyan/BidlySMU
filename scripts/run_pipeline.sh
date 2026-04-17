#!/bin/bash

# Force UTF-8 encoding for all Python processes
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

# ==============================================================================
# SMU Bidding Data Pipeline Orchestrator
# ==============================================================================
# This script runs the data processing pipeline (Steps 2 & 3).
# Step 1 (scraping) is commented out as it requires Chrome/chromedriver.
#
# Execution Flow:
# 1. Step 1 (scraping): DISABLED - requires Chrome
# 2. Step 2 runs: TableBuilderCoordinator (professors, courses, classes, timings, bid windows)
# 3. Step 3 runs: BidPredictorCoordinator (predictions and database upserts)
#
# All output is redirected to timestamped log files in the 'logs/' directory.
# If any step fails, the script will exit immediately.
#
# Coordinators:
# - TableBuilderCoordinator in src/pipeline/table_builder.py
# - BidPredictorCoordinator in src/pipeline/bid_predictor.py
#
# Note: Step 1 (scraping) requires Chrome/chromedriver and is disabled by default.
# To enable, uncomment the Step 1 section below.
# ==============================================================================

# --- Setup ---
mkdir -p logs
mkdir -p script_output

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "============================================================"
echo "🚀 Starting SMU Data Pipeline at $(date)"
echo "============================================================"

# --- Step 1: Scraping (DISABLED - requires Chrome) ---
# The following Step 1 requires Chrome/chromedriver which may not be available.
# To enable, ensure Chrome is installed and uncomment the section below.
#
# Stream A: class_scraper.py (1a) -> html_data_extractor.py (1b)
# Stream B: overall_results_scraper.py (1c)
#
# (
#     echo "[Stream A] Running class_scraper.py (1a)..."
#     python -m src.scraper.class_scraper && \
#     echo "[Stream A] Running html_data_extractor.py (1b)..." && \
#     python -m src.scraper.html_data_extractor
# ) > logs/step_1ab_scrape_and_extract_${TIMESTAMP}.log 2>&1 &
# PID_A=$!
#
# (
#     echo "[Stream B] Running overall_results_scraper.py (1c)..."
#     python -m src.scraper.overall_results_scraper
# ) > logs/step_1c_scrape_overall_${TIMESTAMP}.log 2>&1 &
# PID_B=$!
#
# wait $PID_A
# CODE_A=$?
# wait $PID_B
# CODE_B=$?
#
# if [ $CODE_A -ne 0 ] || [ $CODE_B -ne 0 ]; then
#     echo "❌ ERROR: Step 1 (scraping) failed. Halting pipeline."
#     exit 1
# fi
# echo "✅ Step 1 (scraping) completed."

echo "⚠️ Step 1 (scraping) is DISABLED - requires Chrome/chromedriver"
echo "   Raw data must already exist in script_input/"
echo "------------------------------------------------------------"


# --- Step 2: Table Building (Direct Coordinator Call) ---
echo " Kicking off Step 2: TableBuilderCoordinator..."
python -c "
import sys
from src.config import BIDDING_SCHEDULES, START_AY_TERM
from src.pipeline.table_builder import TableBuilderCoordinator, TableBuilderConfig
from src.logging.logger import get_logger

logger = get_logger(__name__)
config = TableBuilderConfig.from_env(
    bidding_schedules=BIDDING_SCHEDULES,
    start_ay_term=START_AY_TERM
)
coordinator = TableBuilderCoordinator(config=config, logger=logger)
coordinator.run()
" > logs/step_2_TableBuilder_${TIMESTAMP}.log 2>&1

if [ $? -ne 0 ]; then
    echo "❌ ERROR: TableBuilderCoordinator failed. Halting pipeline."
    echo "   - Check logs/step_2_TableBuilder_${TIMESTAMP}.log for details."
    exit 1
fi

echo "✅ Step 2 completed successfully."
echo "------------------------------------------------------------"


# --- Step 3: Bid Prediction (Direct Coordinator Call) ---
echo " Kicking off Step 3: BidPredictorCoordinator..."
python -c "
import sys
from src.config import BIDDING_SCHEDULES, START_AY_TERM
from src.pipeline.bid_predictor import BidPredictorCoordinator, BidPredictorConfig
from src.logging.logger import get_logger

logger = get_logger(__name__)
config = BidPredictorConfig.from_env(
    bidding_schedules=BIDDING_SCHEDULES,
    start_ay_term=START_AY_TERM
)
coordinator = BidPredictorCoordinator(config=config, logger=logger)
coordinator.run()
" > logs/step_3_BidPrediction_${TIMESTAMP}.log 2>&1

if [ $? -ne 0 ]; then
    echo "❌ ERROR: BidPredictorCoordinator failed. Halting pipeline."
    echo "   - Check logs/step_3_BidPrediction_${TIMESTAMP}.log for details."
    exit 1
fi

echo "✅ Step 3 completed successfully."
echo "============================================================"
echo "🎉 SMU Data Pipeline finished successfully at $(date)"
echo "============================================================"