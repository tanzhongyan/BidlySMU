#!/bin/bash

# Force UTF-8 encoding for all Python processes
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

# ==============================================================================
# SMU Bidding Data Pipeline Orchestrator
# ==============================================================================
# This script runs the data processing pipeline (Step 2 only).
# Step 1 (scraping) is commented out as it requires Chrome/chromedriver.
#
# Execution Flow:
# 1. Step 1 (scraping): DISABLED - requires Chrome
# 2. Step 2 runs: PipelineCoordinator
#    - Phase 1: acad_term, courses, professors, bid_windows
#    - Phase 2: classes, timings, availability, bid_results
#    - Phase 3: bid_predictions (with safety_factors)
#
# All output is redirected to timestamped log files in the 'logs/' directory.
# If any step fails, the script will exit immediately.
#
# Coordinator:
# - PipelineCoordinator in src/pipeline/pipeline_coordinator.py
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
echo " Kicking off Step 2: PipelineCoordinator..."

# Generate log filename via Python (combines ACAD_TERM_ID and window code)
LOG_FILENAME=$(python -c "
from src.config import ACAD_TERM_ID, CURRENT_WINDOW_NAME
import re

def window_to_code(name):
    if not name:
        return 'UNKNOWN'
    m = re.search(r'Round\s+(\d+)([A-C]?)\s+Window\s+(\d+)', name, re.IGNORECASE)
    if m:
        return f'R{m.group(1)}{m.group(2)}W{m.group(3)}'
    m = re.search(r'[Rr]nd\s+(\d+)([A-C]?)\s+[Ww]in\s+(\d+)', name)
    if m:
        return f'R{m.group(1)}{m.group(2)}W{m.group(3)}'
    m = re.search(r'Incoming\s+(Freshmen|Exchange)', name, re.IGNORECASE)
    if m:
        suffix = 'F' if m.group(1).lower() == 'freshmen' else ''
        m2 = re.search(r'Rnd\s+(\d+)', name)
        m3 = re.search(r'Win\s+(\d+)', name)
        if m2 and m3:
            return f'R{m2.group(1)}{suffix}W{m3.group(1)}'
    return 'UNKNOWN'

from datetime import datetime
ts = datetime.now().strftime('%Y%m%d_%H%M%S')
wc = window_to_code(CURRENT_WINDOW_NAME)
print(f'{ACAD_TERM_ID}_{wc}_{ts}.log')
")

python -c "
import sys
from src.config import BIDDING_SCHEDULES, START_AY_TERM, DB_CONFIG, PipelineConfig
from src.pipeline.pipeline_coordinator import PipelineCoordinator

config = PipelineConfig.from_env(
    bidding_schedules=BIDDING_SCHEDULES,
    start_ay_term=START_AY_TERM,
    db_config=DB_CONFIG
)
coordinator = PipelineCoordinator(config=config)
coordinator.run()
" 2>&1 | tee "script_output/${LOG_FILENAME}"

if [ $? -ne 0 ]; then
    echo "❌ ERROR: PipelineCoordinator failed. Halting pipeline."
    echo "   - Check script_output/${LOG_FILENAME} for details."
    exit 1
fi

echo "✅ Step 2 completed successfully."
echo "============================================================"
echo "🎉 SMU Data Pipeline finished successfully at $(date)"
echo "============================================================"