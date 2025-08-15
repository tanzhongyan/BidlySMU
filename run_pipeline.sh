#!/bin/bash

# Force UTF-8 encoding for all Python processes
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

# ==============================================================================
# SMU Bidding Data Pipeline Orchestrator
# ==============================================================================
# This script runs the entire data scraping and prediction pipeline in the
# correct order, with parallel processing for Step 1.
#
# Execution Flow:
# 1. (1a -> 1b) runs in parallel with (1c).
# 2. The script waits for all of Step 1 to complete.
# 3. Step 2 runs sequentially.
# 4. Step 3 runs sequentially.
#
# All output is redirected to timestamped log files in the 'logs/' directory.
# If any step fails, the script will exit immediately.
# ==============================================================================

# --- Setup ---
# Create a logs directory if it doesn't exist
mkdir -p logs

# Generate a single timestamp for this entire run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "============================================================"
echo "🚀 Starting SMU Data Pipeline at $(date)"
echo "============================================================"

# --- Step 1: Parallel Scraping ---
echo " Kicking off Step 1 in parallel..."

# Stream A: Run 1a, and if it succeeds, run 1b.
# The output of both is combined into a single log file.
(
    echo "[Stream A] Running step_1a_BOSSClassScraper.py..."
    python step_1a_BOSSClassScraper.py && \
    echo "[Stream A] Running step_1b_HTMLDataExtractor.py..." && \
    python step_1b_HTMLDataExtractor.py
) > logs/step_1ab_scrape_and_extract_${TIMESTAMP}.log 2>&1 &
PID_A=$! # Get the Process ID for Stream A

# Stream B: Run 1c independently.
(
    echo "[Stream B] Running step_1c_ScrapeOverallResults.py..."
    python step_1c_ScrapeOverallResults.py
) > logs/step_1c_scrape_overall_${TIMESTAMP}.log 2>&1 &
PID_B=$! # Get the Process ID for Stream B

# --- Wait for parallel jobs to finish and check for errors ---
echo " > Waiting for Stream A (PID: $PID_A) and Stream B (PID: $PID_B) to complete..."

wait $PID_A
CODE_A=$? # Get the exit code of Stream A

wait $PID_B
CODE_B=$? # Get the exit code of Stream B

if [ $CODE_A -ne 0 ] || [ $CODE_B -ne 0 ]; then
    echo "❌ ERROR: A script in Step 1 failed. Halting pipeline."
    echo "   - Stream A (1a -> 1b) exit code: $CODE_A"
    echo "   - Stream B (1c) exit code: $CODE_B"
    echo "   - Check the log files in the 'logs/' directory for details."
    exit 1
fi

echo "✅ Step 1 completed successfully."
echo "------------------------------------------------------------"


# --- Step 2: Table Building ---
echo " Kicking off Step 2: TableBuilder..."
python step_2_TableBuilder.py > logs/step_2_TableBuilder_${TIMESTAMP}.log 2>&1

if [ $? -ne 0 ]; then
    echo "❌ ERROR: step_2_TableBuilder.py failed. Halting pipeline."
    echo "   - Check logs/step_2_TableBuilder_${TIMESTAMP}.log for details."
    exit 1
fi

echo "✅ Step 2 completed successfully."
echo "------------------------------------------------------------"


# --- Step 3: Bid Prediction ---
echo " Kicking off Step 3: BidPrediction..."
python step_3_BidPrediction.py > logs/step_3_BidPrediction_${TIMESTAMP}.log 2>&1

if [ $? -ne 0 ]; then
    echo "❌ ERROR: step_3_BidPrediction.py failed. Halting pipeline."
    echo "   - Check logs/step_3_BidPrediction_${TIMESTAMP}.log for details."
    exit 1
fi

echo "✅ Step 3 completed successfully."
echo "============================================================"
echo "🎉 SMU Data Pipeline finished successfully at $(date)"
echo "============================================================"