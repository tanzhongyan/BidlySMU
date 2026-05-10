#!/bin/bash

# Force UTF-8 encoding for all Python processes
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

# ==============================================================================
# SMU Bidding Data Pipeline Orchestrator
# ==============================================================================
# This script runs the full data pipeline: scraping (Step 1) + processing (Step 2).
#
# Execution Flow:
# 1. Step 1 (scraping): Parallel streams A & B (requires Chrome/chromedriver)
#    - Stream A: class_scraper → html_data_extractor → raw_data.xlsx
#    - Stream B: overall_results_scraper → overallBossResults/*.xlsx
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
# ==============================================================================

# --- Setup ---
# Consolidated logging to single logs/ directory at project root
mkdir -p logs
mkdir -p script_output

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "============================================================"
echo "🚀 Starting SMU Data Pipeline at $(date)"
echo "============================================================"

# Generate log filename BEFORE Step 1 (combines ACAD_TERM_ID and window code)
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

echo "Log file: logs/${LOG_FILENAME}"
echo "------------------------------------------------------------"

# --- Step 1: Scraping (requires Chrome/chromedriver) ---
# Stream A: class_scraper.py (1a) -> html_data_extractor.py (1b)
# Stream B: overall_results_scraper.py (1c)

STREAM_A_LOG="logs/${LOG_FILENAME/.log/_1a_class_scrape.log}"
STREAM_B_LOG="logs/${LOG_FILENAME/.log/_1b_overall_results.log}"

(
    echo "[Stream A] Running class_scraper.py (1a)..."
    python -c "
import sys
from pathlib import Path
project_root = Path('.').resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import BIDDING_SCHEDULES, START_AY_TERM, ACAD_TERM_ID
from src.driver.authenticator import AutomatedLogin, AuthCredentials
from src.driver.driver_factory import ChromeDriverFactory
from src.scraper.class_scraper import ClassScraper, ClassScraperConfig
from src.logging.logger import get_logger

logger = get_logger(__name__)
logger.info('Starting class_scraper')

config = ClassScraperConfig(bidding_schedules=BIDDING_SCHEDULES, start_ay_term=START_AY_TERM, headless=True)
driver_factory = ChromeDriverFactory(headless=True, window_size='1920,1080')
credentials = AuthCredentials.from_environment()
authenticator = AutomatedLogin(credentials)
scraper = ClassScraper(config=config)
driver = driver_factory.create()
scraper.connect(driver)
driver.get('https://boss.intranet.smu.edu.sg/')
authenticator.login(driver)
logger.info(f'Scraping term={ACAD_TERM_ID}')
result = scraper.scrape(acad_term_id=ACAD_TERM_ID)
logger.info(f'Scraping completed: {result}')
driver.quit()
" 2>&1

    echo "[Stream A] Running html_data_extractor.py (1b)..."
    python -c "
import sys
from pathlib import Path
project_root = Path('.').resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.scraper.html_data_extractor import HTMLDataExtractor
from src.logging.logger import get_logger

logger = get_logger(__name__)
logger.info('Starting html_data_extractor')
extractor = HTMLDataExtractor()
result = extractor.scrape(output_path='script_input/raw_data.xlsx')
logger.info(f'Extraction completed: {result}')
" 2>&1
) >> "$STREAM_A_LOG" 2>&1 &
PID_A=$!

(
    echo "[Stream B] Running overall_results_scraper.py (1c)..."
    python -c "
import sys
from pathlib import Path
project_root = Path('.').resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import BIDDING_SCHEDULES, START_AY_TERM
from src.driver.authenticator import AutomatedLogin, AuthCredentials
from src.driver.driver_factory import ChromeDriverFactory
from src.scraper.overall_results_scraper import OverallResultsScraper, OverallResultsConfig
from src.logging.logger import get_logger

logger = get_logger(__name__)
logger.info('Starting overall_results_scraper')

config = OverallResultsConfig(bidding_schedules=BIDDING_SCHEDULES, start_ay_term=START_AY_TERM, headless=True)
driver_factory = ChromeDriverFactory(headless=True, window_size='1920,1080')
credentials = AuthCredentials.from_environment()
authenticator = AutomatedLogin(credentials)
scraper = OverallResultsScraper(config=config)
driver = driver_factory.create()
scraper.connect(driver)
driver.get('https://boss.intranet.smu.edu.sg/')
authenticator.login(driver)
logger.info(f'Scraping term={START_AY_TERM}')
result = scraper.scrape(term=START_AY_TERM, bid_round=None, bid_window=None, output_dir='./script_input/overallBossResults', authenticator=None)
logger.info(f'Scraping completed: {result}')
driver.quit()
" 2>&1
) >> "$STREAM_B_LOG" 2>&1 &
PID_B=$!

wait $PID_A
CODE_A=$?
wait $PID_B
CODE_B=$?

if [ $CODE_A -ne 0 ] || [ $CODE_B -ne 0 ]; then
    echo "❌ ERROR: Step 1 (scraping) failed. Halting pipeline."
    exit 1
fi
echo "✅ Step 1 (scraping) completed."
echo "------------------------------------------------------------"


# --- Step 2: Table Building (Direct Coordinator Call) ---
echo " Kicking off Step 2: PipelineCoordinator..."

STEP2_LOG="logs/${LOG_FILENAME/.log/_2_process_pipeline.log}"

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
" 2>&1 | tee -a "$STEP2_LOG"

if [ $? -ne 0 ]; then
    echo "❌ ERROR: PipelineCoordinator failed. Halting pipeline."
    echo "   - Check $STEP2_LOG for details."
    exit 1
fi

echo "✅ Step 2 completed successfully."
echo "============================================================"
echo "🎉 SMU Data Pipeline finished successfully at $(date)"
echo "============================================================"
echo "📁 Step 2 log saved to: $STEP2_LOG"