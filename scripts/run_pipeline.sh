#!/bin/bash
set -o pipefail

# Force UTF-8 encoding for all Python processes
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

# ==============================================================================
# SMU Bidding Data Pipeline Orchestrator
# ==============================================================================
# This script runs the full data pipeline: scraping (Step 1) + processing (Step 2).
#
# Execution Flow:
# 0. Step 0 (optional): Download files from Supabase Storage if USE_SUPABASE_STORAGE=true
# 1. Step 1 (scraping): Parallel streams A & B (requires Chrome/chromedriver)
#    - Stream A: class_scraper → html_data_extractor → raw_data.xlsx
#    - Stream B: overall_results_scraper → overallBossResults/*.xlsx
# 2. Step 2 runs: PipelineCoordinator
#    - Phase 1: acad_term, courses, professors, bid_windows
#    - Phase 2: classes, timings, availability, bid_results
#    - Phase 3: bid_predictions (with safety_factors)
# 3. Step 3 (optional): Upload results to Supabase Storage if USE_SUPABASE_STORAGE=true
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

# Reset the local success/started markers. _SUCCESS is written only after ALL
# steps succeed; _STARTED is written immediately so the Lambda's retry logic
# can tell that this run is in progress (and avoid concurrent ECS tasks).
rm -f script_output/_SUCCESS script_output/_STARTED
touch script_output/_STARTED

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "============================================================"
echo "🚀 Starting SMU Data Pipeline at $(date)"
echo "============================================================"

# --- Canonical term/window identifiers ---
# TARGET_CURRENT_WINDOW is the canonical ABBREV (e.g. R1FW4) passed by the
# Lambda. It is used verbatim for marker/CSV/log paths so they line up with the
# Lambda's output/{ACAD_TERM_ID}/{window}/_SUCCESS checks — no re-parsing.
WINDOW_CODE="${TARGET_CURRENT_WINDOW:-UNKNOWN}"
export WINDOW_CODE
echo "Target window code: ${WINDOW_CODE}"

# --- Step 0: Download files from Supabase Storage (if enabled) ---
if [ "${USE_SUPABASE_STORAGE}" = "true" ]; then
    echo "[Step 0] Downloading files from Supabase Storage..."

    python -c "
import os
import sys
from pathlib import Path

project_root = Path('.').resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from supabase import create_client
from src.config import SUPABASE_URL, SUPABASE_SERVICE_KEY
from src.logging.logger import get_logger

logger = get_logger(__name__)
logger.info('Supabase Storage download started')

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
bucket = 'bidlysmu-files'

# Download raw_data.xlsx (shared across all terms/windows)
try:
    response = supabase.storage.from_(bucket).download('input/raw_data.xlsx')
    Path('script_input/raw_data.xlsx').write_bytes(response)
    logger.info('Downloaded: input/raw_data.xlsx')
except Exception as e:
    logger.info(f'raw_data.xlsx not found (will be created by scraper): {e}')

# Download overall results files (flat structure)
try:
    files = supabase.storage.from_(bucket).list('input/overallBossResults')
    Path('script_input/overallBossResults').mkdir(parents=True, exist_ok=True)
    for f in files:
        if f['name'].endswith('.xlsx'):
            remote_path = f'input/overallBossResults/{f[\"name\"]}'
            local_path = Path('script_input/overallBossResults') / f['name']
            response = supabase.storage.from_(bucket).download(remote_path)
            local_path.write_bytes(response)
            logger.info(f'Downloaded: {remote_path}')
except Exception as e:
    logger.info(f'overallBossResults not found (will be created by scraper): {e}')
logger.info('Supabase Storage download completed')
" 2>&1

    if [ $? -ne 0 ]; then
        echo "⚠️ WARNING: Step 0 (Supabase download) failed. Continuing with local files."
    else
        echo "✅ Step 0 (Supabase download) completed."
    fi
    echo "------------------------------------------------------------"
fi

# --- Step 0b: Reset markers for THIS term/window run ---
# Delete any stale _SUCCESS / _STARTED for the CURRENT term/window only (a
# previous run's _SUCCESS would otherwise make the Lambda suppress retries for
# the current run), then upload a fresh _STARTED so retries can detect that
# this run is still in progress and avoid spawning a concurrent ECS task.
if [ "${USE_SUPABASE_STORAGE}" = "true" ]; then
    echo "[Step 0b] Resetting pipeline markers for ${WINDOW_CODE}..."

    python -c "
import os
import sys
from pathlib import Path

project_root = Path('.').resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from supabase import create_client
from src.config import SUPABASE_URL, SUPABASE_SERVICE_KEY, ACAD_TERM_ID
from src.logging.logger import get_logger

logger = get_logger(__name__)
logger.info('Resetting pipeline markers')

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
bucket = 'bidlysmu-files'
window_code = os.environ.get('WINDOW_CODE', 'UNKNOWN')
remote_dir = f'output/{ACAD_TERM_ID}/{window_code}'

# Remove stale markers for THIS term/window only — never touch other
# terms/windows (their markers belong to their own runs).
for marker in ('_SUCCESS', '_STARTED'):
    try:
        supabase.storage.from_(bucket).remove([f'{remote_dir}/{marker}'])
        logger.info(f'Removed stale marker: {remote_dir}/{marker}')
    except Exception as e:
        logger.info(f'No stale marker to remove: {remote_dir}/{marker} ({e})')

# Upload fresh _STARTED marker (Lambda retry detection: run in progress)
_started = Path('script_output/_STARTED')
if _started.exists():
    try:
        supabase.storage.from_(bucket).upload(
            f'{remote_dir}/_STARTED',
            _started.read_bytes(),
            {'content-type': 'text/plain', 'upsert': 'true'}
        )
        logger.info(f'Uploaded: {remote_dir}/_STARTED')
    except Exception as e:
        logger.error(f'Failed to upload _STARTED marker: {e}')
" 2>&1

    if [ $? -ne 0 ]; then
        echo "⚠️ WARNING: Step 0b (marker reset) failed. Continuing with pipeline."
    else
        echo "✅ Step 0b (marker reset) completed."
    fi
    echo "------------------------------------------------------------"
fi

# Generate log filename BEFORE Step 1 (combines ACAD_TERM_ID and window code).
# WINDOW_CODE is the canonical abbrev from TARGET_CURRENT_WINDOW (above), so
# the log name matches the marker/output paths exactly.
LOG_FILENAME=$(python -c "
import os
import sys
from pathlib import Path
project_root = Path('.').resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import ACAD_TERM_ID
from datetime import datetime

ts = datetime.now().strftime('%Y%m%d_%H%M%S')
wc = os.environ.get('WINDOW_CODE', 'UNKNOWN')
print(f'{ACAD_TERM_ID}_{wc}_{ts}.log')
")

# Fallback if LOG_FILENAME is empty (config import warning leaked to stdout)
if [ -z "$LOG_FILENAME" ] || [ "$LOG_FILENAME" = "logs/" ]; then
    LOG_FILENAME="pipeline_$(date +%Y%m%d_%H%M%S).log"
fi
echo "Log file: logs/${LOG_FILENAME}"
echo "------------------------------------------------------------"

# --- Step 1: Scraping (requires Chrome/chromedriver) ---
# Stream A: class_scraper.py (1a) -> html_data_extractor.py (1b)
# Stream B: overall_results_scraper.py (1c)
# NOTE: Stream B is skipped on R1W1 — no past results exist for the first window

STREAM_A_LOG="logs/${LOG_FILENAME/.log/_1a_class_scrape.log}"
STREAM_B_LOG="logs/${LOG_FILENAME/.log/_1b_overall_results.log}"

SKIP_STREAM_B=false
# Stream B (overall_results_scraper) seeds predictions from PAST results, so
# it has nothing to scrape for the first window of a term (R1W1).  WINDOW_CODE
# is the canonical abbrev (R1W1) passed verbatim by the Lambda, so compare on
# it directly instead of grepping a full window name.
if [ "$WINDOW_CODE" = "R1W1" ]; then
    echo "Skipping Stream B (OverallResults): no past results for first window (R1W1)"
    SKIP_STREAM_B=true
fi

(
    echo "[Stream A] Running class_scraper.py (1a)..."
    python -c "
import sys
from pathlib import Path
project_root = Path('.').resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import BIDDING_SCHEDULES, ACAD_TERM_ID
from src.driver.authenticator import AutomatedLogin, AuthCredentials
from src.driver.driver_factory import ChromeDriverFactory
from src.scraper.class_scraper import ClassScraper, ClassScraperConfig
from src.scraper.scraper_coordinator import ScraperCoordinator
from src.logging.logger import get_logger

logger = get_logger(__name__)
logger.info('Starting class_scraper')

config = ClassScraperConfig(bidding_schedules=BIDDING_SCHEDULES, start_ay_term=ACAD_TERM_ID, headless=True)
driver_factory = ChromeDriverFactory(headless=True, window_size='1920,1080')
credentials = AuthCredentials.from_environment()
authenticator = AutomatedLogin(credentials, driver_factory=driver_factory.create)
scraper = ClassScraper(config=config)
coordinator = ScraperCoordinator(authenticator=authenticator, scraper=scraper, logger=logger)
logger.info(f'Scraping term={ACAD_TERM_ID}')
result = coordinator.run(acad_term_id=ACAD_TERM_ID)
logger.info(f'Scraping completed: {result}')
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
) 2>&1 | tee "$STREAM_A_LOG" &
PID_A=$!

if [ "$SKIP_STREAM_B" = true ]; then
    echo "[Stream B] SKIPPED — no past results for R1W1"
    PID_B=0
    CODE_B=0
else
(
    echo "[Stream B] Running overall_results_scraper.py (1c)..."
    python -c "
import sys
from pathlib import Path
project_root = Path('.').resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import BIDDING_SCHEDULES, ACAD_TERM_ID
from src.driver.authenticator import AutomatedLogin, AuthCredentials
from src.driver.driver_factory import ChromeDriverFactory
from src.scraper.overall_results_scraper import OverallResultsScraper, OverallResultsConfig
from src.scraper.scraper_coordinator import ScraperCoordinator
from src.logging.logger import get_logger

logger = get_logger(__name__)
logger.info('Starting overall_results_scraper')

config = OverallResultsConfig(bidding_schedules=BIDDING_SCHEDULES, start_ay_term=ACAD_TERM_ID, headless=True)
driver_factory = ChromeDriverFactory(headless=True, window_size='1920,1080')
credentials = AuthCredentials.from_environment()
authenticator = AutomatedLogin(credentials, driver_factory=driver_factory.create)
scraper = OverallResultsScraper(config=config)
coordinator = ScraperCoordinator(authenticator=authenticator, scraper=scraper, logger=logger)
logger.info(f'Scraping term={ACAD_TERM_ID}')
result = coordinator.run(term=ACAD_TERM_ID, bid_round=None, bid_window=None, output_dir='./script_input/overallBossResults', authenticator=None)
logger.info(f'Scraping completed: {result}')
" 2>&1
) 2>&1 | tee "$STREAM_B_LOG" &
PID_B=$!
fi  # End of SKIP_STREAM_B conditional

wait $PID_A
CODE_A=$?
if [ "$SKIP_STREAM_B" != true ]; then
    wait $PID_B
    CODE_B=$?
fi

if [ $CODE_A -ne 0 ] || [ $CODE_B -ne 0 ]; then
    echo "ERROR: Step 1 (scraping) failed. Halting pipeline."
    echo "--- Stream A log (last 30 lines) ---"
    tail -30 "$STREAM_A_LOG" 2>/dev/null || echo "(no output)"
    echo "--- Stream B log (last 30 lines) ---"
    tail -30 "$STREAM_B_LOG" 2>/dev/null || echo "(no output)"
    exit 1
fi
echo "✅ Step 1 (scraping) completed."
echo "------------------------------------------------------------"

# --- Step 1b: Upload raw scraped data to Supabase (BEFORE DB ingestion) ---
# This ensures scraped HTML/raw data is preserved even if Step 2 crashes.
if [ "${USE_SUPABASE_STORAGE}" = "true" ]; then
    echo "[Step 1b] Uploading raw scraped data to Supabase Storage..."

    python -c "
import os
import sys
from pathlib import Path

project_root = Path('.').resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from supabase import create_client
from src.config import SUPABASE_URL, SUPABASE_SERVICE_KEY
from src.logging.logger import get_logger

logger = get_logger(__name__)
logger.info('Supabase Storage upload of raw scraped data started')

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
bucket = 'bidlysmu-files'

# Upload scraped HTML files (raw data — most important to preserve)
html_dir = Path('script_input/classTimingsFull')
if html_dir.exists():
    for html_file in html_dir.rglob('*.html'):
        remote_path = f'input/classTimingsFull/{html_file.relative_to(html_dir)}'
        try:
            supabase.storage.from_(bucket).upload(
                str(remote_path),
                html_file.read_bytes(),
                {'content-type': 'text/html', 'upsert': 'true'}
            )
        except Exception as e:
            logger.error(f'Failed to upload HTML {html_file.name}: {e}')
    logger.info('Uploaded scraped HTML files')

# Upload raw_data.xlsx (extracted from HTML)
raw_data = Path('script_input/raw_data.xlsx')
if raw_data.exists():
    try:
        supabase.storage.from_(bucket).upload(
            'input/raw_data.xlsx',
            raw_data.read_bytes(),
            {'content-type': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'upsert': 'true'}
        )
        logger.info('Uploaded: input/raw_data.xlsx')
    except Exception as e:
        logger.error(f'Failed to upload raw_data.xlsx: {e}')

# NOTE: bidding_schedules.json is deliberately NOT re-uploaded here.
# The canonical path schedules/bidding_schedules.json is written ONLY by the
# monthly scheduler Lambda (from fresh Trumba data).  Re-uploading the baked-in
# git copy from script_input/ would clobber the canonical schedules with stale
# data.  The local file stays local-only for local pipeline runs.

# Upload overall results files
overall_dir = Path('script_input/overallBossResults')
if overall_dir.exists():
    for xlsx_file in overall_dir.glob('*.xlsx'):
        remote_path = f'input/overallBossResults/{xlsx_file.name}'
        try:
            supabase.storage.from_(bucket).upload(
                remote_path,
                xlsx_file.read_bytes(),
                {'content-type': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'upsert': 'true'}
            )
        except Exception as e:
            logger.error(f'Failed to upload {xlsx_file.name}: {e}')
    logger.info('Uploaded overall results files')

logger.info('Supabase Storage upload of raw scraped data completed')
" 2>&1

    if [ $? -ne 0 ]; then
        echo "⚠️ WARNING: Step 1b (Supabase upload of raw data) failed. Continuing with pipeline."
    else
        echo "✅ Step 1b (Supabase upload of raw data) completed."
    fi
    echo "------------------------------------------------------------"
fi

# --- Step 2: Table Building (Direct Coordinator Call) ---
echo " Kicking off Step 2: PipelineCoordinator..."

STEP2_LOG="logs/${LOG_FILENAME/.log/_2_process_pipeline.log}"

python -c "
import sys
from src.config import BIDDING_SCHEDULES, ACAD_TERM_ID, DB_CONFIG, PipelineConfig
from src.pipeline.pipeline_coordinator import PipelineCoordinator

config = PipelineConfig.from_env(
    bidding_schedules=BIDDING_SCHEDULES,
    start_ay_term=ACAD_TERM_ID,
    db_config=DB_CONFIG
)
coordinator = PipelineCoordinator(config=config)
coordinator.run()
" 2>&1 | tee -a "$STEP2_LOG"

EXIT_CODE=${PIPESTATUS[0]}
if [ $EXIT_CODE -ne 0 ]; then
    echo "❌ ERROR: PipelineCoordinator failed with exit code $EXIT_CODE. Halting pipeline."
    echo "   - Check $STEP2_LOG for details."
    exit 1
fi

echo "✅ Step 2 completed successfully."

# Write success marker for retry detection
date -u +%Y-%m-%dT%H:%M:%SZ > script_output/_SUCCESS
echo "✅ Success marker written to script_output/_SUCCESS"

# --- Step 3: Upload results to Supabase Storage (if enabled) ---
if [ "${USE_SUPABASE_STORAGE}" = "true" ]; then
    echo "------------------------------------------------------------"
    echo "[Step 3] Uploading results to Supabase Storage..."

    python -c "
import os
import sys
from pathlib import Path
import json

project_root = Path('.').resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from supabase import create_client
from src.config import SUPABASE_URL, SUPABASE_SERVICE_KEY, ACAD_TERM_ID
from src.logging.logger import get_logger

logger = get_logger(__name__)
logger.info('Supabase Storage upload started')

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
bucket = 'bidlysmu-files'

# Canonical marker path: output/{ACAD_TERM_ID}/{WINDOW_CODE} — ACAD_TERM_ID is
# AY format (matches the Lambda's _pipeline_already_succeeded / _STARTED
# checks) and WINDOW_CODE is the canonical abbrev from TARGET_CURRENT_WINDOW
# (see the WINDOW_CODE computation at the top).
window_code = os.environ.get('WINDOW_CODE', 'UNKNOWN')
remote_dir = f'output/{ACAD_TERM_ID}/{window_code}'

# Upload generated CSV files from script_output (organized by term/window)
output_dir = Path('script_output')
for csv_file in output_dir.glob('*.csv'):
    remote_path = f'{remote_dir}/{csv_file.name}'
    try:
        supabase.storage.from_(bucket).upload(
            remote_path,
            csv_file.read_bytes(),
            {'content-type': 'text/csv', 'upsert': 'true'}
        )
        logger.info(f'Uploaded: {remote_path}')
    except Exception as e:
        logger.error(f'Failed to upload {csv_file.name}: {e}')

# Upload _SUCCESS marker for Lambda retry detection
_success_file = Path('script_output/_SUCCESS')
if _success_file.exists():
    remote_success_path = f'{remote_dir}/_SUCCESS'
    try:
        supabase.storage.from_(bucket).upload(
            remote_success_path,
            _success_file.read_bytes(),
            {'content-type': 'text/plain', 'upsert': 'true'}
        )
        logger.info(f'Uploaded: {remote_success_path}')
    except Exception as e:
        logger.error(f'Failed to upload _SUCCESS marker: {e}')

logger.info('Supabase Storage upload of pipeline outputs completed')
" 2>&1

    if [ $? -ne 0 ]; then
        echo "⚠️ WARNING: Step 3 (Supabase upload) failed."
    else
        echo "✅ Step 3 (Supabase upload) completed."
    fi
fi

echo "============================================================"
echo "🎉 SMU Data Pipeline finished successfully at $(date)"
echo "============================================================"
echo "📁 Step 2 log saved to: $STEP2_LOG"