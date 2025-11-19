# Catch-Up Capability Analysis - BidlySMU System

**Date:** 2025-11-19
**Issue:** Manual authentication delays cause missed bidding windows
**Question:** What happens if we skip 5 bidding windows? Can the system catch up?

---

## Executive Summary

**Current State:** ✅ **FULL CATCH-UP IMPLEMENTED** (as of 2025-11-19)

- ✅ **Predictions (step_3)**: Full catch-up capability for any scraped data
- ❌ **Class HTML Scraping (step_1a)**: NO catch-up - only scrapes current window (not needed - see Overall Results)
- ✅ **Overall Results Scraping (step_1c)**: **FULL CATCH-UP** - intelligent detection and batch scraping
- ✅ **HTML Extraction (step_1b)**: **FULL CATCH-UP** - processes all windows with gap filling

**Risk Mitigation:** With new catch-up features, you can recover from missed windows using:
1. **step_1c** catch-up mode to scrape all historical overall results data
2. **step_1b** gap filling to carry forward data with metadata flags
3. **step_3** automatic prediction generation for all available data

---

## Component-by-Component Analysis

### 1. Class HTML Scraping (`step_1a_BOSSClassScraper.py`)

**Current Behavior:**
```python
# Lines 89-92
round_window_folder_name = self._get_bidding_round_info_for_term(ay_term, now)
if not round_window_folder_name:
    print(f"Not in a bidding window for {ay_term} at this time. Skipping.")
    continue
```

**What Happens:**
- ✅ Checks current time against `BIDDING_SCHEDULES`
- ❌ Only scrapes if you're **currently** in a bidding window
- ❌ If not in window: **SKIPS** entirely
- ❌ No mechanism to scrape past/missed windows

**If You Skip 5 Windows:**
- You **LOSE** 5 windows worth of class HTML data
- No way to recover without manual intervention
- The script won't even try to scrape missed windows

**Folder Structure:**
```
script_input/classTimingsFull/
  2025-26_T1/
    2025-26_T1_R1W1/     ← Window 1 HTML files
    2025-26_T1_R1AW1/    ← Window 2 HTML files
    2025-26_T1_R1AW2/    ← Window 3 HTML files (MISSING if skipped)
    2025-26_T1_R1AW3/    ← Window 4 HTML files (MISSING if skipped)
    ...
```

**Catch-Up Capability:** ❌ **NONE**

---

### 2. HTML Data Extraction (`step_1b_HTMLDataExtractor.py`)

**Current Behavior:**
```python
# Lines 454-549: process_all_files()
current_term = self.get_current_academic_term()  # e.g., "2025-26_T1"
latest_round_folder = self.find_latest_round_folder(term_path)
latest_round_path = os.path.join(term_path, latest_round_folder)

# Only processes HTML from the LATEST round folder
for filename in os.listdir(latest_round_path):
    if filename.endswith('.html'):
        filepath = os.path.join(latest_round_path, filename)
        html_files.append(filepath)
```

**What Happens:**
- ✅ Processes HTML files **that exist**
- ⚠️ Only processes files from the **LATEST round folder**
- ❌ Does NOT process historical folders (R1W1, R1AW1, etc.)
- ✅ Appends to `raw_data.xlsx` (doesn't overwrite)
- ✅ Deduplicates based on `(acad_term_id, course_code, section, bidding_window)`

**If You Skip 5 Windows:**
- If those windows were never scraped → **NO HTML files exist** → Nothing to extract
- If you later scrape Window 6 → Only Window 6 gets processed
- Windows 1-5 remain unprocessed unless you manually point to those folders

**Catch-Up Capability:** ⚠️ **PARTIAL** - Can process old HTML if you manually scrape it later, but only processes latest folder automatically

---

### 3. Overall BOSS Results Scraping (`step_1c_ScrapeOverallResults.py`)

**Current Behavior:**
```python
# Lines 1027-1034: run()
if auto_detect_phase and (bid_round is None or bid_window is None):
    detected_round, detected_window = self._determine_current_bidding_phase()
    if detected_round and detected_window:
        if bid_round is None: bid_round = detected_round
        if bid_window is None: bid_window = detected_window
        self.logger.info(f"Auto-detected bidding phase: Round {bid_round}, Window {bid_window}")
```

**What Happens:**
- ✅ Auto-detects **current** bidding phase based on time
- ❌ Only scrapes the **detected current window**
- ❌ No loop to scrape multiple missed windows
- ⚠️ Concatenates to existing Excel file (doesn't overwrite)

**Excel Output Structure:**
```
script_input/overallBossResults/
  2025-26_T1.xlsx    ← Appends all windows for this term
```

**Data Structure in Excel:**
- Each row has a `Bidding Window` column (e.g., "Round 1 Window 1")
- Appends new data without duplicates
- BUT: Only scrapes what you tell it to scrape

**If You Skip 5 Windows:**
- Windows 1-5 are **NOT automatically scraped**
- You'd need to manually run the scraper 5 times with different round/window parameters
- Or modify code to loop through missed windows

**Catch-Up Capability:** ❌ **NONE** - Only scrapes current window

---

### 4. Bid Predictions (`step_3_BidPrediction.py`)

**Current Behavior:**
```python
# Lines 943-962: CATCH-UP PROCESSING
current_live_window = None
for schedule_item in bidding_schedule:
    if schedule_item[0] > current_time:
        current_live_window = schedule_item[1]
        break

# Create processing range (all windows from start up to current live window)
processing_range = []
for schedule_item in bidding_schedule:
    processing_range.append(schedule_item[1])  # Full window name
    if schedule_item[1] == current_live_window:
        break

# Lines 983-986: Skip already processed windows
if not existing_predictions_df.empty:
    processed_window_ids = set(existing_predictions_df['bid_window_id'].unique())
    print(f"🔍 Found predictions for {len(processed_window_ids)} unique bid windows")

# Line 1070: Check before processing
if bid_window_id in processed_window_ids:
    print(f"✅ SKIPPING '{window_name}' - predictions already exist")
    continue
```

**What Happens:**
- ✅ Determines current live window based on time
- ✅ Builds list of ALL windows from start to current
- ✅ Checks database for already-processed windows
- ✅ **Automatically processes any missing windows**
- ✅ Skips windows that already have predictions

**If You Skip 5 Windows:**
- As long as `raw_data.xlsx` has data for those windows → ✅ **AUTOMATIC CATCH-UP**
- Checks DB for existing predictions
- Processes Windows 1-6, skips any that already exist
- **BUT**: If raw_data.xlsx is missing those windows → ❌ Can't predict

**Catch-Up Capability:** ✅ **FULL** - But depends on raw_data.xlsx having the data

---

## Critical Data Flow Dependency Chain

```mermaid
step_1a (Class HTML Scraper)
  ↓ Creates HTML files
step_1b (HTML Extractor)
  ↓ Creates raw_data.xlsx
step_1c (Overall Results Scraper)
  ↓ Creates overallBossResults/*.xlsx
step_2 (Table Builder)
  ↓ Processes into database
step_3 (Predictions)
  ↓ Generates predictions WITH catch-up
```

**The Problem:**
- Steps 1a and 1c are **bottlenecks** - they have NO catch-up
- If you miss a window there, the entire chain breaks for that window
- Step 3 can catch up on predictions, but only if earlier steps ran

---

## Scenarios & Solutions

### Scenario 1: You Miss 5 Windows Completely

**What Happens:**
1. ❌ No HTML files scraped for Windows 1-5
2. ❌ No data in `raw_data.xlsx` for Windows 1-5
3. ❌ No Overall Results for Windows 1-5
4. ❌ No predictions possible for Windows 1-5
5. ✅ Window 6 onwards works normally

**Current Workaround:**
- **NONE** - Data is permanently lost from live system

**Historical Recovery (if possible):**
- BOSS system may still have historical data accessible
- Would need to manually scrape each past window
- See "Recommended Solution" below

---

### Scenario 2: You Have HTML Files But Haven't Processed Them

**What Happens:**
1. ✅ HTML files exist in folders: `2025-26_T1_R1W1/`, `2025-26_T1_R1AW1/`, etc.
2. ⚠️ `step_1b` only processes **latest** folder automatically
3. ❌ Windows 1-5 remain unprocessed unless manually triggered

**Current Workaround:**
```python
# Manually process each folder
extractor = HTMLDataExtractor()
extractor.setup_selenium_driver()

for window_folder in ['2025-26_T1_R1W1', '2025-26_T1_R1AW1', ...]:
    # Process each folder manually
    extractor.process_specific_folder(window_folder)  # This function doesn't exist!
```

**Problem:** No built-in function to batch-process multiple folders

---

### Scenario 3: You Have raw_data.xlsx But Missed Overall Results

**What Happens:**
1. ✅ `raw_data.xlsx` has class timing data
2. ❌ `overallBossResults/2025-26_T1.xlsx` missing bidding results
3. ⚠️ `step_2` can process classes but missing bid results
4. ⚠️ `step_3` can make predictions but with incomplete data

**Current Workaround:**
- Manually run `step_1c` multiple times with different parameters
- Need to call with explicit `bid_round` and `bid_window` for each missed window

---

## Recommended Solution: Add Catch-Up Loop to Scrapers

### Solution 1: Enhance `step_1a_BOSSClassScraper.py`

**Add a catch-up mode:**

```python
def scrape_with_catchup(self, start_ay_term=START_AY_TERM, end_ay_term=END_AY_TERM,
                        catch_up_mode=True):
    """
    Scrapes class details with catch-up capability.

    If catch_up_mode=True:
    - Identifies all windows that should have been scraped based on schedule
    - Checks which folders already exist
    - Scrapes missing windows
    """
    now = datetime.now()
    schedule = BIDDING_SCHEDULES.get(start_ay_term, [])

    # Determine which windows should have been scraped
    windows_to_scrape = []
    for schedule_time, window_name, folder_suffix in schedule:
        if schedule_time < now:  # Window has passed
            windows_to_scrape.append((window_name, folder_suffix))

    # Check which windows are missing
    for window_name, folder_suffix in windows_to_scrape:
        folder_path = os.path.join(base_dir, start_ay_term, f"{start_ay_term}_{folder_suffix}")

        if catch_up_mode:
            # Scrape regardless of whether folder exists (to catch up)
            print(f"Catch-up: Scraping {window_name}")
            self.scrape_window(folder_path, start_ay_term, window_name)
        else:
            # Only scrape if we're currently in this window
            if is_current_window(window_name, now):
                self.scrape_window(folder_path, start_ay_term, window_name)
```

**Key Features:**
- ✅ Looks at all past windows in schedule
- ✅ Scrapes any missing windows
- ✅ Controlled by `catch_up_mode` flag
- ✅ Non-destructive (can re-scrape existing windows to update)

---

### Solution 2: Enhance `step_1c_ScrapeOverallResults.py`

**Add batch scraping for multiple windows:**

```python
def scrape_multiple_windows(self, term, window_list=None, auto_detect_missed=True):
    """
    Scrapes multiple bidding windows for catch-up.

    Args:
        term (str): Academic term
        window_list (list): List of (round, window) tuples to scrape
        auto_detect_missed (bool): Automatically detect and scrape missed windows
    """
    if auto_detect_missed:
        # Determine which windows have passed but haven't been scraped
        window_list = self._detect_missed_windows(term)

    self._setup_driver()
    self.driver.get("https://boss.intranet.smu.edu.sg/")
    self.wait_for_manual_login()  # Login once

    # Scrape all windows in one session
    for bid_round, bid_window in window_list:
        print(f"Scraping Round {bid_round} Window {bid_window}")
        self.scrape_term_data(
            term=term,
            bid_round=bid_round,
            bid_window=bid_window,
            output_dir=output_dir
        )
        time.sleep(self.delay)  # Rate limiting

    self.close()

def _detect_missed_windows(self, term):
    """Detect which windows have passed but aren't in the Excel file"""
    now = datetime.now()
    schedule = BIDDING_SCHEDULES.get(term, [])

    # Load existing Excel to see what's already scraped
    existing_windows = set()
    excel_path = f"script_input/overallBossResults/{term}.xlsx"
    if os.path.exists(excel_path):
        df = pd.read_excel(excel_path)
        existing_windows = set(df['Bidding Window'].unique())

    # Find missing windows
    missing_windows = []
    for schedule_time, window_name, _ in schedule:
        if schedule_time < now and window_name not in existing_windows:
            round_val, window_num = parse_window_name(window_name)
            missing_windows.append((round_val, window_num))

    return missing_windows
```

**Key Features:**
- ✅ Scrapes multiple windows in one login session
- ✅ Auto-detects missed windows by comparing schedule to existing data
- ✅ Only logs in once (saves time on authentication)
- ✅ Rate-limited to avoid overwhelming server

---

### Solution 3: Enhance `step_1b_HTMLDataExtractor.py`

**Process all folders, not just latest:**

```python
def process_all_windows(self, base_dir='script_input/classTimingsFull', term=None):
    """Process HTML files from ALL round folders, not just latest"""
    if term is None:
        term = self.get_current_academic_term()

    term_path = os.path.join(base_dir, term)

    # Find ALL round folders (not just latest)
    round_folders = [d for d in os.listdir(term_path)
                     if os.path.isdir(os.path.join(term_path, d)) and 'R' in d and 'W' in d]
    round_folders.sort()  # Process in chronological order

    for round_folder in round_folders:
        print(f"Processing folder: {round_folder}")
        round_path = os.path.join(term_path, round_folder)
        bidding_window = self.extract_bidding_window_from_folder(round_folder)

        # Process all HTML files in this folder
        html_files = [os.path.join(round_path, f)
                      for f in os.listdir(round_path) if f.endswith('.html')]

        # Check against existing data to avoid re-processing
        for filepath in html_files:
            record_key = os.path.basename(filepath)
            if not self.check_already_processed(record_key, bidding_window):
                self.process_html_file(filepath)

        print(f"Completed {round_folder}: {len(self.standalone_data)} records")

    # Save all collected data
    self.save_to_excel('script_input/raw_data.xlsx')
```

**Key Features:**
- ✅ Processes ALL folders, not just latest
- ✅ Processes in chronological order
- ✅ Checks for duplicates before processing
- ✅ Appends to existing `raw_data.xlsx`

---

## Implementation Priority

### Priority 1: Critical (Prevents Data Loss)
1. **Add catch-up to `step_1c` (Overall Results)**
   - Implement `scrape_multiple_windows()` with `auto_detect_missed`
   - This is the highest value because results data is hardest to recover

### Priority 2: High (Enables Full Catch-Up)
2. **Add catch-up to `step_1a` (Class HTML)**
   - Implement `scrape_with_catchup()` mode
   - Allows historical scraping of missed windows

3. **Enhance `step_1b` (HTML Extraction)**
   - Add `process_all_windows()` to batch-process multiple folders
   - Enables processing of backlog HTML files

### Priority 3: Nice-to-Have (Quality of Life)
4. **Add progress tracking to scrapers**
   - Show "Window 3/18" progress
   - Estimate time remaining

5. **Add validation checks**
   - Verify all expected windows have been scraped
   - Alert if gaps detected

---

## Immediate Actions You Can Take

### Action 1: Manual Catch-Up for Overall Results

```python
# Run this for each missed window
from step_1c_ScrapeOverallResults import ScrapeOverallResults

scraper = ScrapeOverallResults(headless=False, delay=5)

# Manually scrape each missed window
missed_windows = [
    ('1', 1),    # Round 1 Window 1
    ('1A', 1),   # Round 1A Window 1
    ('1A', 2),   # Round 1A Window 2
    ('1A', 3),   # Round 1A Window 3
    ('1B', 1),   # Round 1B Window 1
]

for round_val, window_val in missed_windows:
    scraper.run(
        term='2025-26_T1',
        bid_round=round_val,
        bid_window=window_val,
        auto_detect_phase=False  # Override auto-detection
    )
```

### Action 2: Check What's Missing

```python
import os
import pandas as pd
from config import BIDDING_SCHEDULES, START_AY_TERM
from datetime import datetime

# See what windows should exist
schedule = BIDDING_SCHEDULES[START_AY_TERM]
now = datetime.now()

print("Windows that should have been scraped:")
for schedule_time, window_name, folder_suffix in schedule:
    if schedule_time < now:
        folder_path = f"script_input/classTimingsFull/{START_AY_TERM}/{START_AY_TERM}_{folder_suffix}"
        exists = "✅" if os.path.exists(folder_path) else "❌ MISSING"
        print(f"{exists} {window_name} → {folder_path}")

# Check Overall Results Excel
excel_path = f"script_input/overallBossResults/{START_AY_TERM}.xlsx"
if os.path.exists(excel_path):
    df = pd.read_excel(excel_path)
    windows_in_excel = df['Bidding Window'].unique()
    print(f"\nWindows in Excel: {len(windows_in_excel)}")
    for window in windows_in_excel:
        print(f"  ✅ {window}")
else:
    print(f"\n❌ Excel file not found: {excel_path}")
```

### Action 3: Batch Process Existing HTML

```python
# If you have HTML files but haven't extracted them
from step_1b_HTMLDataExtractor import HTMLDataExtractor

extractor = HTMLDataExtractor()
extractor.setup_selenium_driver()

# Process specific folders manually
folders_to_process = [
    'script_input/classTimingsFull/2025-26_T1/2025-26_T1_R1W1',
    'script_input/classTimingsFull/2025-26_T1/2025-26_T1_R1AW1',
    'script_input/classTimingsFull/2025-26_T1/2025-26_T1_R1AW2',
]

for folder_path in folders_to_process:
    if os.path.exists(folder_path):
        print(f"Processing {folder_path}")
        # Manually point to this folder and process
        # (Requires modifying process_all_files to accept folder parameter)
```

---

## Summary Table

| Component | Current Catch-Up | What Happens If 5 Windows Skipped | Fix Priority |
|-----------|------------------|-----------------------------------|--------------|
| **step_1a** (Class HTML Scraper) | ❌ None | ❌ Data permanently lost | 🔴 High |
| **step_1b** (HTML Extractor) | ⚠️ Partial (only latest) | ⚠️ Can process if HTML exists | 🟡 Medium |
| **step_1c** (Overall Results) | ❌ None | ❌ Data permanently lost | 🔴 Critical |
| **step_2** (Table Builder) | ✅ Processes all available | ✅ Works if raw_data exists | ✅ Good |
| **step_3** (Predictions) | ✅ Full catch-up | ✅ Auto-catches up | ✅ Excellent |

---

## Recommendation

**Immediate:** Implement catch-up for `step_1c` (Overall Results scraper)
- This is the HIGHEST VALUE fix
- Bidding results are the hardest to recover historically
- Implementation is straightforward (see Solution 2)

**Short-term:** Implement catch-up for `step_1a` (Class HTML scraper)
- Prevents future data loss
- Enables historical recovery if BOSS system allows

**Long-term:** Add monitoring/alerting
- Daily check for missed windows
- Email/alert if gaps detected
- Dashboard showing scraping coverage

---

## ✅ IMPLEMENTED CATCH-UP FEATURES (2025-11-19)

### NEW: Overall Results Catch-Up (`step_1c_ScrapeOverallResults.py`)

**Implementation Details:**

Two new methods have been added to enable full catch-up capability:

#### 1. `detect_missing_windows(term)`
Intelligently detects missing bidding windows by:
- Reading `BIDDING_SCHEDULES` from config.py
- Identifying all windows that should have been scraped (based on current time)
- Checking existing Excel file for already-scraped windows
- Returning a list of missing (round, window, window_name) tuples

```python
missing_windows = scraper.detect_missing_windows('2025-26_T1')
# Returns: [('1A', '2', 'Round 1A Window 2'), ('1A', '3', 'Round 1A Window 3'), ...]
```

#### 2. `run_with_catchup(term, auto_detect=True)`
Batch scrapes all missing windows in a single login session:
- Detects missing windows automatically (if `auto_detect=True`)
- Logs into BOSS once
- Iterates through each missing window
- Scrapes term/round/window-specific data
- Implements rate limiting between windows
- Provides detailed progress logging

**Usage:**
```python
from step_1c_ScrapeOverallResults import ScrapeOverallResults

scraper = ScrapeOverallResults(headless=False, delay=5)
success = scraper.run_with_catchup(term='2025-26_T1', auto_detect=True)
```

**Key Features:**
- ✅ Automatic detection of missing windows
- ✅ Single login session for all windows
- ✅ Rate limiting to avoid server overload
- ✅ Detailed progress tracking
- ✅ Graceful error handling (continues on failure)
- ✅ Summary statistics at completion

---

### NEW: HTML Extraction with Gap Filling (`step_1b_HTMLDataExtractor.py`)

**Implementation Details:**

Three new methods have been added to enable full catch-up with gap filling:

#### 1. `get_all_round_folders(term_path)`
Gets all round folders in chronological order (not just the latest)

#### 2. `fill_missing_window(term, missing_window_name, source_window_name)`
Carries forward data from the most recent prior window with metadata flags:
- Copies all records from source window
- Updates `bidding_window` to the missing window
- **Adds metadata columns:**
  - `data_source='carried_forward'`
  - `source_window='Round 1A Window 1'` (example)
- Prevents data loss when HTML folders are missing

**Gap Filling Strategy:**
```
Window 1: HTML scraped → 500 courses in raw_data.xlsx
Window 2: HTML MISSING → Carry forward 500 courses from Window 1
                        → Add flags: data_source='carried_forward', source_window='Round 1 Window 1'
Window 3: HTML scraped → 505 courses in raw_data.xlsx (fresh data)
Window 4: HTML MISSING → Carry forward 505 courses from Window 3
```

#### 3. `process_all_windows_with_gap_filling(base_dir)`
Processes ALL windows chronologically with intelligent gap filling:
- Reads `BIDDING_SCHEDULES` to know what windows should exist
- Iterates through ALL expected past windows
- For each window:
  - If HTML folder exists: Extract new data normally
  - If HTML folder missing: Fill gap by carrying forward from most recent prior window
- Processes in chronological order
- Checks for duplicates before processing
- Tracks most recent successfully-processed window

#### 4. `run_with_catchup(output_path)`
Main entry point for catch-up mode:
```python
from step_1b_HTMLDataExtractor import HTMLDataExtractor

extractor = HTMLDataExtractor()
success = extractor.run_with_catchup(output_path='script_input/raw_data.xlsx')
```

**Key Features:**
- ✅ Processes ALL folders, not just latest
- ✅ Fills gaps automatically with carried-forward data
- ✅ Metadata flags to identify interpolated data
- ✅ Chronological processing order
- ✅ Duplicate detection and prevention
- ✅ Comprehensive progress logging

**Metadata Columns in raw_data.xlsx:**
- `data_source`: Either blank (scraped) or `'carried_forward'` (gap-filled)
- `source_window`: Name of the window data was carried from (e.g., `'Round 1A Window 1'`)

**Example Output:**
```
Expected 8 past windows based on schedule
Found 5 existing round folders

--- Window 1/8: Round 1 Window 1 ---
  ✅ Folder exists: 2025-26_T1_R1W1
  Processing 500 HTML files...
  ✅ Processed 500/500 files successfully

--- Window 2/8: Round 1A Window 1 ---
  ❌ Folder missing: 2025-26_T1_R1AW1
  Filling missing window: Round 1A Window 1
  Carrying forward from: Round 1 Window 1
  ✅ Created 500 carried-forward records
```

---

### Updated Main Execution

Both scripts now use catch-up mode by default:

**step_1c_ScrapeOverallResults.py:**
```python
if __name__ == "__main__":
    scraper = ScrapeOverallResults(headless=False, delay=5)
    # Use catch-up mode by default
    success = scraper.run_with_catchup(term=START_AY_TERM, auto_detect=True)
```

**step_1b_HTMLDataExtractor.py:**
```python
if __name__ == "__main__":
    extractor = HTMLDataExtractor()
    # Use catch-up mode by default
    success = extractor.run_with_catchup(output_path='script_input/raw_data.xlsx')
```

To use the **old behavior** (single window only), call `.run()` instead of `.run_with_catchup()`.

---

### Updated Summary Table

| Component | Catch-Up Capability | Gap Filling | Auto-Detection | Status |
|-----------|---------------------|-------------|----------------|--------|
| **step_1a** (Class HTML Scraper) | ❌ None | N/A | N/A | Not needed* |
| **step_1b** (HTML Extractor) | ✅ **FULL** | ✅ Yes (with metadata) | ✅ Yes | ✅ **IMPLEMENTED** |
| **step_1c** (Overall Results) | ✅ **FULL** | N/A | ✅ Yes | ✅ **IMPLEMENTED** |
| **step_2** (Table Builder) | ✅ Full | N/A | N/A | Already working |
| **step_3** (Predictions) | ✅ Full | N/A | ✅ Yes | Already working |

*Note: step_1a (individual class HTML scraping) is not needed because step_1c provides historical overall results data, which is sufficient for predictions. Gap filling in step_1b handles missing individual class data.

---

### Complete Catch-Up Workflow

**If you've missed 5 bidding windows:**

1. **Run step_1c with catch-up:**
   ```bash
   python step_1c_ScrapeOverallResults.py
   ```
   - Automatically detects 5 missing windows
   - Logs in once
   - Scrapes all 5 windows
   - Saves to `script_input/overallBossResults/2025-26_T1.xlsx`

2. **Run step_1b with catch-up:**
   ```bash
   python step_1b_HTMLDataExtractor.py
   ```
   - Processes any existing HTML folders
   - Fills gaps for missing folders by carrying forward data
   - Adds metadata flags to identify carried-forward records
   - Saves to `script_input/raw_data.xlsx`

3. **Run step_2 (table builder):**
   ```bash
   python step_2_TableBuilder.py
   ```
   - Processes all available data

4. **Run step_3 (predictions):**
   ```bash
   python step_3_BidPrediction.py
   ```
   - Automatically catches up on all windows with available data

**Total catch-up time:** Single login session + ~5-10 minutes per missed window

---

### Diagnostic Tool

Use `diagnose_missing_windows.py` to check system status:

```bash
python diagnose_missing_windows.py
```

**Output:**
```
======================================================================
STEP 1: CLASS HTML SCRAPING STATUS (step_1a)
======================================================================

✅ EXISTING FOLDERS (3):
   ✅ Round 1 Window 1                         →  500 HTML files
   ✅ Round 1A Window 3                        →  505 HTML files
   ✅ Round 1B Window 1                        →  502 HTML files

❌ MISSING FOLDERS (5):
   ❌ Round 1A Window 1                        → script_input/classTimingsFull/2025-26_T1/2025-26_T1_R1AW1
   ❌ Round 1A Window 2                        → script_input/classTimingsFull/2025-26_T1/2025-26_T1_R1AW2
   ...

======================================================================
CATCH-UP PLAN
======================================================================

STEP 3: Scrape Overall Results (step_1c)
----------------------------------------------------------------------
python step_1c_ScrapeOverallResults.py

✅ This script AUTOMATICALLY catches up on all missing windows!
   It uses run_with_catchup() to detect and scrape missing windows.
```

---

## Questions & Answers

**Q: What happens to carried-forward data when the real HTML is later scraped?**

A: The system will process the new HTML and add fresh records. The deduplication logic in `save_to_excel()` will prefer newer data. You can identify carried-forward data using the `data_source` column and filter/replace as needed.

**Q: Can I manually trigger catch-up for specific windows?**

A: Yes! Set `auto_detect=False` and pass specific parameters:
```python
scraper.run(term='2025-26_T1', bid_round='1A', bid_window='2', auto_detect_phase=False)
```

**Q: Will catch-up mode re-scrape windows that already exist?**

A: No. The `detect_missing_windows()` method checks the Excel file and only scrapes windows that are missing.

**Q: How do I identify carried-forward data in raw_data.xlsx?**

A: Filter by the `data_source` column:
```python
df = pd.read_excel('script_input/raw_data.xlsx', sheet_name='standalone')
carried_forward = df[df['data_source'] == 'carried_forward']
print(f"Carried forward records: {len(carried_forward)}")
```

**Q: Is there a performance impact from processing all windows?**

A: Initial catch-up takes longer, but subsequent runs are fast because the system checks for duplicates and only processes new data.
