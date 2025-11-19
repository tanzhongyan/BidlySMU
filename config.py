from datetime import datetime

# ============================================================================
# STORAGE MODE CONFIGURATION
# ============================================================================
# Choose how scraped data should be stored:
# - 'excel': Save to Excel files (step_1b → raw_data.xlsx, step_1c → overallBossResults/*.xlsx)
#            Then use step_2 to process and upload to database
# - 'database': Directly insert to database (bypasses Excel files, faster for large datasets)
#
# Default: 'excel' (recommended for easier debugging and manual inspection)
STORAGE_MODE = 'excel'  # Options: 'excel' or 'database'

# ============================================================================
# ACADEMIC TERM CONFIGURATION
# ============================================================================
# Define the academic term range you want to scrape or process.
# For a single term, set both START and END to the same value.
START_AY_TERM = '2025-26_T1'
END_AY_TERM = '2025-26_T1'
ACAD_TERM_ID = 'AY202526T1'

# Define the specific bidding round and window you want to target.
# Set to None to let the script auto-detect the current phase based on the schedule.
TARGET_ROUND = None   # e.g., '1A', '2', etc.
TARGET_WINDOW = None  # e.g., 1, 2, 3, etc.

# Central bidding schedule for each academic term.
# The script will use this to determine the correct folder names and bidding phases.
# Format: (results_datetime, "Full Bidding Window Name", "Folder_Suffix")
BIDDING_SCHEDULES = {
    '2025-26_T1': [
        (datetime(2025, 7, 9, 14, 0), "Round 1 Window 1", "R1W1"),
        (datetime(2025, 7, 11, 14, 0), "Round 1A Window 1", "R1AW1"),
        (datetime(2025, 7, 14, 14, 0), "Round 1A Window 2", "R1AW2"),
        (datetime(2025, 7, 16, 14, 0), "Round 1A Window 3", "R1AW3"),
        (datetime(2025, 7, 18, 14, 0), "Round 1B Window 1", "R1BW1"),
        (datetime(2025, 7, 21, 14, 0), "Round 1B Window 2", "R1BW2"),
        (datetime(2025, 7, 30, 14, 0), "Incoming Exchange Rnd 1C Win 1", "R1CW1"),
        (datetime(2025, 7, 31, 14, 0), "Incoming Exchange Rnd 1C Win 2", "R1CW2"),
        (datetime(2025, 8, 1, 14, 0), "Incoming Exchange Rnd 1C Win 3", "R1CW3"),
        (datetime(2025, 8, 11, 14, 0), "Incoming Freshmen Rnd 1 Win 1", "R1FW1"),
        (datetime(2025, 8, 12, 14, 0), "Incoming Freshmen Rnd 1 Win 2", "R1FW2"),
        (datetime(2025, 8, 13, 14, 0), "Incoming Freshmen Rnd 1 Win 3", "R1FW3"),
        (datetime(2025, 8, 14, 14, 0), "Incoming Freshmen Rnd 1 Win 4", "R1FW4"),
        (datetime(2025, 8, 20, 14, 0), "Round 2 Window 1", "R2W1"),
        (datetime(2025, 8, 22, 14, 0), "Round 2 Window 2", "R2W2"),
        (datetime(2025, 8, 25, 14, 0), "Round 2 Window 3", "R2W3"),
        (datetime(2025, 8, 27, 14, 0), "Round 2A Window 1", "R2AW1"),
        (datetime(2025, 8, 29, 14, 0), "Round 2A Window 2", "R2AW2"),
        (datetime(2025, 9, 1, 14, 0), "Round 2A Window 3", "R2AW3"),
    ]
    # You can add schedules for other terms here, e.g., '2025-26_T2': [...]
}