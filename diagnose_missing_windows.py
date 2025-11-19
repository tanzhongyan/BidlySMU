#!/usr/bin/env python3
"""
Diagnostic Script: Check for Missing Bidding Windows
======================================================

This script analyzes your BidlySMU data to identify which bidding windows
are missing and need to be scraped for catch-up.

Usage:
    python diagnose_missing_windows.py
"""

from config import BIDDING_SCHEDULES, START_AY_TERM
from datetime import datetime
import os
import pandas as pd
from pathlib import Path

def check_html_folders():
    """Check which HTML folders exist"""
    print("=" * 70)
    print("STEP 1: CLASS HTML SCRAPING STATUS (step_1a)")
    print("=" * 70)

    schedule = BIDDING_SCHEDULES.get(START_AY_TERM, [])
    now = datetime.now()

    missing_folders = []
    existing_folders = []
    future_windows = []

    for schedule_time, window_name, folder_suffix in schedule:
        folder_path = f"script_input/classTimingsFull/{START_AY_TERM}/{START_AY_TERM}_{folder_suffix}"

        if schedule_time > now:
            future_windows.append((window_name, folder_path))
        elif os.path.exists(folder_path):
            # Count HTML files
            try:
                html_count = len([f for f in os.listdir(folder_path) if f.endswith('.html')])
                existing_folders.append((window_name, folder_path, html_count))
            except:
                existing_folders.append((window_name, folder_path, 0))
        else:
            missing_folders.append((window_name, folder_path))

    print(f"\n✅ EXISTING FOLDERS ({len(existing_folders)}):")
    for window_name, folder_path, html_count in existing_folders:
        print(f"   ✅ {window_name:40s} → {html_count:4d} HTML files")

    print(f"\n❌ MISSING FOLDERS ({len(missing_folders)}):")
    if missing_folders:
        for window_name, folder_path in missing_folders:
            print(f"   ❌ {window_name:40s} → {folder_path}")
    else:
        print("   None! All past windows have been scraped.")

    print(f"\n⏳ FUTURE WINDOWS ({len(future_windows)}):")
    for window_name, folder_path in future_windows:
        print(f"   ⏳ {window_name:40s} (not yet started)")

    return missing_folders, existing_folders

def check_raw_data_excel():
    """Check which windows are in raw_data.xlsx"""
    print("\n" + "=" * 70)
    print("STEP 2: HTML EXTRACTION STATUS (step_1b)")
    print("=" * 70)

    excel_path = 'script_input/raw_data.xlsx'

    if not os.path.exists(excel_path):
        print(f"\n❌ {excel_path} does not exist!")
        print("   You need to run step_1b to extract data from HTML files.")
        return set()

    try:
        df = pd.read_excel(excel_path, sheet_name='standalone')

        if 'bidding_window' not in df.columns:
            print("\n⚠️ 'bidding_window' column not found in raw_data.xlsx")
            return set()

        windows_in_excel = df['bidding_window'].dropna().unique()

        print(f"\n✅ Windows in raw_data.xlsx: {len(windows_in_excel)}")
        print(f"   Total records: {len(df)}")

        # Group by window and count
        window_counts = df.groupby('bidding_window').size().sort_index()

        for window in window_counts.index:
            count = window_counts[window]
            print(f"   ✅ {window:40s} → {count:4d} records")

        # Check for windows that should be there but aren't
        schedule = BIDDING_SCHEDULES.get(START_AY_TERM, [])
        now = datetime.now()

        expected_windows = set()
        for schedule_time, window_name, folder_suffix in schedule:
            if schedule_time < now:
                expected_windows.add(window_name)

        missing_from_excel = expected_windows - set(windows_in_excel)

        if missing_from_excel:
            print(f"\n❌ MISSING FROM EXCEL ({len(missing_from_excel)}):")
            for window in sorted(missing_from_excel):
                print(f"   ❌ {window}")
        else:
            print(f"\n✅ All past windows are in raw_data.xlsx!")

        return set(windows_in_excel)

    except Exception as e:
        print(f"\n❌ Error reading {excel_path}: {e}")
        return set()

def check_overall_results():
    """Check which windows are in overallBossResults Excel"""
    print("\n" + "=" * 70)
    print("STEP 3: OVERALL BOSS RESULTS STATUS (step_1c)")
    print("=" * 70)

    excel_path = f'script_input/overallBossResults/{START_AY_TERM}.xlsx'

    if not os.path.exists(excel_path):
        print(f"\n❌ {excel_path} does not exist!")
        print("   You need to run step_1c to scrape overall results.")
        return set()

    try:
        df = pd.read_excel(excel_path)

        if 'Bidding Window' not in df.columns:
            print("\n⚠️ 'Bidding Window' column not found in Excel")
            return set()

        windows_in_excel = df['Bidding Window'].dropna().unique()

        print(f"\n✅ Windows in overallBossResults: {len(windows_in_excel)}")
        print(f"   Total records: {len(df)}")

        # Group by window and count
        window_counts = df.groupby('Bidding Window').size().sort_index()

        for window in window_counts.index:
            count = window_counts[window]
            print(f"   ✅ {window:40s} → {count:4d} records")

        # Check for windows that should be there but aren't
        schedule = BIDDING_SCHEDULES.get(START_AY_TERM, [])
        now = datetime.now()

        expected_windows = set()
        for schedule_time, window_name, folder_suffix in schedule:
            if schedule_time < now:
                expected_windows.add(window_name)

        missing_from_excel = expected_windows - set(windows_in_excel)

        if missing_from_excel:
            print(f"\n❌ MISSING FROM EXCEL ({len(missing_from_excel)}):")
            for window in sorted(missing_from_excel):
                print(f"   ❌ {window}")
        else:
            print(f"\n✅ All past windows are in overallBossResults!")

        return set(windows_in_excel)

    except Exception as e:
        print(f"\n❌ Error reading {excel_path}: {e}")
        return set()

def check_predictions():
    """Check database for existing predictions"""
    print("\n" + "=" * 70)
    print("STEP 4: PREDICTIONS STATUS (step_3)")
    print("=" * 70)

    try:
        from dotenv import load_dotenv
        import psycopg2

        load_dotenv()
        db_config = {
            'host': os.getenv('DB_HOST'),
            'database': os.getenv('DB_NAME'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD'),
            'port': int(os.getenv('DB_PORT', 5432))
        }

        connection = psycopg2.connect(**db_config)

        # Query bid windows
        query = """
        SELECT bw.round, bw.window, bw.id, COUNT(bp.id) as prediction_count
        FROM bid_window bw
        LEFT JOIN "BidPrediction" bp ON bw.id = bp."bidWindowId"
        WHERE bw.acad_term_id = %s
        GROUP BY bw.round, bw.window, bw.id
        ORDER BY bw.round, bw.window
        """

        # Convert START_AY_TERM to database format (e.g., '2025-26_T1' -> 'AY202526T1')
        import re
        match = re.match(r'(\d{4})-(\d{2})_T(\d+)', START_AY_TERM)
        if match:
            acad_term_id = f"AY{match.group(1)}{match.group(2)}T{match.group(3)}"
        else:
            acad_term_id = START_AY_TERM

        cursor = connection.cursor()
        cursor.execute(query, (acad_term_id,))
        results = cursor.fetchall()

        print(f"\n✅ Bid windows in database:")
        windows_with_predictions = 0
        windows_without_predictions = 0

        for round_val, window_val, window_id, pred_count in results:
            if pred_count > 0:
                print(f"   ✅ Round {round_val} Window {window_val:2d} → {pred_count:5d} predictions")
                windows_with_predictions += 1
            else:
                print(f"   ⚠️ Round {round_val} Window {window_val:2d} → No predictions yet")
                windows_without_predictions += 1

        print(f"\n📊 Summary:")
        print(f"   Windows with predictions: {windows_with_predictions}")
        print(f"   Windows without predictions: {windows_without_predictions}")

        cursor.close()
        connection.close()

    except ImportError:
        print("\n⚠️ psycopg2 not installed. Skipping database check.")
        print("   Install with: pip install psycopg2-binary")
    except Exception as e:
        print(f"\n⚠️ Could not check database: {e}")
        print("   This is optional - predictions check skipped.")

def generate_catch_up_plan():
    """Generate a catch-up plan based on missing data"""
    print("\n" + "=" * 70)
    print("CATCH-UP PLAN")
    print("=" * 70)

    schedule = BIDDING_SCHEDULES.get(START_AY_TERM, [])
    now = datetime.now()

    print("\n📋 To catch up on missing data, follow these steps:\n")

    # Step 1: Identify missing windows
    past_windows = []
    for schedule_time, window_name, folder_suffix in schedule:
        if schedule_time < now:
            past_windows.append((window_name, folder_suffix))

    if not past_windows:
        print("✅ No past windows found. You're up to date!")
        return

    print("STEP 1: Scrape Class HTML (step_1a)")
    print("-" * 70)
    print("Run the following command for EACH missing window:\n")
    print("python step_1a_BOSSClassScraper.py")
    print("\n⚠️ LIMITATION: Current script only scrapes current window!")
    print("⚠️ You may need to manually adjust BIDDING_SCHEDULES in config.py")
    print("   to trick the script into thinking each past window is 'current'.\n")

    print("\nSTEP 2: Extract HTML Data (step_1b)")
    print("-" * 70)
    print("After scraping HTML, run:\n")
    print("python step_1b_HTMLDataExtractor.py")
    print("\n⚠️ LIMITATION: Only processes latest round folder!")
    print("⚠️ Modify process_all_files() to process specific folders.\n")

    print("\nSTEP 3: Scrape Overall Results (step_1c)")
    print("-" * 70)
    print("For EACH missing window, run:\n")

    for window_name, folder_suffix in past_windows:
        # Parse window name to extract round and window number
        import re
        match = re.search(r'Round\s+(\d+[A-Z]?)\s+Window\s+(\d+)', window_name)
        if match:
            round_val = match.group(1)
            window_val = match.group(2)
        else:
            # Try alternative format
            match = re.search(r'Rnd\s+(\d+[A-Z]?)\s+Win\s+(\d+)', window_name)
            if match:
                round_val = match.group(1)
                window_val = match.group(2)
            else:
                round_val = "?"
                window_val = "?"

        print(f"# {window_name}")
        print(f"python step_1c_ScrapeOverallResults.py --round {round_val} --window {window_val}")
        print()

    print("\n⚠️ You'll need to modify step_1c to accept command-line arguments,")
    print("   or manually edit config.py TARGET_ROUND and TARGET_WINDOW for each run.\n")

    print("\nSTEP 4: Process Data (step_2)")
    print("-" * 70)
    print("Run once after all windows are scraped:\n")
    print("python step_2_TableBuilder.py\n")

    print("\nSTEP 5: Generate Predictions (step_3)")
    print("-" * 70)
    print("Run to generate predictions (has built-in catch-up!):\n")
    print("python step_3_BidPrediction.py")
    print("\n✅ This script AUTOMATICALLY catches up on all missing predictions!")
    print("   It will process all windows from start to current that are in raw_data.xlsx\n")

def main():
    """Main diagnostic function"""
    print("\n" + "=" * 70)
    print("BIDLY SMU - MISSING WINDOWS DIAGNOSTIC")
    print("=" * 70)
    print(f"\nCurrent time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target term: {START_AY_TERM}")
    print()

    # Run all checks
    check_html_folders()
    check_raw_data_excel()
    check_overall_results()
    check_predictions()
    generate_catch_up_plan()

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)
    print("\nSee CATCH_UP_ANALYSIS.md for detailed solutions and recommendations.")
    print()

if __name__ == "__main__":
    main()
