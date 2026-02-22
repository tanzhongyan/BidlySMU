"""
Shared utility functions for BOSS web scraping modules.

This module contains common utilities used by multiple scraper scripts,
including WebDriver setup and manual login handling.
"""

# Import global configuration settings
from config import *

# Import dependencies
import os
import time
import logging
import pyotp
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager


def setup_driver(headless=False):
    """
    Setup Chrome WebDriver with appropriate options.
    
    Args:
        headless (bool): Run browser in headless mode. Default is False.
        
    Returns:
        webdriver.Chrome: Configured Chrome WebDriver instance.
        
    Raises:
        Exception: If WebDriver initialization fails.
    """
    chrome_options = Options()
    
    if headless:
        chrome_options.add_argument("--headless")
    
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        return driver
    except Exception as e:
        raise Exception(f"Failed to initialize WebDriver: {str(e)}")


def wait_for_manual_login(driver, timeout=120, logger=None):
    """
    Wait for manual login and Microsoft Authenticator process completion.
    
    This function waits for the BOSS dashboard to load after the user has
    manually completed the login process. It checks for the presence of
    the username label and sign-out link to confirm successful login.
    
    Args:
        driver (webdriver.Chrome): The Selenium WebDriver instance.
        timeout (int): Maximum time to wait for login in seconds. Default is 120.
        logger (logging.Logger, optional): Logger instance for logging. If None,
                                          print statements will be used.
        
    Returns:
        str: The username of the logged-in user.
        
    Raises:
        Exception: If login fails or times out.
    """
    log = logger.info if logger else print
    
    log("Please log in manually and complete the Microsoft Authenticator process.")
    log("Waiting for BOSS dashboard to load...")
    
    wait = WebDriverWait(driver, timeout)
    
    try:
        # Wait for login success indicators
        wait.until(EC.presence_of_element_located((By.ID, "Label_UserName")))
        wait.until(EC.presence_of_element_located((By.XPATH, "//a[contains(text(),'Sign out')]")))
        
        username = driver.find_element(By.ID, "Label_UserName").text
        log(f"Login successful! Logged in as {username}")
        
    except TimeoutException:
        error_msg = "Login failed or timed out. Could not detect login elements."
        if logger:
            logger.error(error_msg)
        raise Exception(error_msg)
    
    time.sleep(2)
    return username


def get_term_code_map():
    """
    Get the mapping of term codes to BOSS term codes.
    
    Returns:
        dict: Mapping of term abbreviations to BOSS term codes.
              e.g., {'T1': '10', 'T2': '20', 'T3A': '31', 'T3B': '32'}
    """
    return {'T1': '10', 'T2': '20', 'T3A': '31', 'T3B': '32'}


def get_all_terms():
    """
    Get the list of all term abbreviations.
    
    Returns:
        list: List of term abbreviations ['T1', 'T2', 'T3A', 'T3B'].
    """
    return ['T1', 'T2', 'T3A', 'T3B']


def generate_academic_year_range(start_ay_term, end_ay_term):
    """
    Generate a list of academic year terms between start and end terms.
    
    Args:
        start_ay_term (str): Starting academic year term (e.g., '2024-25_T1').
        end_ay_term (str): Ending academic year term (e.g., '2025-26_T2').
        
    Returns:
        list: List of academic year term strings.
        
    Raises:
        ValueError: If start or end term format is invalid.
    """
    term_code_map = get_term_code_map()
    all_terms = get_all_terms()
    
    start_year = int(start_ay_term[:4])
    end_year = int(end_ay_term[:4])
    
    all_academic_years = [
        f"{year}-{(year + 1) % 100:02d}" 
        for year in range(start_year, end_year + 1)
    ]
    
    all_ay_terms = [
        f"{ay}_{term}" 
        for ay in all_academic_years 
        for term in all_terms
    ]
    
    try:
        start_idx = all_ay_terms.index(start_ay_term)
        end_idx = all_ay_terms.index(end_ay_term)
    except ValueError:
        raise ValueError("Invalid start or end term provided.")
    
    return all_ay_terms[start_idx:end_idx+1]


def transform_term_format(short_term):
    """
    Converts a short-form term into the website's full-text format.
    
    Example: '2025-26_T1' -> '2025-26 Term 1'
    
    Args:
        short_term (str): The term in short format (e.g., 'YYYY-YY_TX').
        
    Returns:
        str: The term in the website's format.
        
    Raises:
        ValueError: If the term format is invalid or term suffix is unknown.
    """
    # Mapping from short-form to the website's text.
    term_map = {
        'T1': 'Term 1',
        'T2': 'Term 2',
        'T3A': 'Term 3A',
        'T3B': 'Term 3B'
    }
    
    try:
        # Split the string into the year part and the term part (e.g., '2025-26' and 'T1')
        year_part, term_part = short_term.split('_')
        
        # Look up the full term name from our map.
        full_term_name = term_map.get(term_part)
        
        if full_term_name:
            # Combine them back into the final format.
            return f"{year_part} {full_term_name}"
        else:
            # If the term part is not in our map, raise an error.
            raise ValueError(f"Unknown term suffix: '{term_part}'")
            
    except (ValueError, IndexError) as e:
        raise ValueError(f"Invalid term format: '{short_term}'. Expected format like '2025-26_T1'.")


def get_bidding_round_info_for_term(ay_term, now, bidding_schedule=None):
    """
    Determines the bidding round folder name for a given academic term based on the current time.
    
    Args:
        ay_term (str): Academic year term (e.g., '2024-25_T1').
        now (datetime): Current datetime to check against schedule.
        bidding_schedule (dict, optional): Bidding schedule dictionary. 
                                          If None, uses BIDDING_SCHEDULES from config.
        
    Returns:
        str or None: Folder name suffix if in a bidding window, None otherwise.
                     Format: "{ay_term}_{folder_suffix}"
    """
    if bidding_schedule is None:
        bidding_schedule = BIDDING_SCHEDULES
    
    # Get the schedule for the specific academic term
    schedule = bidding_schedule.get(ay_term)
    if not schedule:
        return None
    
    # Find the correct window from the schedule
    for results_date, _, folder_suffix in schedule:
        if now < results_date:
            return f"{ay_term}_{folder_suffix}"
    
    return None


def setup_logger(name, level=logging.INFO):
    """
    Setup a logger with consistent formatting.
    
    Args:
        name (str): Name of the logger.
        level (int): Logging level. Default is logging.INFO.
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(name)


def perform_automated_login(driver, email=None, password=None, mfa_secret=None, timeout=60, logger=None):
    """
    Perform automated login to BOSS system with TOTP-based MFA.
    
    This function automates the entire login flow:
    1. Navigate to BOSS and get redirected to Microsoft login
    2. Enter email/username on Microsoft login page
    3. Get redirected to SMU ADFS page and enter password
    4. Handle MFA by generating TOTP and entering verification code
    
    The email, password, and MFA secret should be stored in environment variables
    or passed directly as arguments. Environment variables take precedence.
    
    Required environment variables (if not passed as arguments):
        - BOSS_EMAIL: SMU email address (e.g., john.2023@business.smu.edu.sg)
        - BOSS_PASSWORD: SMU account password
        - BOSS_MFA_SECRET: Base32 encoded TOTP secret from Microsoft Authenticator
    
    Args:
        driver (webdriver.Chrome): The Selenium WebDriver instance.
        email (str, optional): SMU email address. If None, uses BOSS_EMAIL env var.
        password (str, optional): SMU password. If None, uses BOSS_PASSWORD env var.
        mfa_secret (str, optional): TOTP secret key. If None, uses BOSS_MFA_SECRET env var.
        timeout (int): Maximum time to wait for each page/element in seconds. Default is 60.
        logger (logging.Logger, optional): Logger instance for logging. If None,
                                          print statements will be used.
        
    Returns:
        str: The username of the logged-in user.
        
    Raises:
        ValueError: If required credentials are not provided.
        Exception: If login fails at any step.
    """
    log = logger.info if logger else print
    
    # Get credentials from environment variables if not provided
    email = email or os.environ.get('BOSS_EMAIL')
    password = password or os.environ.get('BOSS_PASSWORD')
    mfa_secret = mfa_secret or os.environ.get('BOSS_MFA_SECRET')
    
    # Validate credentials
    if not email:
        raise ValueError("Email is required. Provide it as argument or set BOSS_EMAIL environment variable.")
    if not password:
        raise ValueError("Password is required. Provide it as argument or set BOSS_PASSWORD environment variable.")
    if not mfa_secret:
        raise ValueError("MFA secret is required. Provide it as argument or set BOSS_MFA_SECRET environment variable.")
    
    wait = WebDriverWait(driver, timeout)
    
    try:
        log("Starting automated login process...")
        
        # Step 1: Navigate to BOSS (will redirect to Microsoft login)
        log("Navigating to BOSS...")
        driver.get("https://boss.intranet.smu.edu.sg/")
        
        # Step 2: Wait for Microsoft login page and enter email
        log("Waiting for Microsoft login page...")
        wait.until(EC.presence_of_element_located((By.ID, "i0116")))
        
        log(f"Entering email: {email}")
        email_input = driver.find_element(By.ID, "i0116")
        email_input.clear()
        email_input.send_keys(email)
        
        # Click Next button
        next_button = driver.find_element(By.ID, "idSIButton9")
        next_button.click()
        
        # Step 3: Wait for SMU ADFS page and enter password
        log("Waiting for SMU ADFS login page...")
        wait.until(EC.presence_of_element_located((By.ID, "passwordInput")))
        
        log("Entering password...")
        password_input = driver.find_element(By.ID, "passwordInput")
        password_input.clear()
        password_input.send_keys(password)
        
        # Click Sign in button
        signin_button = driver.find_element(By.ID, "submitButton")
        signin_button.click()
        
        # Step 4: Handle MFA - Wait for MFA page
        log("Waiting for MFA challenge...")
        time.sleep(3)  # Brief wait for page to load
        
        # Click "I can't use my Microsoft Authenticator app right now" link
        log("Selecting alternative MFA method...")
        try:
            other_way_link = wait.until(EC.element_to_be_clickable((By.ID, "signInAnotherWay")))
            other_way_link.click()
        except TimeoutException:
            # Maybe we're already on the verification code page
            log("Alternative MFA link not found, checking if already on verification code page...")
        
        # Step 5: Select "Use a verification code" option
        log("Selecting 'Use a verification code' option...")
        time.sleep(2)
        
        # Find the element with data-value="PhoneAppOTP"
        try:
            verification_code_option = wait.until(
                EC.element_to_be_clickable((By.XPATH, "//div[@data-value='PhoneAppOTP']"))
            )
            verification_code_option.click()
        except TimeoutException:
            # Check if we're already on the OTP input page
            log("Verification code option not found, checking if already on OTP input page...")
        
        # Step 6: Generate TOTP and enter it
        log("Generating TOTP code...")
        totp = pyotp.TOTP(mfa_secret)
        current_code = totp.now()
        log(f"Generated TOTP code: {current_code}")
        
        # Wait for OTP input field
        log("Entering verification code...")
        otp_input = wait.until(EC.presence_of_element_located((By.ID, "idTxtBx_SAOTCC_OTC")))
        otp_input.clear()
        otp_input.send_keys(current_code)
        
        # Step 7: Click Verify button
        log("Clicking Verify button...")
        verify_button = driver.find_element(By.ID, "idSubmit_SAOTCC_Continue")
        verify_button.click()
        
        # Step 8: Wait for successful login (BOSS dashboard)
        log("Waiting for BOSS dashboard to load...")
        wait.until(EC.presence_of_element_located((By.ID, "Label_UserName")))
        
        username = driver.find_element(By.ID, "Label_UserName").text
        log(f"Login successful! Logged in as {username}")
        
        time.sleep(2)
        return username
        
    except TimeoutException as e:
        error_msg = f"Login timed out. Element not found: {str(e)}"
        if logger:
            logger.error(error_msg)
        raise Exception(error_msg)
    except Exception as e:
        error_msg = f"Automated login failed: {str(e)}"
        if logger:
            logger.error(error_msg)
        raise Exception(error_msg)
