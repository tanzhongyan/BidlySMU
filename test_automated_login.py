"""
Test script for automated login flow.

This script tests the automated login functionality using Selenium
to navigate the Microsoft login flow with TOTP-based MFA.
"""

import os
import sys
from dotenv import load_dotenv
from util import setup_driver, perform_automated_login

# Load environment variables
load_dotenv()

def test_automated_login():
    """Test the automated login flow."""
    
    # Check if credentials are configured
    email = os.environ.get('BOSS_EMAIL')
    password = os.environ.get('BOSS_PASSWORD')
    mfa_secret = os.environ.get('BOSS_MFA_SECRET')
    
    print("=" * 60)
    print("Testing Automated Login Flow")
    print("=" * 60)
    
    # Validate credentials
    if not email or email == 'zy.tan.2023@scis.smu.edu.sg':
        print(f"\n[!] Warning: Using default/test email: {email}")
    else:
        print(f"\n[+] Email configured: {email}")
    
    if not password or password == 'Ypnnht8749*2711':
        print("[!] Warning: Using default/test password")
    else:
        print("[+] Password configured: [HIDDEN]")
    
    if not mfa_secret or mfa_secret == 'YOUR_MFA_SECRET_KEY_HERE':
        print("[X] Error: MFA secret not configured properly")
        print("\nPlease set the BOSS_MFA_SECRET environment variable")
        print("To get your MFA secret:")
        print("1. Open Microsoft Authenticator app")
        print("2. Select your SMU account")
        print("3. Tap 'Set up' → 'Can't scan?'")
        print("4. Copy the secret key")
        return False
    else:
        # Show first 4 chars only for security
        print(f"[+] MFA Secret configured: {mfa_secret[:4]}... [HIDDEN]")
    
    print("\n" + "-" * 60)
    print("Starting login test...")
    print("-" * 60)
    
    driver = None
    try:
        # Setup driver (non-headless so you can see what's happening)
        print("\n[>] Setting up Chrome WebDriver...")
        driver = setup_driver(headless=False)
        print("[+] WebDriver initialized")
        
        # Perform automated login
        print("\n[>] Starting automated login flow...")
        print("   Step 1: Navigating to BOSS...")
        print("   Step 2: Microsoft login page...")
        print("   Step 3: SMU ADFS login page...")
        print("   Step 4: MFA challenge...")
        
        username = perform_automated_login(driver)
        
        print("\n" + "=" * 60)
        print("[SUCCESS] LOGIN SUCCESSFUL!")
        print(f"[SUCCESS] Logged in as: {username}")
        print("=" * 60)
        
        # Keep the browser open for a few seconds to verify
        print("\n[>] Keeping browser open for 10 seconds to verify...")
        import time
        time.sleep(10)
        
        return True
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("[FAILED] LOGIN FAILED!")
        print(f"[ERROR] {str(e)}")
        print("=" * 60)
        return False
        
    finally:
        if driver:
            print("\n[>] Closing WebDriver...")
            driver.quit()
            print("[+] WebDriver closed")

if __name__ == "__main__":
    success = test_automated_login()
    sys.exit(0 if success else 1)
