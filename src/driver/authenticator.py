"""
Authenticator interface and implementations for BOSS login.

Usage:
    # Manual login
    authenticator = ManualLogin(timeout=120)
    username = authenticator.login(driver)

    # Automated login with credentials
    credentials = AuthCredentials(
        email="test@business.smu.edu.sg",
        password="password",
        mfa_secret="SECRET"
    )
    authenticator = AutomatedLogin(credentials)
    username = authenticator.login(driver)

    # Pass as abstraction
    def run(self, authenticator: Authenticator, driver: WebDriver):
        authenticator.login(driver)
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import os
import time
import logging

from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import pyotp


from src.logging.logger import get_logger

@dataclass(frozen=True)
class AuthCredentials:
    """
    Immutable credentials DTO for automated login.

    Usage:
        credentials = AuthCredentials(
            email="test@business.smu.edu.sg",
            password="password",
            mfa_secret="BASE32SECRET"
        )
    """
    email: str
    password: str
    mfa_secret: str

    @classmethod
    def from_environment(cls) -> "AuthCredentials":
        """
        Create credentials from environment variables.

        Required env vars:
            - BOSS_EMAIL: SMU email address
            - BOSS_PASSWORD: SMU account password
            - BOSS_MFA_SECRET: Base32 TOTP secret

        Returns:
            AuthCredentials instance

        Raises:
            ValueError: If required env vars are missing
        """
        email = os.environ.get("BOSS_EMAIL")
        password = os.environ.get("BOSS_PASSWORD")
        mfa_secret = os.environ.get("BOSS_MFA_SECRET")

        missing = []
        if not email:
            missing.append("BOSS_EMAIL")
        if not password:
            missing.append("BOSS_PASSWORD")
        if not mfa_secret:
            missing.append("BOSS_MFA_SECRET")

        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}"
            )

        return cls(email=email, password=password, mfa_secret=mfa_secret)


class Authenticator(ABC):
    """
    Abstract interface for login strategies.

    Implementations handle different login mechanisms (manual, automated TOTP, etc.)
    """

    @abstractmethod
    def login(self, driver: WebDriver) -> str:
        """
        Perform login and return username.

        Args:
            driver: Pre-configured WebDriver at BOSS login page

        Returns:
            str: Username of logged-in user

        Raises:
            Exception: If login fails
        """
        pass


class ManualLogin(Authenticator):
    """
    Manual login authenticator - waits for user to complete login process.

    Usage:
        authenticator = ManualLogin(timeout=120, logger=logging.getLogger(__name__))
        username = authenticator.login(driver)
    """

    def __init__(
        self,
        timeout: int = 120,
        logger: Optional[logging.Logger] = None,
    ):
        self._timeout = timeout
        self._logger = logger or get_logger(__name__)

    def login(self, driver: WebDriver) -> str:
        """
        Wait for user to manually log in and complete Microsoft Authenticator process.

        Args:
            driver: WebDriver at BOSS login page

        Returns:
            str: Username of logged-in user

        Raises:
            Exception: If login fails or times out
        """
        self._logger.info(
            "Please log in manually and complete the Microsoft Authenticator process."
        )
        self._logger.info("Waiting for BOSS dashboard to load...")

        wait = WebDriverWait(driver, self._timeout)

        try:
            # Wait for login success indicators
            wait.until(EC.presence_of_element_located((By.ID, "Label_UserName")))
            wait.until(EC.presence_of_element_located((By.XPATH, "//a[contains(text(),'Sign out')]")))

            username = driver.find_element(By.ID, "Label_UserName").text
            self._logger.info(f"Login successful! Logged in as {username}")

        except TimeoutException:
            raise Exception("Login failed or timed out. Could not detect login elements.")

        time.sleep(2)
        return username


class AutomatedLogin(Authenticator):
    """
    Automated TOTP-based login authenticator.

    Usage:
        credentials = AuthCredentials.from_environment()
        authenticator = AutomatedLogin(credentials, logger=logging.getLogger(__name__))
        username = authenticator.login(driver)
    """

    def __init__(
        self,
        credentials: AuthCredentials,
        timeout: int = 60,
        logger: Optional[logging.Logger] = None,
    ):
        self._credentials = credentials
        self._timeout = timeout
        self._logger = logger or get_logger(__name__)

    def login(self, driver: WebDriver) -> str:
        """
        Perform automated TOTP-based login to BOSS.

        Args:
            driver: WebDriver instance

        Returns:
            str: Username of logged-in user

        Raises:
            ValueError: If credentials are invalid
            Exception: If login fails at any step
        """
        creds = self._credentials
        self._logger.info("Starting automated login process...")

        wait = WebDriverWait(driver, self._timeout)

        def _save_debug_screenshot(label: str):
            """Save a screenshot for debugging login failures."""
            try:
                path = f"/tmp/login_failure_{label}_{int(time.time())}.png"
                driver.save_screenshot(path)
                self._logger.error(f"Screenshot saved: {path}, URL: {driver.current_url}")
            except Exception:
                pass

        try:
            # Step 1: Navigate to BOSS (redirects to Microsoft login)
            self._logger.info("Navigating to BOSS...")
            driver.get("https://boss.intranet.smu.edu.sg/")

            # Step 2: Enter email on Microsoft login page
            self._logger.info("Waiting for Microsoft login page...")
            try:
                wait.until(EC.presence_of_element_located((By.ID, "i0116")))
            except TimeoutException:
                _save_debug_screenshot("ms_login")
                raise Exception("Step 2 failed: Microsoft login page (i0116) did not load. URL: " + driver.current_url)
            time.sleep(1.5)

            self._logger.info(f"Entering email: {creds.email}")
            email_input = driver.find_element(By.ID, "i0116")
            email_input.clear()
            email_input.send_keys(creds.email)

            # Click Next button
            time.sleep(1)
            driver.find_element(By.ID, "idSIButton9").click()

            # Step 3: Enter password on SMU ADFS page
            # Try multiple possible selectors for the password field (SMU may change the page)
            self._logger.info("Waiting for SMU ADFS login page...")
            password_input = None
            password_selectors = [
                (By.ID, "passwordInput"),       # Old SMU ADFS page
                (By.ID, "i0118"),               # Microsoft password field
                (By.NAME, "passwd"),            # Generic Microsoft login
                (By.XPATH, "//input[@type='password']"),  # Any password field
            ]
            for by, selector in password_selectors:
                try:
                    password_input = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((by, selector))
                    )
                    self._logger.info(f"Found password field: {by}={selector}")
                    break
                except TimeoutException:
                    continue

            if password_input is None:
                _save_debug_screenshot("adfs_password")
                raise Exception(
                    f"Step 3 failed: SMU ADFS password field not found. "
                    f"Tried: {[s[1] for s in password_selectors]}. URL: {driver.current_url}"
                )
            time.sleep(1.5)

            self._logger.info("Entering password...")
            password_input.clear()
            password_input.send_keys(creds.password)

            # Click Sign in button — try multiple selectors
            submit_button = None
            submit_selectors = [
                (By.ID, "submitButton"),        # Old SMU ADFS
                (By.ID, "idSIButton9"),         # Microsoft Next/Sign in
                (By.XPATH, "//input[@type='submit']"),
                (By.XPATH, "//button[@type='submit']"),
            ]
            time.sleep(1)
            for by, selector in submit_selectors:
                try:
                    submit_button = driver.find_element(by, selector)
                    self._logger.info(f"Found submit button: {by}={selector}")
                    break
                except Exception:
                    continue

            if submit_button is None:
                _save_debug_screenshot("adfs_submit")
                raise Exception(
                    f"Step 3 failed: Submit button not found. "
                    f"Tried: {[s[1] for s in submit_selectors]}. URL: {driver.current_url}"
                )
            submit_button.click()

            # Step 4: Handle MFA
            self._logger.info("Waiting for MFA challenge...")
            time.sleep(3)

            # Select alternative MFA method (if Microsoft "Verify your identity" page)
            try:
                other_way_link = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.ID, "signInAnotherWay"))
                )
                time.sleep(1)
                other_way_link.click()
                self._logger.info("Clicked 'Sign in another way'")
            except TimeoutException:
                self._logger.info("Alternative MFA link not found, checking for OTP directly...")

            # Step 5: Select "Use a verification code" option
            time.sleep(2)
            try:
                verification_code_option = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.XPATH, "//div[@data-value='PhoneAppOTP']"))
                )
                time.sleep(1)
                verification_code_option.click()
                self._logger.info("Selected 'Use a verification code'")
            except TimeoutException:
                self._logger.info("OTP option selector not found, checking if already on OTP page...")

            # Step 6: Generate TOTP and enter it
            self._logger.info("Generating TOTP code...")
            totp = pyotp.TOTP(creds.mfa_secret)
            code = totp.now()
            self._logger.info(f"Generated TOTP code: {code}")

            # Wait for OTP input field — try multiple selectors
            self._logger.info("Entering verification code...")
            otp_input = None
            otp_selectors = [
                (By.ID, "idTxtBx_SAOTCC_OTC"),      # Microsoft OTP field
                (By.ID, "i0118"),                     # Microsoft password/OTP
                (By.NAME, "otc"),                     # Generic OTC
                (By.XPATH, "//input[@type='tel']"),   # TOTP often uses tel input
            ]
            for by, selector in otp_selectors:
                try:
                    otp_input = WebDriverWait(driver, 8).until(
                        EC.presence_of_element_located((by, selector))
                    )
                    self._logger.info(f"Found OTP field: {by}={selector}")
                    break
                except TimeoutException:
                    continue

            if otp_input is None:
                _save_debug_screenshot("otp_missing")
                raise Exception(
                    f"Step 6 failed: OTP input field not found. "
                    f"Tried: {[s[1] for s in otp_selectors]}. URL: {driver.current_url}"
                )
            otp_input.clear()
            otp_input.send_keys(code)

            # Step 7: Click Verify/Submit button
            self._logger.info("Clicking Verify button...")
            verify_button = None
            verify_selectors = [
                (By.ID, "idSubmit_SAOTCC_Continue"),
                (By.ID, "idSIButton9"),
                (By.XPATH, "//input[@type='submit']"),
            ]
            for by, selector in verify_selectors:
                try:
                    verify_button = driver.find_element(by, selector)
                    self._logger.info(f"Found verify button: {by}={selector}")
                    break
                except Exception:
                    continue

            if verify_button is None:
                _save_debug_screenshot("verify_missing")
                raise Exception(
                    f"Step 7 failed: Verify button not found. "
                    f"Tried: {[s[1] for s in verify_selectors]}. URL: {driver.current_url}"
                )
            verify_button.click()
            self._logger.info("Verify clicked. Waiting for session cookies...")
            time.sleep(5)

            # Step 8: Navigate directly to BOSS (session is already authenticated)
            # The Microsoft OAuth form_post redirect may fail in Selenium,
            # but the session cookies are set after successful MFA.
            self._logger.info("Navigating directly to BOSS dashboard...")
            driver.get("https://boss.intranet.smu.edu.sg/")
            time.sleep(5)
            self._logger.info(f"After direct navigation: {driver.current_url[:120]}")

            # If redirected back to Microsoft, the session didn't stick
            if "login.microsoftonline.com" in driver.current_url:
                _save_debug_screenshot("boss_rejected")
                raise Exception(
                    f"BOSS rejected direct navigation — session not authenticated. "
                    f"URL: {driver.current_url[:120]}"
                )

            # Wait for BOSS dashboard
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.ID, "Label_UserName"))
                )
            except TimeoutException:
                _save_debug_screenshot("boss_dashboard")
                raise Exception(
                    f"BOSS dashboard did not load. URL: {driver.current_url[:120]}"
                )

            username = driver.find_element(By.ID, "Label_UserName").text
            self._logger.info(f"Login successful! Logged in as {username}")

            time.sleep(2)
            return username

        except TimeoutException as e:
            _save_debug_screenshot("timeout")
            error_msg = (
                f"Login timed out. Element: {str(e)[:200]}. URL: {driver.current_url}"
            )
            self._logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            import traceback
            traceback.print_exc()
            error_msg = f"Automated login failed: {str(e)}"
            self._logger.error(error_msg)
            raise