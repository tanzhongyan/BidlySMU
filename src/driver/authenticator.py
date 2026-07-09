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
from typing import Callable, Optional
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
    def login(self, driver: WebDriver) -> WebDriver:
        """
        Perform login and return the WebDriver.

        The returned driver may differ from the input driver if the login
        process creates a fresh session (e.g. on retry). Callers MUST
        use the returned driver for subsequent operations.

        Args:
            driver: Pre-configured WebDriver at BOSS login page

        Returns:
            WebDriver: The driver to use after login (may be new)

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

    def login(self, driver: WebDriver) -> WebDriver:
        """
        Wait for user to manually log in and complete Microsoft Authenticator process.

        Args:
            driver: WebDriver at BOSS login page

        Returns:
            WebDriver: The driver (unchanged for manual login)

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
        return driver


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
        driver_factory: Optional[Callable[[], WebDriver]] = None,
    ):
        self._credentials = credentials
        self._timeout = timeout
        self._logger = logger or get_logger(__name__)
        self._driver_factory = driver_factory

    def _capture_failure_context(self, driver: WebDriver, label: str) -> None:
        """Capture screenshot + URL and log as ERROR for Sentry visibility."""
        try:
            current_url = driver.current_url
            timestamp = int(time.time())

            # Save screenshot to /tmp/login_debug/ as fallback
            try:
                import os as _os
                _os.makedirs("/tmp/login_debug", exist_ok=True)
                path = f"/tmp/login_debug/failure_{label}_{timestamp}.png"
                driver.save_screenshot(path)
                self._logger.warning(f"Screenshot saved: {path}")
            except Exception:
                pass

            # Sentry picks up ERROR-level logs automatically
            self._logger.error(
                f"Login failure [{label}] at URL: {current_url[:300]} "
                f"(timestamp: {timestamp})"
            )
        except Exception as screenshot_err:
            self._logger.warning(f"Could not capture failure context: {screenshot_err}")

    def _do_login(self, driver: WebDriver) -> str:
        """
        Execute the full 8-step Microsoft Entra ID login flow.

        Args:
            driver: WebDriver instance

        Returns:
            str: Username of logged-in user

        Raises:
            Exception: If login fails at any step
        """
        creds = self._credentials
        self._logger.info("Starting automated login process...")

        wait = WebDriverWait(driver, self._timeout)

        try:
            # Step 1: Navigate to BOSS (redirects to Microsoft login)
            self._logger.info("Navigating to BOSS...")
            driver.get("https://boss.intranet.smu.edu.sg/")

            # Step 2: Enter email on Microsoft login page
            self._logger.info("Waiting for Microsoft login page...")
            try:
                wait.until(EC.presence_of_element_located((By.ID, "i0116")))
            except TimeoutException:
                self._capture_failure_context(driver, "ms_login")
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
                self._capture_failure_context(driver, "adfs_password")
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
                self._capture_failure_context(driver, "adfs_submit")
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
                self._capture_failure_context(driver, "otp_missing")
                raise Exception(
                    f"Step 6 failed: OTP input field not found. "
                    f"Tried: {[s[1] for s in otp_selectors]}. URL: {driver.current_url}"
                )
            otp_input.clear()
            otp_input.send_keys(code)

            # Step 7: Click Verify/Submit button (wait for clickable to avoid ElementNotInteractableException)
            self._logger.info("Clicking Verify button...")
            verify_button = None
            verify_selectors = [
                (By.ID, "idSubmit_SAOTCC_Continue"),
                (By.ID, "idSIButton9"),
                (By.XPATH, "//input[@type='submit']"),
            ]
            for by, selector in verify_selectors:
                try:
                    verify_button = WebDriverWait(driver, 5).until(
                        EC.element_to_be_clickable((by, selector))
                    )
                    self._logger.info(f"Found verify button: {by}={selector}")
                    break
                except TimeoutException:
                    continue

            if verify_button is None:
                self._capture_failure_context(driver, "verify_missing")
                raise Exception(
                    f"Step 7 failed: Verify button not found or not clickable. "
                    f"Tried: {[s[1] for s in verify_selectors]}. URL: {driver.current_url}"
                )
            verify_button.click()
            self._logger.info("Verify clicked. Waiting for form_post redirect...")

            # Step 8: Wait for the OAuth form_post to redirect us to BOSS.
            # With anti-detection flags, the redirect should complete naturally.
            # If it doesn't within 10s, navigate directly as fallback.
            try:
                WebDriverWait(driver, 10).until(
                    lambda d: "login.microsoftonline.com" not in d.current_url
                )
                self._logger.info(
                    f"form_post redirect completed: {driver.current_url[:120]}"
                )
            except TimeoutException:
                self._logger.warning(
                    "form_post redirect did not complete, "
                    "navigating directly as fallback..."
                )
                driver.get("https://boss.intranet.smu.edu.sg/")
                time.sleep(5)

            self._logger.info(f"Current URL: {driver.current_url[:120]}")

            # If still on Microsoft after everything, session failed
            if "login.microsoftonline.com" in driver.current_url:
                self._capture_failure_context(driver, "boss_rejected")
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
                self._capture_failure_context(driver, "boss_dashboard")
                raise Exception(
                    f"BOSS dashboard did not load. URL: {driver.current_url[:120]}"
                )

            username = driver.find_element(By.ID, "Label_UserName").text
            self._logger.info(f"Login successful! Logged in as {username}")

            time.sleep(2)
            return username

        except TimeoutException as e:
            self._capture_failure_context(driver, "timeout")
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

    def login(self, driver: WebDriver) -> WebDriver:
        """
        Perform automated TOTP-based login to BOSS with retry on failure.

        Retries up to 3 times with exponential backoff (10s, 20s, 40s).
        Each retry creates a fresh WebDriver session via driver_factory.
        If driver_factory is not provided, falls back to single-attempt behavior.

        IMPORTANT: Returns the WebDriver to use after login. The returned driver
        may be a new instance if a retry occurred. Callers MUST use the returned
        driver — the input driver may have been quit and replaced.

        Args:
            driver: WebDriver instance (replaced with fresh instance on retry)

        Returns:
            WebDriver: The driver to use after login (may be new)

        Raises:
            ValueError: If credentials are invalid
            Exception: If login fails after all retry attempts
        """
        max_attempts = 3
        backoff_seconds = [10, 20, 40]
        last_exception = None
        current_driver = driver

        for attempt in range(max_attempts):
            try:
                self._do_login(current_driver)
                return current_driver
            except Exception as e:
                last_exception = e
                # First 2 attempts are expected to fail due to Entra ID redirect
                # timing — log as WARNING to avoid Sentry noise. Only the final
                # exhausted-retries failure is a genuine ERROR.
                if attempt < max_attempts - 1:
                    self._logger.warning(
                        f"Login attempt {attempt + 1}/{max_attempts} failed: {e}"
                    )
                else:
                    self._logger.error(
                        f"Login attempt {attempt + 1}/{max_attempts} failed: {e}"
                    )

                # Quit the tainted driver before creating a fresh one
                try:
                    current_driver.quit()
                except Exception:
                    pass

                # If we can retry with a fresh driver, do so
                if attempt < max_attempts - 1:
                    if self._driver_factory is None:
                        self._logger.error(
                            "No driver_factory available; cannot retry with fresh session. "
                            "Provide driver_factory to AutomatedLogin to enable retries."
                        )
                        raise

                    sleep_sec = backoff_seconds[attempt]
                    self._logger.info(
                        f"Waiting {sleep_sec}s before retry {attempt + 2}/{max_attempts}..."
                    )
                    time.sleep(sleep_sec)

                    self._logger.info("Creating fresh WebDriver for retry...")
                    current_driver = self._driver_factory()

        raise Exception(
            f"Login failed after {max_attempts} attempts. "
            f"Last error: {last_exception}"
        )