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

        try:
            # Step 1: Navigate to BOSS (redirects to Microsoft login)
            self._logger.info("Navigating to BOSS...")
            driver.get("https://boss.intranet.smu.edu.sg/")

            # Step 2: Enter email on Microsoft login page
            self._logger.info("Waiting for Microsoft login page...")
            wait.until(EC.presence_of_element_located((By.ID, "i0116")))
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
            wait.until(EC.presence_of_element_located((By.ID, "passwordInput")))
            time.sleep(1.5)

            self._logger.info("Entering password...")
            password_input = driver.find_element(By.ID, "passwordInput")
            password_input.clear()
            password_input.send_keys(creds.password)

            # Click Sign in button
            time.sleep(1)
            driver.find_element(By.ID, "submitButton").click()

            # Step 4: Handle MFA
            self._logger.info("Waiting for MFA challenge...")
            time.sleep(3)

            # Select alternative MFA method
            try:
                other_way_link = wait.until(
                    EC.presence_of_element_located((By.ID, "signInAnotherWay"))
                )
                time.sleep(1)
                other_way_link.click()
            except TimeoutException:
                self._logger.info("Alternative MFA link not found, continuing...")

            # Step 5: Select "Use a verification code" option
            self._logger.info("Selecting 'Use a verification code' option...")
            time.sleep(2)

            # Find the element with data-value="PhoneAppOTP"
            try:
                verification_code_option = wait.until(
                    EC.presence_of_element_located((By.XPATH, "//div[@data-value='PhoneAppOTP']"))
                )
                time.sleep(1)
                verification_code_option.click()
            except TimeoutException:
                # Check if we're already on the OTP input page
                self._logger.info("Verification code option not found, checking if already on OTP input page...")

            # Step 6: Generate TOTP and enter it
            self._logger.info("Generating TOTP code...")
            totp = pyotp.TOTP(creds.mfa_secret)
            code = totp.now()
            self._logger.info(f"Generated TOTP code: {code}")

            # Wait for OTP input field
            self._logger.info("Entering verification code...")
            otp_input = wait.until(EC.presence_of_element_located((By.ID, "idTxtBx_SAOTCC_OTC")))
            otp_input.clear()
            otp_input.send_keys(code)

            # Step 7: Click Verify button
            self._logger.info("Clicking Verify button...")
            verify_button = driver.find_element(By.ID, "idSubmit_SAOTCC_Continue")
            verify_button.click()

            # Step 8: Wait for successful login (BOSS dashboard)
            self._logger.info("Waiting for BOSS dashboard to load...")
            wait.until(EC.presence_of_element_located((By.ID, "Label_UserName")))

            username = driver.find_element(By.ID, "Label_UserName").text
            self._logger.info(f"Automated login successful! Logged in as {username}")

            time.sleep(2)
            return username

        except TimeoutException as e:
            error_msg = f"Login timed out. Element not found: {str(e)}"
            self._logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            import traceback
            traceback.print_exc()
            error_msg = f"Automated login failed: {str(e)}"
            self._logger.error(error_msg)
            raise