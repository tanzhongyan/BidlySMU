"""
Unit tests for Authenticator classes.
"""
import pytest
import os
from unittest.mock import MagicMock, patch
from selenium.common.exceptions import TimeoutException

from src.driver.authenticator import (
    AuthCredentials,
    Authenticator,
    ManualLogin,
    AutomatedLogin,
)


class TestAuthCredentials:
    """Tests for AuthCredentials."""

    def test_from_environment_returns_credentials(self):
        """from_environment should return credentials when env vars are set."""
        with patch.dict(
            os.environ,
            {
                "BOSS_EMAIL": "test@business.smu.edu.sg",
                "BOSS_PASSWORD": "test_password",
                "BOSS_MFA_SECRET": "SECRET123",
            },
        ):
            creds = AuthCredentials.from_environment()
            assert creds.email == "test@business.smu.edu.sg"
            assert creds.password == "test_password"
            assert creds.mfa_secret == "SECRET123"

    def test_from_environment_raises_when_email_missing(self):
        """from_environment should raise ValueError when BOSS_EMAIL is missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Missing required environment variables"):
                AuthCredentials.from_environment()

    def test_from_environment_raises_when_password_missing(self):
        """from_environment should raise ValueError when BOSS_PASSWORD is missing."""
        env = {
            "BOSS_EMAIL": "test@business.smu.edu.sg",
            "BOSS_MFA_SECRET": "SECRET123",
        }
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="Missing required environment variables"):
                AuthCredentials.from_environment()

    def test_from_environment_raises_when_mfa_secret_missing(self):
        """from_environment should raise ValueError when BOSS_MFA_SECRET is missing."""
        env = {
            "BOSS_EMAIL": "test@business.smu.edu.sg",
            "BOSS_PASSWORD": "test_password",
        }
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="Missing required environment variables"):
                AuthCredentials.from_environment()


class TestManualLogin:
    """Tests for ManualLogin."""

    def test_login_returns_username_on_success(self, mock_webdriver, mock_logger):
        """login should return username when dashboard elements are found."""
        username_element = MagicMock()
        username_element.text = "testuser@business.smu.edu.sg"
        # WebDriverWait calls find_element multiple times until condition is met
        # Return the same element for all calls
        mock_webdriver.find_element.return_value = username_element

        login = ManualLogin(timeout=5, logger=mock_logger)
        result = login.login(mock_webdriver)

        assert result == "testuser@business.smu.edu.sg"

    def test_login_raises_on_timeout(self, mock_webdriver, mock_logger):
        """login should raise Exception when dashboard elements not found."""
        mock_webdriver.find_element.side_effect = TimeoutException("Element not found")

        login = ManualLogin(timeout=1, logger=mock_logger)
        with pytest.raises(Exception, match="Login failed or timed out"):
            login.login(mock_webdriver)


class TestAutomatedLogin:
    """Tests for AutomatedLogin."""

    def test_login_navigates_and_enters_credentials(
        self, mock_webdriver, auth_credentials, mock_logger
    ):
        """login should navigate to BOSS and enter credentials."""
        mock_element = MagicMock()
        mock_webdriver.find_element.return_value = mock_element

        # Patch TOTP to avoid base32 validation issues in test
        with patch("src.driver.authenticator.pyotp.TOTP") as mock_totp:
            mock_totp.return_value.now.return_value = "123456"

            login = AutomatedLogin(
                credentials=auth_credentials, timeout=5, logger=mock_logger
            )
            result = login.login(mock_webdriver)

        # Should have called get to navigate to BOSS
        mock_webdriver.get.assert_called_once_with(
            "https://boss.intranet.smu.edu.sg/"
        )
        # Should have found email input
        assert mock_webdriver.find_element.call_count >= 1

    def test_credentials_are_stored(self, auth_credentials, mock_logger):
        """Credentials should be stored on the instance."""
        login = AutomatedLogin(
            credentials=auth_credentials, timeout=5, logger=mock_logger
        )
        assert login._credentials is auth_credentials
        assert login._timeout == 5


class TestAuthenticatorInterface:
    """Tests for Authenticator abstract interface."""

    def test_authenticator_is_abc(self):
        """Authenticator should be an abstract base class."""
        assert hasattr(Authenticator, "__abstractmethods__")

    def test_login_is_abstract_method(self):
        """login should be marked as abstract method."""
        assert "login" in Authenticator.__abstractmethods__
