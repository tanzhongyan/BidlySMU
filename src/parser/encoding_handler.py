"""
Encoding handler for cleaning text from HTML pages.
"""
import re


class EncodingHandler:
    """
    Handles text encoding issues from HTML content.

    Fixes common encoding problems like â€" -> — etc.
    """

    # Common encoding fixes - ORDER MATTERS! Process longer patterns first
    ENCODING_FIXES = [
        ('â€"', '—'),   # em-dash
        ('â€™', "'"),   # right single quotation mark
        ('â€œ', '"'),   # left double quotation mark
        ('â€¦', '…'),   # horizontal ellipsis
        ('â€¢', '•'),   # bullet
        ('â€‹', ''),    # zero-width space
        ('â€‚', ' '),   # en space
        ('â€ƒ', ' '),   # em space
        ('â€‰', ' '),   # thin space
        ('â€', '"'),    # right double quotation mark
        ('Â', ''),      # non-breaking space artifacts
    ]

    @classmethod
    def clean(cls, text: str) -> str:
        """
        Clean text to fix encoding issues.

        Args:
            text: Raw text from HTML

        Returns:
            Cleaned text with encoding issues fixed
        """
        if not text:
            return text

        cleaned_text = text
        for bad, good in cls.ENCODING_FIXES:
            cleaned_text = cleaned_text.replace(bad, good)

        # Remove any remaining problematic characters
        cleaned_text = re.sub(r'â€[^\w]', '', cleaned_text)

        return cleaned_text.strip()
