"""
Shared utility for parsing boss_aliases values from various formats.
Used by DatabaseHelper and ProfessorProcessor.
"""
import json
import pandas as pd


def parse_boss_aliases(boss_aliases_val: any) -> list[str]:
    """
    Robustly parses the boss_aliases value from various formats into a clean list of strings.

    This function correctly handles:
    - None, pd.isna(), or other "empty" values.
    - A standard Python list.
    - A NumPy array.
    - A raw PostgreSQL array string (e.g., '{"item1","item2"}').
    - A JSON-formatted string array (e.g., '["item1", "item2"]').
    - Encoding issues with special characters like non-standard apostrophes.

    Returns:
        A clean Python list of strings. Returns an empty list for any invalid or empty input.
    """
    # Return an empty list for any "empty" or None-like value.
    if boss_aliases_val is None:
        return []

    # Handle arrays/lists before using pd.isna
    if hasattr(boss_aliases_val, '__len__') and not isinstance(boss_aliases_val, str):
        # It's already an array/list, so check if it's empty
        if len(boss_aliases_val) == 0:
            return []
        # If it's a non-empty array, process it
        if isinstance(boss_aliases_val, list):
            return [_clean_alias(str(item).strip()) for item in boss_aliases_val if item and str(item).strip()]
        elif hasattr(boss_aliases_val, 'tolist'):
            # NumPy array
            return [_clean_alias(str(item).strip()) for item in boss_aliases_val.tolist() if item and str(item).strip()]
        else:
            # Other iterable
            return [_clean_alias(str(item).strip()) for item in boss_aliases_val if item and str(item).strip()]

    # Now safe to use pd.isna for non-array values
    try:
        if pd.isna(boss_aliases_val):
            return []
    except:
        # If pd.isna fails for any reason, continue processing
        pass

    # Handle standard Python list.
    if isinstance(boss_aliases_val, list):
        return [_clean_alias(str(item).strip()) for item in boss_aliases_val if item and str(item).strip()]

    # Handle NumPy array by checking for the .tolist() method.
    if hasattr(boss_aliases_val, 'tolist'):
        return [_clean_alias(str(item).strip()) for item in boss_aliases_val.tolist() if item and str(item).strip()]

    # Handle various string formats.
    if isinstance(boss_aliases_val, str):
        aliases_str = boss_aliases_val.strip()

        if not aliases_str:
            return []

        # Case 1: PostgreSQL array format '{"item1","item2"}'
        if aliases_str.startswith('{') and aliases_str.endswith('}'):
            content = aliases_str[1:-1]
            # Split by comma, then strip whitespace and quotes from each item.
            return [_clean_alias(item.strip().strip('"')) for item in content.split(',') if item.strip()]

        # Case 2: JSON array format '["item1", "item2"]'
        if aliases_str.startswith('[') and aliases_str.endswith(']'):
            try:
                parsed_list = json.loads(aliases_str)
                if isinstance(parsed_list, list):
                    return [_clean_alias(str(item).strip()) for item in parsed_list if item and str(item).strip()]
            except (json.JSONDecodeError, TypeError):
                # If JSON is malformed, fall back to treating it as a plain string.
                pass

        # Case 3: A single alias provided as a plain string.
        return [_clean_alias(aliases_str)]

    # Fallback for other iterable types like tuples or sets.
    if hasattr(boss_aliases_val, '__iter__'):
        return [_clean_alias(str(item).strip()) for item in boss_aliases_val if item and str(item).strip()]

    return []


def _clean_alias(alias: str) -> str:
    """
    Clean an alias string by normalizing special characters and fixing encoding issues.
    """
    if not alias:
        return ""

    # Decode Unicode escape sequences like \\U2019
    import codecs
    try:
        # Handle both \\uXXXX and \\UXXXXXXXX formats
        alias = codecs.decode(alias, 'unicode_escape')
    except Exception:
        pass

    # Normalize encoding issues by replacing problematic characters
    alias = (
        alias.replace("’", "'")  # Replace smart apostrophe with standard apostrophe
        .replace("‘", "'")  # Replace left single quote with standard apostrophe
        .replace("”", '"')  # Replace smart double quote with standard double quote
        .replace("“", '"')  # Replace left double quote with standard double quote
        .replace("â€™", "'")  # Fix common UTF-8 encoding error
        .replace("ï¿½", "'")  # Fix common encoding issue
        .replace("???", "'")  # Fix for the specific RACHEL TAN XI'EN case
    )

    # Remove any other problematic characters
    import re
    # Replace any non-printable characters or invalid Unicode with nothing
    alias = re.sub(r'[^\x00-\x7F]', '', alias)

    return alias.strip()
