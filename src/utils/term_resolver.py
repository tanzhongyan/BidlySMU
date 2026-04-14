"""
Term resolution utilities for BOSS academic terms.

Provides term code mapping and academic year range generation.
"""
from typing import List

# Term code mapping to BOSS term codes
TERM_CODE_MAP = {'T1': '10', 'T2': '20', 'T3A': '31', 'T3B': '32'}

# All available terms
ALL_TERMS = ['T1', 'T2', 'T3A', 'T3B']


def get_term_code_map() -> dict:
    """
    Get the mapping of term codes to BOSS term codes.

    Returns:
        dict: Mapping of term abbreviations to BOSS term codes.
              e.g., {'T1': '10', 'T2': '20', 'T3A': '31', 'T3B': '32'}
    """
    return TERM_CODE_MAP.copy()


def get_all_terms() -> List[str]:
    """
    Get the list of all term abbreviations.

    Returns:
        list: List of term abbreviations ['T1', 'T2', 'T3A', 'T3B'].
    """
    return ALL_TERMS.copy()


def transform_term_format(short_term: str) -> str:
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
    term_map = {
        'T1': 'Term 1',
        'T2': 'Term 2',
        'T3A': 'Term 3A',
        'T3B': 'Term 3B'
    }

    try:
        year_part, term_part = short_term.split('_')
        full_term_name = term_map.get(term_part)

        if full_term_name:
            return f"{year_part} {full_term_name}"
        else:
            raise ValueError(f"Unknown term suffix: '{term_part}'")

    except (ValueError, IndexError) as e:
        raise ValueError(f"Invalid term format: '{short_term}'. Expected format like '2025-26_T1'.")


def generate_academic_year_range(start_ay_term: str, end_ay_term: str) -> List[str]:
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
