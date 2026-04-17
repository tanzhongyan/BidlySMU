"""
Professor name normalization utilities.
Handles Asian/Western name parsing, fuzzy matching, and LLM fallback.
"""
import re
import json
import time
from typing import Tuple, List, Dict
from collections import defaultdict

from src.utils.name_data import (
    ALL_ASIAN_SURNAMES, WESTERN_GIVEN_NAMES, PATRONYMIC_KEYWORDS, SURNAME_PARTICLES
)


class ProfessorNormalizer:
    """
    Handles professor name normalization using rule-based fallback and LLM.
    """

    def __init__(
        self,
        logger=None,
        professor_lookup: Dict = None,
        professors_cache: Dict = None,
        llm_client=None,
        llm_model_name: str = "gemini-2.5-flash",
        llm_batch_size: int = 50,
        llm_prompt: str = None
    ):
        self._logger = logger
        self.professor_lookup = professor_lookup or {}
        self.professors_cache = professors_cache or {}
        self.llm_client = llm_client
        self.llm_model_name = llm_model_name
        self.llm_batch_size = llm_batch_size
        self.llm_prompt = llm_prompt or self._default_prompt()

    @staticmethod
    def _default_prompt() -> str:
        return """
        You are an expert in academic name structures from around the world.
        You will be given a JSON list of professor names.
        Your task is to identify the primary surname for each name.
        You MUST return a single JSON array of strings, where each string is the identified surname.
        The order of surnames in your response must exactly match the order of the full names in the input list.
        Provide ONLY the JSON array in your response.
        """

    def set_professor_lookup(self, lookup: Dict) -> None:
        self.professor_lookup = lookup

    def set_professors_cache(self, cache: Dict) -> None:
        self.professors_cache = cache

    def set_llm_client(self, client) -> None:
        self.llm_client = client

    def normalize(self, full_name: str) -> Tuple[str, str]:
        """Main entry point. Returns (boss_name, afterclass_name)."""
        return self._normalize_professor_name_fallback(full_name)

    def _normalize_professor_name_fallback(self, name: str) -> Tuple[str, str]:
        """
        (Fallback Method) Normalizes professor names using a definitive, rule-based system.
        """
        # BEGIN EXCERPT from table_builder.py:510-591
        if name is None or not str(name).strip():
            return "UNKNOWN", "Unknown"

        # --- Step 1: Aggressive Preprocessing ---
        name_str = str(name).strip().replace("'", "'")
        name_str = re.sub(r'\s*\(.*\)\s*', ' ', name_str).strip()
        # Remove all middle initials (e.g., "S.", "H.", "H H", "S") to standardize names
        words = name_str.split()
        words_no_initials = [w for w in words if not (len(w) == 1 and w.isalpha()) and not (len(w) == 2 and w.endswith('.'))]
        name_str = ' '.join(words_no_initials)

        boss_name = name_str.upper()

        # --- Step 2: Handle High-Certainty Delimiters ---
        if ',' in name_str:
            parts = [p.strip() for p in name_str.split(',')]
            words = ' '.join(parts).split()
            surname_to_check = words[0].upper()
            if len(parts) == 2:
                words_after_comma = parts[1].split()
                if words_after_comma and words_after_comma[0].upper() in ALL_ASIAN_SURNAMES:
                    surname_to_check = words_after_comma[0].upper()
                else:
                    words_before_comma = parts[0].split()
                    if words_before_comma and words_before_comma[0].upper() in ALL_ASIAN_SURNAMES:
                        surname_to_check = words_before_comma[0].upper()
            afterclass_parts = [word.capitalize() for word in words]
            for i, word in enumerate(words):
                if word.upper() == surname_to_check:
                    afterclass_parts[i] = word.upper()
            return boss_name, ' '.join(afterclass_parts)

        words = name_str.split()
        if not words:
            return boss_name, "Unknown"
        if len(words) == 1:
            return boss_name, words[0].capitalize()

        for i, word in enumerate(words):
            if word.upper() in PATRONYMIC_KEYWORDS and i < len(words) - 1:
                surname_index = i + 1
                afterclass_parts = [w.capitalize() for w in words]
                afterclass_parts[i] = word.lower()
                afterclass_parts[surname_index] = words[surname_index].upper()
                return boss_name, ' '.join(afterclass_parts)

        # --- Step 3: Definitive Rule-Based Surname Identification ---
        surname_index = -1

        # Rule 1 (Fixes "Middle Surname"): If the name starts with a Western/Indian given name,
        # actively search for the first known Asian/Indian surname that follows.
        if words[0].upper() in WESTERN_GIVEN_NAMES:
            for i in range(1, len(words)):
                if words[i].upper() in ALL_ASIAN_SURNAMES:
                    surname_index = i
                    break

        # Rule 2 (Fixes "Surname-First Western"): If a name contains a Western given name but
        # does NOT start with it, and the first word is not an Asian surname, assume the first word is the surname.
        elif any(w.upper() in WESTERN_GIVEN_NAMES for w in words) and words[0].upper() not in ALL_ASIAN_SURNAMES:
            surname_index = 0

        # Rule 3: If neither of the complex cases above apply, check if the name starts with a known Asian surname.
        elif words[0].upper() in ALL_ASIAN_SURNAMES:
            surname_index = 0

        # Rule 4 (Fallback): If no specific pattern has been matched, default to the last word.
        if surname_index == -1:
            surname_index = len(words) - 1

        # Post-processing for European particles
        afterclass_parts = [word.capitalize() for word in words]
        if surname_index > 0 and words[surname_index-1].upper() in SURNAME_PARTICLES:
            afterclass_parts[surname_index-1] = words[surname_index-1].upper()

        afterclass_parts[surname_index] = words[surname_index].upper()

        return boss_name, ' '.join(afterclass_parts)
        # END EXCERPT

    def _calculate_fuzzy_score(self, new_name: str, known_alias: str) -> float:
        """
        Calculates a fuzzy match score using a hybrid strategy.
        """
        # BEGIN EXCERPT from table_builder.py:837-871
        if not new_name or not known_alias:
            return 0.0

        # Clean and normalize names for consistent comparison
        new_name_clean = ' '.join(str(new_name).upper().replace(',', ' ').split())
        known_alias_clean = ' '.join(str(known_alias).upper().replace(',', ' ').split())

        # --- Layer 1: High-Precision Substring Check ---
        if new_name_clean in known_alias_clean or known_alias_clean in new_name_clean:
            return 95

        # --- Layer 2: Hybrid Fuzzy Logic ---
        from thefuzz import fuzz
        partial_score = fuzz.partial_ratio(new_name_clean, known_alias_clean)
        token_set_score = fuzz.token_set_ratio(new_name_clean, known_alias_clean)

        return max(partial_score, token_set_score)
        # END EXCERPT

    def _split_professor_names(self, prof_name: str) -> List[str]:
        """
        Intelligently splits a string of professor names using a greedy, longest-match-first approach.
        """
        # BEGIN EXCERPT from table_builder.py:1466-1519
        if prof_name is None or not str(prof_name).strip():
            return []

        prof_name_str = str(prof_name).strip()

        # First, check if the entire string is already a known professor.
        if prof_name_str.upper() in self.professor_lookup:
            return [prof_name_str]

        # If there are no commas, it can only be one professor.
        if ',' not in prof_name_str:
            return [prof_name_str]

        parts = [p.strip() for p in prof_name_str.split(',') if p.strip()]

        found_professors = []
        i = 0
        while i < len(parts):
            match_found = False
            for j in range(len(parts), i, -1):
                candidate = ', '.join(parts[i:j])

                if candidate.upper() in self.professor_lookup:
                    found_professors.append(candidate)
                    i = j
                    match_found = True
                    break

            if not match_found:
                unknown_part = parts[i]
                if found_professors and len(unknown_part.split()) == 1:
                    found_professors[-1] = f"{found_professors[-1]}, {unknown_part}"
                    if self._logger:
                        self._logger.info(f"Combined unknown single word '{unknown_part}' with previous professor")
                else:
                    found_professors.append(unknown_part)
                i += 1

        return found_professors
        # END EXCERPT

    def _names_match_fuzzy_exact(self, name1: str, name2: str) -> bool:
        """Exact fuzzy matching for names - only matches if completely identical after normalization"""
        # BEGIN EXCERPT from table_builder.py:1633-1649
        if name1 is None or name2 is None:
            return False

        name1 = str(name1) if name1 is not None else ""
        name2 = str(name2) if name2 is not None else ""

        clean1 = ' '.join(name1.replace(',', ' ').replace('.', ' ').split()).upper()
        clean2 = ' '.join(name2.replace(',', ' ').replace('.', ' ').split()).upper()

        return clean1 == clean2
        # END EXCERPT

    def extract_unique_professors(self, multiple_data) -> Tuple[set, dict]:
        """Extracts unique professor names and their variations from the raw data."""
        unique_professors = set()
        professor_variations = defaultdict(set)

        import pandas as pd
        for _, row in multiple_data.iterrows():
            prof_name_raw = row.get('professor_name')
            if prof_name_raw is None or pd.isna(prof_name_raw):
                continue

            prof_name = str(prof_name_raw).strip()
            if not prof_name or prof_name.lower() in ['nan', 'tba', 'to be announced']:
                continue

            split_professors = self._split_professor_names(prof_name)
            for individual in split_professors:
                clean_prof = individual.strip()
                if clean_prof:
                    unique_professors.add(clean_prof)
                    if ', ' in clean_prof:
                        parts = clean_prof.split(', ')
                        if len(parts) == 2:
                            base_name = parts[0].strip()
                            extension = parts[1].strip()
                            if len(extension.split()) == 1:
                                professor_variations[clean_prof].add(base_name)
                                professor_variations[clean_prof].add(clean_prof)
                                if base_name in professor_variations:
                                    professor_variations[base_name].add(clean_prof)
                    else:
                        professor_variations[clean_prof].add(clean_prof)

        return unique_professors, professor_variations

    def normalize_professors_batch(self, names_to_process: list) -> dict:
        """
        Normalizes a list of professor names using the pre-configured LLM client,
        with a rule-based fallback.
        """
        # BEGIN EXCERPT from table_builder.py:4493-4566
        normalized_map = {}
        if not names_to_process:
            return normalized_map

        # --- LLM Pathway ---
        try:
            if not self.llm_client:
                raise ValueError("LLM client not configured. Check API key.")

            import genai
            total_batches = (len(names_to_process) + self.llm_batch_size - 1) // self.llm_batch_size
            if self._logger:
                self._logger.info(f"Normalizing {len(names_to_process)} names in {total_batches} batches using '{self.llm_model_name}'...")

            for i in range(0, len(names_to_process), self.llm_batch_size):
                batch_names = names_to_process[i:i + self.llm_batch_size]
                if self._logger:
                    self._logger.info(f"  -> Processing batch {i // self.llm_batch_size + 1} of {total_batches} ({len(batch_names)} names)...")

                response = self.llm_client.models.generate_content(
                    model=self.llm_model_name,
                    contents=f"{self.llm_prompt}\n\n{json.dumps(batch_names)}",
                    config=genai.GenerateContentConfig(
                        response_mime_type="application/json"
                    )
                )

                response_text = response.text
                match = re.search(r'\[.*\]', response_text, re.DOTALL)
                if not match:
                    raise ValueError("LLM response did not contain a valid JSON array.")

                json_text = match.group(0)
                surnames = json.loads(json_text)

                if not isinstance(surnames, list) or len(surnames) != len(batch_names):
                    raise ValueError(f"LLM returned malformed data for batch {i // self.llm_batch_size + 1}.")

                for original_name, surname in zip(batch_names, surnames):
                    name_str = str(original_name).strip().replace("'", "'")
                    name_str = re.sub(r'\s*\(.*\)\s*', ' ', name_str).strip()
                    words = name_str.split()
                    words_no_initials = [w for w in words if not (len(w) == 1 and w.isalpha()) and not (len(w) == 2 and w.endswith('.'))]
                    boss_name = ' '.join(words_no_initials).upper()

                    name_parts = re.split(r'([ ,])', original_name)
                    afterclass_parts = []
                    surname_found = False
                    for part in name_parts:
                        if not surname_found and part.strip(" ,").upper() == surname.upper():
                            afterclass_parts.append(part.upper())
                            surname_found = True
                        else:
                            afterclass_parts.append(part.capitalize())
                    afterclass_name = "".join(afterclass_parts)
                    normalized_map[original_name] = (boss_name, afterclass_name)

                time.sleep(6)

            if self._logger:
                self._logger.info("Batch normalization completed using Gemini LLM.")

        # --- Fallback Pathway ---
        except Exception as e:
            if self._logger:
                self._logger.warning(f"LLM normalization failed ({e}). Falling back to rule-based method.")
            normalized_map.clear()
            for name in names_to_process:
                normalized_map[name] = self._normalize_professor_name_fallback(name)

        return normalized_map
        # END EXCERPT