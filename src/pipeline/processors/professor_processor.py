"""
ProfessorProcessor - Refactored processor using DTO pattern.
Processes professor records from raw data with comprehensive resolution chain.
"""
import codecs
import os
import re
import json
import uuid
import time
import logging
import pandas as pd
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set

from src.pipeline.processors.abstract_processor import AbstractProcessor
from src.pipeline.dtos.professor_dto import ProfessorDTO
from src.pipeline.processors.professor_resolution_service import ProfessorResolutionService


# ============================================================================
# SECTION 1: Constants (lifted from src/utils/name_data.py)
# ============================================================================

ASIAN_SURNAMES: Dict[str, list] = {
    'chinese': [
        'WANG', 'LI', 'ZHANG', 'LIU', 'CHEN', 'YANG', 'HUANG', 'ZHAO', 'WU', 'ZHOU', 'XU', 'SUN', 'MA', 'ZHU', 'HU', 'GUO', 'HE', 'LIN', 'GAO', 'LUO',
        'CHENG', 'LIANG', 'XIE', 'SONG', 'TANG', 'HAN', 'FENG', 'DENG', 'CAO', 'PENG', 'YUAN', 'SU', 'JIANG', 'JIA', 'LU', 'WEI', 'XIAO', 'YU', 'QIAN',
        'PAN', 'YAO', 'TAN', 'DU', 'YE', 'TIAN', 'SHI', 'BAI', 'QIN', 'XUE', 'YAN', 'DAI', 'MO', 'CHANG', 'WAN', 'GU', 'ZENG', 'LUO', 'FAN', 'JIN',
        'ONG', 'LIM', 'LEE', 'TEO', 'NG', 'GOH', 'CHUA', 'CHAN', 'KOH', 'ANG', 'YEO', 'SIM', 'CHIA', 'CHONG', 'LAM', 'CHEW', 'TOH', 'LOW', 'SEAH',
        'PEK', 'KWEK', 'QUEK', 'LOH', 'AW', 'CHYE', 'LOK'
    ],
    'korean': [
        'KIM', 'LEE', 'PARK', 'CHOI', 'JEONG', 'KANG', 'CHO', 'YOON', 'JANG', 'LIM', 'HAN', 'OH', 'SEO', 'KWON', 'HWANG', 'SONG', 'JUNG', 'HONG',
        'AHN', 'GO', 'MOON', 'SON', 'BAE', 'BAEK', 'HEO', 'NAM'
    ],
    'vietnamese': [
        'NGUYEN', 'TRAN', 'LE', 'PHAM', 'HOANG', 'PHAN', 'VU', 'VO', 'DANG', 'BUI', 'DO', 'HO', 'NGO', 'DUONG', 'LY'
    ],
    'indian': [
        'SHARMA', 'SINGH', 'KUMAR', 'GUPTA', 'PATEL', 'KHAN', 'REDDY', 'YADAV', 'DAS', 'JAIN', 'RAO', 'MEHTA', 'CHOPRA', 'KAPOOR', 'MALHOTRA',
        'AGGARWAL', 'JOSHI', 'MISHRA', 'TRIPATHI', 'PANDEY', 'NAIR', 'MENON', 'PILLAI', 'IYER', 'MUKHERJEE', 'BANERJEE', 'CHATTERJEE'
    ],
    'japanese': [
        'SATO', 'SUZUKI', 'TAKAHASHI', 'TANAKA', 'WATANABE', 'ITO', 'YAMAMOTO', 'NAKAMURA', 'KOBAYASHI', 'SAITO', 'KATO', 'YOSHIDA', 'YAMADA'
    ]
}

ALL_ASIAN_SURNAMES: Set[str] = set().union(*ASIAN_SURNAMES.values())

WESTERN_GIVEN_NAMES: Set[str] = {
    'AARON', 'ADAM', 'ADRIAN', 'ALAN', 'ALBERT', 'ALEX', 'ALEXANDER', 'ALFRED', 'ALVIN', 'AMANDA', 'AMY', 'ANDREA', 'ANDREW', 'ANGELA', 'ANNA', 'ANTHONY', 'ARTHUR', 'AUDREY',
    'BEN', 'BENJAMIN', 'BERNARD', 'BETTY', 'BILLY', 'BOB', 'BOWEN', 'BRANDON', 'BRENDA', 'BRIAN', 'BRYAN', 'BRUCE',
    'CARL', 'CAROL', 'CATHERINE', 'CHARLES', 'CHRIS', 'CHRISTIAN', 'CHRISTINA', 'CHRISTINE', 'CHRISTOPHER', 'COLIN', 'CRAIG', 'CRYS',
    'DANIEL', 'DANNY', 'DARREN', 'DAVID', 'DEBORAH', 'DENISE', 'DENNIS', 'DEREK', 'DIANA', 'DONALD', 'DOUGLAS',
    'EDWARD', 'EDWIN', 'ELAINE', 'ELIZABETH', 'EMILY', 'ERIC', 'EUGENE', 'EVELYN',
    'FELIX', 'FRANCIS', 'FRANK',
    'GABRIEL', 'GARY', 'GEOFFREY', 'GEORGE', 'GERALD', 'GLORIA', 'GORDON', 'GRACE', 'GRAHAM', 'GREGORY',
    'HANNAH', 'HARRY', 'HELEN', 'HENRY', 'HOWARD',
    'IAN', 'IVAN',
    'JACK', 'JACOB', 'JAMES', 'JANE', 'JANET', 'JASON', 'JEAN', 'JEFFREY', 'JENNIFER', 'JEREMY', 'JERRY', 'JESSICA', 'JIM', 'JOAN', 'JOE', 'JOHN', 'JONATHAN', 'JOSEPH', 'JOSHUA', 'JOYCE', 'JUDY', 'JULIA', 'JULIE', 'JUSTIN',
    'KAREN', 'KATHERINE', 'KATHY', 'KEITH', 'KELLY', 'KELVIN', 'KENNETH', 'KEVIN', 'KIMBERLY',
    'LARRY', 'LAURA', 'LAWRENCE', 'LEO', 'LEONARD', 'LINDA', 'LISA',
    'MARGARET', 'MARIA', 'MARK', 'MARTIN', 'MARY', 'MATTHEW', 'MEGAN', 'MELISSA', 'MICHAEL', 'MICHELLE', 'MIKE',
    'NANCY', 'NATHAN', 'NEHA', 'NICHOLAS', 'NICOLE',
    'OLIVER', 'OLIVIA',
    'PAMELA', 'PATRICIA', 'PATRICK', 'PAUL', 'PETER', 'PHILIP',
    'RACHEL', 'RAYMOND', 'REBECCA', 'RICHARD', 'ROBERT', 'ROGER', 'RONALD', 'ROY', 'RUSSELL', 'RYAN',
    'SAM', 'SAMUEL', 'SANDRA', 'SARAH', 'SCOTT', 'SEAN', 'SHARON', 'SOPHIA', 'STANLEY', 'STEPHANIE', 'STEPHEN', 'STEVEN', 'SUSAN',
    'TERENCE', 'TERRY', 'THERESA', 'THOMAS', 'TIMOTHY', 'TONY',
    'VALERIE', 'VICTOR', 'VINCENT', 'VIRGINIA',
    'WALTER', 'WAYNE', 'WENDY', 'WILLIAM', 'WILLIE'
}

PATRONYMIC_KEYWORDS: Set[str] = {'BIN', 'BINTE', 'S/O', 'D/O'}

SURNAME_PARTICLES: Set[str] = {'DE', 'DI', 'DA', 'VAN', 'VON', 'LA', 'LE', 'DEL', 'DELLA'}


# ============================================================================
# SECTION 3: ProfessorProcessor Class
# ============================================================================

class ProfessorProcessor(AbstractProcessor):
    """
    Processes professor records from raw data.

    Returns Tuple[List[ProfessorDTO], List[ProfessorDTO]] representing
    (new_professors, updated_professors).

    Architecture:
    - Runs BEFORE class processing (so professors are fully resolved)
    - Uses 8-strategy resolution chain to match existing professors
    - Only creates new professors when ALL strategies fail
    - Updates boss_aliases when new variations are found for existing professors
    """

    # LLM constants for surname refinement (new professors only)
    _LLM_MODEL_NAME = "gemini-2.5-flash"
    _LLM_BATCH_SIZE = 50
    _LLM_RATE_LIMIT_SECONDS = 6
    _LLM_PROMPT = (
        "You are an expert in academic name structures from around the world. "
        "You will be given a JSON list of professor names. Your task is to identify "
        "the primary surname for each name. You MUST return a single JSON array of "
        "strings, where each string is the identified surname. The surname should be "
        "in the same language/script as the input name. For Asian names (Chinese, Korean, "
        "Japanese, Vietnamese), the surname is typically the first word. For Western names, "
        "the surname is typically the last word. For South/Southeast Asian names with "
        "patronymics (Singh, Kumar, Bin, Binte), identify the actual family surname."
    )

    # ------------------------------------------------------------------------
    # Helper Functions (moved inside class - only used by ProfessorProcessor)
    # ------------------------------------------------------------------------

    @staticmethod
    def _clean_alias(alias: str) -> str:
        """Clean an alias string by normalizing special characters and fixing encoding issues."""
        if not alias:
            return ""

        try:
            alias = codecs.decode(alias, 'unicode_escape')
        except Exception:
            pass

        alias = (
            alias.replace("'", "'")
            .replace("'", "'")
            .replace('"', '"')
            .replace('"', '"')
            .replace("â€™", "'")
            .replace("ï¿½", "'")
            .replace("???", "'")
        )

        alias = re.sub(r'[^\x00-\x7F]', '', alias)

        return alias.strip()

    @staticmethod
    def parse_boss_aliases(boss_aliases_val: any) -> List[str]:
        """
        Robustly parses the boss_aliases value from various formats into a clean list of strings.
        Handles: None, list, numpy array, PostgreSQL array string, JSON string, etc.
        """
        if boss_aliases_val is None:
            return []

        if hasattr(boss_aliases_val, '__len__') and not isinstance(boss_aliases_val, str):
            if len(boss_aliases_val) == 0:
                return []
            if isinstance(boss_aliases_val, list):
                return [ProfessorProcessor._clean_alias(str(item).strip()) for item in boss_aliases_val if item and str(item).strip()]
            elif hasattr(boss_aliases_val, 'tolist'):
                return [ProfessorProcessor._clean_alias(str(item).strip()) for item in boss_aliases_val.tolist() if item and str(item).strip()]
            else:
                return [ProfessorProcessor._clean_alias(str(item).strip()) for item in boss_aliases_val if item and str(item).strip()]

        try:
            if pd.isna(boss_aliases_val):
                return []
        except:
            pass

        if isinstance(boss_aliases_val, list):
            return [ProfessorProcessor._clean_alias(str(item).strip()) for item in boss_aliases_val if item and str(item).strip()]

        if hasattr(boss_aliases_val, 'tolist'):
            return [ProfessorProcessor._clean_alias(str(item).strip()) for item in boss_aliases_val.tolist() if item and str(item).strip()]

        if isinstance(boss_aliases_val, str):
            aliases_str = boss_aliases_val.strip()
            if not aliases_str:
                return []

            if aliases_str.startswith('{') and aliases_str.endswith('}'):
                content = aliases_str[1:-1]
                # Quote-aware split: PostgreSQL arrays use {"A, B", "C"} format
                # where commas inside quotes are part of the value, not delimiters
                items = []
                current = []
                in_quotes = False
                for char in content:
                    if char == '"':
                        in_quotes = not in_quotes
                    elif char == ',' and not in_quotes:
                        items.append(''.join(current))
                        current = []
                    else:
                        current.append(char)
                if current:
                    items.append(''.join(current))
                return [ProfessorProcessor._clean_alias(item.strip().strip('"')) for item in items if item.strip()]

            if aliases_str.startswith('[') and aliases_str.endswith(']'):
                try:
                    parsed_list = json.loads(aliases_str)
                    if isinstance(parsed_list, list):
                        return [ProfessorProcessor._clean_alias(str(item).strip()) for item in parsed_list if item and str(item).strip()]
                except (json.JSONDecodeError, TypeError):
                    pass

            return [ProfessorProcessor._clean_alias(aliases_str)]

        if hasattr(boss_aliases_val, '__iter__'):
            return [ProfessorProcessor._clean_alias(str(item).strip()) for item in boss_aliases_val if item and str(item).strip()]

        return []

    def __init__(
        self,
        raw_data: pd.DataFrame,
        professors_cache: Dict[str, Dict],
        logger: Optional[logging.Logger] = None
    ):
        super().__init__(logger)
        self._raw_data = raw_data
        self._professors_cache = professors_cache
        self._professor_lookup: Dict[str, Dict] = {}  # Built from cache during processing
        self._new_professors_dtos: List[ProfessorDTO] = []  # Session deduplication
        self._results_updated: List[ProfessorDTO] = []  # Updated professor DTOs from process()
        # Build set of valid professor IDs from DB cache for validation
        self._valid_professor_ids = {str(p['id']) for p in professors_cache.values() if 'id' in p}
        # Resolution service built early (with DB cache) and updated with session professors at end of process()
        self.resolution_service = ProfessorResolutionService(
            professors_cache=professors_cache,
            new_professors=[],
            updated_professors=[],
            logger=logger,
            valid_professor_ids=self._valid_professor_ids
        )

    def process(self) -> Tuple[List[ProfessorDTO], List[ProfessorDTO]]:
        """Main entry point - returns (new_professors, updated_professors)."""
        self._logger.info("Processing professors...")

        # Step 1: Build internal lookup from professors_cache (DB) for lookups
        self._build_lookup_from_cache()

        # Step 2: Extract unique professor names from raw data
        unique_professors, professor_variations = self._extract_unique_professors()
        self._logger.info(f"Found {len(unique_professors)} unique professor names")

        # Step 3: Normalize all names using rule-based approach
        normalized_map = self._normalize_professors_batch(unique_professors)

        # Step 4: For each professor - run FULL resolution chain
        results_new, results_updated = self._resolve_all_professors(
            unique_professors, normalized_map, professor_variations
        )

        # Step 5: Refine new professors' afterclass_name with LLM surname extraction
        # Only runs on genuinely new professors (all 8 resolution strategies failed)
        # Requires GEMINI_API_KEY env var; falls back to rule-based if unavailable
        self._refine_new_professors_with_llm(results_new)

        # Step 6: Save updated lookup to CSV (for human reference only)
        self._results_updated = results_updated
        self._save_professor_lookup_csv()

        self._logger.info(f"Professor processing complete: {len(results_new)} new, {len(results_updated)} updated")

        # Update resolution service with session-created professors
        self.resolution_service.update_with_session_professors(
            new_professors=results_new,
            updated_professors=results_updated
        )

        # Also update with all discovered name variations from professor_lookup
        # This is critical for ClassProcessor to resolve professor names correctly
        self.resolution_service.update_with_professor_lookup(self._professor_lookup)

        return results_new, results_updated

    def _get_professor_id_from_lookup(self, prof_name: str) -> Optional[str]:
        """Get professor_id from lookup after a successful match.

        This method retrieves the database_id from the professor_lookup
        that was populated during the resolution chain.
        """
        normalized_name = prof_name.upper()

        # Check direct lookup
        if normalized_name in self._professor_lookup:
            return self._professor_lookup[normalized_name]['database_id']

        # Check boss_name lookup (normalized)
        boss_name, _ = self._normalize_professor_name_fallback(prof_name)
        if boss_name.upper() in self._professor_lookup:
            return self._professor_lookup[boss_name.upper()]['database_id']

        return None

    def _update_existing_professors_with_new_variations(
        self,
        new_variations: Dict[str, Set[str]]
    ) -> List[ProfessorDTO]:
        """Update existing professors in database with new boss_aliases variations.

        This method creates UPDATE DTOs for professors whose boss_aliases need to be
        extended with newly discovered raw name variations.

        Args:
            new_variations: Dict mapping professor_id -> set of new aliases to add

        Returns:
            List of ProfessorDTO representing professors to be updated
        """
        updated_dtos = []

        for prof_id, new_aliases in new_variations.items():
            # Find the professor in the cache
            prof_data = None
            for _, data in self._professors_cache.items():
                if str(data.get('id')) == prof_id:
                    prof_data = data
                    break

            if prof_data is None:
                self._logger.warning(f"Could not find professor {prof_id} in cache to update aliases")
                continue

            # Get current aliases
            current_aliases = set(self.parse_boss_aliases(prof_data.get('boss_aliases', [])))

            # Add new aliases (normalize to upper for comparison)
            current_aliases_upper = {a.upper() for a in current_aliases}
            added_count = 0
            for new_alias in new_aliases:
                if new_alias.upper() not in current_aliases_upper:
                    current_aliases.add(new_alias.upper())
                    added_count += 1

            if added_count == 0:
                # No new aliases to add
                continue

            # Create UPDATE DTO
            updated_aliases_list = sorted(list(current_aliases))

            # Build the row for the DTO
            row = {
                'id': prof_id,
                'name': prof_data.get('name', ''),
                'slug': prof_data.get('slug', ''),
                'email': prof_data.get('email', ''),
                'photo_url': prof_data.get('photo_url', ''),
                'profile_url': prof_data.get('profile_url', ''),
                'belong_to_university': prof_data.get('belong_to_university', 1),
                'boss_aliases': json.dumps(updated_aliases_list),
                'original_scraped_name': prof_data.get('original_scraped_name', ''),
                'updated_at': datetime.now().isoformat()
            }

            dto = ProfessorDTO.from_row(row)
            # Set updated_at to indicate this is an UPDATE, not CREATE
            dto.updated_at = datetime.now().isoformat()

            updated_dtos.append(dto)
            self._logger.info(f"Added {added_count} new alias(es) to professor {prof_data.get('name')} ({prof_id}): {new_aliases}")

        return updated_dtos

    # ------------------------------------------------------------------------
    # Normalization (LLM + fallback)
    # ------------------------------------------------------------------------

    def _normalize_professors_batch(self, names: List[str]) -> Dict[str, Tuple[str, str]]:
        """Normalize all names using rule-based approach.

        LLM surname refinement is handled separately in _refine_new_professors_with_llm()
        for new professors only (after the resolution chain identifies them).
        """
        if not names:
            return {}

        normalized_map = {}
        for name in names:
            normalized_map[name] = self._normalize_professor_name_fallback(name)
        self._logger.info("Used rule-based normalization for all names.")

        return normalized_map

    def _call_llm_batch(
        self,
        names: List[str],
        client,
        model_name: str,
        prompt: str,
        batch_size: int
    ) -> Dict[str, Tuple[str, str]]:
        """Call LLM to normalize a batch of names.

        Args:
            names: List of professor names to normalize
            client: Initialized genai.Client instance
            model_name: Gemini model name (e.g. "gemini-2.5-flash")
            prompt: LLM prompt for surname extraction
            batch_size: Number of names per API call
        """
        from google.genai import types
        normalized_map = {}
        total_batches = (len(names) + batch_size - 1) // batch_size

        self._logger.info(f"Normalizing {len(names)} names in {total_batches} batches using '{model_name}'...")

        for i in range(0, len(names), batch_size):
            batch_names = names[i:i + batch_size]
            batch_num = i // batch_size + 1
            self._logger.info(f"  -> Processing batch {batch_num} of {total_batches} ({len(batch_names)} names)...")

            response = client.models.generate_content(
                model=model_name,
                contents=f"{prompt}\n\n{json.dumps(batch_names)}",
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )

            response_text = response.text
            match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if not match:
                raise ValueError("LLM response did not contain a valid JSON array.")

            json_text = match.group(0)
            surnames = json.loads(json_text)

            if not isinstance(surnames, list) or len(surnames) != len(batch_names):
                raise ValueError(f"LLM returned malformed data for batch {batch_num}.")

            for original_name, surname in zip(batch_names, surnames):
                boss_name, afterclass_name = self._format_normalized_name(original_name, surname)
                normalized_map[original_name] = (boss_name, afterclass_name)

            time.sleep(self._LLM_RATE_LIMIT_SECONDS)  # Rate limiting

        return normalized_map

    def _refine_new_professors_with_llm(self, results_new: List[ProfessorDTO]) -> None:
        """Use LLM to refine afterclass_name surnames for newly created professors.

        Only called for professors that the 8-strategy resolution chain couldn't match
        (genuinely new). Sends all new professor names in a single batched LLM call.
        If GEMINI_API_KEY is not set or LLM fails, keeps rule-based afterclass_name.

        Args:
            results_new: List of newly created ProfessorDTOs to refine
        """
        if not results_new:
            return

        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            self._logger.info(
                f"GEMINI_API_KEY not set, keeping rule-based afterclass_name "
                f"for {len(results_new)} new professors"
            )
            return

        try:
            from google import genai
            client = genai.Client(api_key=api_key)
        except ImportError:
            self._logger.warning(
                "google-genai not installed, keeping rule-based afterclass_name "
                "for new professors. Install with: pip install google-genai"
            )
            return

        # Extract original scraped names for LLM
        names_to_normalize = [dto.original_scraped_name for dto in results_new]

        self._logger.info(
            f"Refining {len(results_new)} new professor surnames via Gemini LLM..."
        )

        try:
            llm_normalized = self._call_llm_batch(
                names=names_to_normalize,
                client=client,
                model_name=self._LLM_MODEL_NAME,
                prompt=self._LLM_PROMPT,
                batch_size=self._LLM_BATCH_SIZE
            )
        except Exception as e:
            self._logger.warning(
                f"LLM surname refinement failed ({e}). "
                f"Keeping rule-based afterclass_name for {len(results_new)} new professors."
            )
            return

        # Update DTOs with LLM-corrected afterclass_name
        refined_count = 0
        for dto in results_new:
            if dto.original_scraped_name in llm_normalized:
                _, llm_afterclass_name = llm_normalized[dto.original_scraped_name]
                if llm_afterclass_name != dto.name:
                    self._logger.info(
                        f"LLM refined afterclass_name: '{dto.name}' -> '{llm_afterclass_name}' "
                        f"(original: '{dto.original_scraped_name}')"
                    )
                    dto.name = llm_afterclass_name
                    dto.slug = re.sub(r'[^a-zA-Z0-9]+', '-', llm_afterclass_name.lower()).strip('-')
                    refined_count += 1

                    # Also update professor_lookup so resolution service gets corrected name
                    for alias_key, lookup_data in self._professor_lookup.items():
                        if lookup_data.get('database_id') == dto.id:
                            lookup_data['afterclass_name'] = llm_afterclass_name

        self._logger.info(
            f"LLM surname refinement complete: {refined_count}/{len(results_new)} new professors updated"
        )

    def _format_normalized_name(self, original_name: str, surname: str) -> Tuple[str, str]:
        """Format original name + identified surname into (boss_name, afterclass_name)."""
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

        return boss_name, afterclass_name

    def _normalize_professor_name_fallback(self, name: str) -> Tuple[str, str]:
        """Rule-based normalization - handles Asian/Western name formats."""
        if name is None or not str(name).strip():
            return "UNKNOWN", "Unknown"

        name_str = str(name).strip().replace("'", "'")
        name_str = re.sub(r'\s*\(.*\)\s*', ' ', name_str).strip()
        words = name_str.split()
        words_no_initials = [w for w in words if not (len(w) == 1 and w.isalpha()) and not (len(w) == 2 and w.endswith('.'))]
        name_str = ' '.join(words_no_initials)

        boss_name = name_str.upper()

        # Handle comma-separated Asian names (SURNAME, GIVEN NAME format)
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

        # Handle patronymic keywords (BIN, BINTE, S/O, D/O)
        for i, word in enumerate(words):
            if word.upper() in PATRONYMIC_KEYWORDS and i < len(words) - 1:
                surname_index = i + 1
                afterclass_parts = [w.capitalize() for w in words]
                afterclass_parts[i] = word.lower()
                afterclass_parts[surname_index] = words[surname_index].upper()
                return boss_name, ' '.join(afterclass_parts)

        # Rule-based surname identification
        surname_index = -1

        if words[0].upper() in WESTERN_GIVEN_NAMES:
            for i in range(1, len(words)):
                if words[i].upper() in ALL_ASIAN_SURNAMES:
                    surname_index = i
                    break
        elif any(w.upper() in WESTERN_GIVEN_NAMES for w in words) and words[0].upper() not in ALL_ASIAN_SURNAMES:
            surname_index = 0
        elif words[0].upper() in ALL_ASIAN_SURNAMES:
            surname_index = 0

        if surname_index == -1:
            surname_index = len(words) - 1

        # Post-processing for European particles
        afterclass_parts = [word.capitalize() for word in words]
        if surname_index > 0 and words[surname_index-1].upper() in SURNAME_PARTICLES:
            afterclass_parts[surname_index-1] = words[surname_index-1].upper()

        afterclass_parts[surname_index] = words[surname_index].upper()

        return boss_name, ' '.join(afterclass_parts)

    # ------------------------------------------------------------------------
    # Resolution Chain
    # ------------------------------------------------------------------------

    def _resolve_all_professors(
        self,
        unique_professors: set,
        normalized_map: Dict[str, Tuple[str, str]],
        professor_variations: Dict[str, set]
    ) -> Tuple[List[ProfessorDTO], List[ProfessorDTO]]:
        """Resolve all unique professors, returning new and updated lists."""
        results_new = []
        results_updated = []

        # Track new variations for EXISTING professors
        # Key: professor_id, Value: set of new variations to add to boss_aliases
        new_variations_for_existing: Dict[str, Set[str]] = {}

        for prof_name in unique_professors:
            boss_name, afterclass_name = normalized_map.get(prof_name, ("UNKNOWN", "Unknown"))

            result = self._resolve_single_professor(prof_name, boss_name, afterclass_name)

            if result is None:
                # No match found → create new
                dto = self._create_new_professor(prof_name, boss_name, afterclass_name, professor_variations.get(prof_name, set()))
                results_new.append(dto)
                self._new_professors_dtos.append(dto)
            else:
                # Professor was matched - track this variation for the existing professor
                # This ensures ALL raw name variations are stored as aliases
                prof_id = self._get_professor_id_from_lookup(prof_name)
                if prof_id:
                    if prof_id not in new_variations_for_existing:
                        new_variations_for_existing[prof_id] = set()
                    # Store the original scraped name as an alias
                    new_variations_for_existing[prof_id].add(prof_name.upper())
                    # Also store the boss_name variation
                    new_variations_for_existing[prof_id].add(boss_name.upper())

        # Update existing professors with new variations (add to boss_aliases)
        if new_variations_for_existing:
            results_updated = self._update_existing_professors_with_new_variations(new_variations_for_existing)

        return results_new, results_updated

    def _resolve_single_professor(
        self,
        prof_name: str,
        boss_name: str,
        afterclass_name: str
    ) -> Optional[str]:
        """Run 8-strategy resolution chain for a single professor.

        Returns:
            None if no match found (should CREATE)
            'matched' if found in lookup (no action needed)
        """
        normalized_prof_name = prof_name.upper()

        # Strategy 1: Direct lookup in professor_lookup
        if normalized_prof_name in self._professor_lookup:
            return 'matched'

        # Strategy 2: Boss name lookup in professor_lookup
        if boss_name.upper() in self._professor_lookup:
            self._professor_lookup[normalized_prof_name] = self._professor_lookup[boss_name.upper()]
            return 'matched'

        # Strategy 3-6: Cache lookups (includes boss_aliases from DB)
        cache_match = self._check_cache_for_professor(normalized_prof_name, boss_name, afterclass_name)
        if cache_match:
            self._professor_lookup[normalized_prof_name] = cache_match
            return 'matched'

        # Strategy 7: Subset matching (partial word match, requires >=2 words)
        # Handles cases like "Warren B. CHIK" matching "Kam Wai Warren Bartholomew CHIK"
        # Use boss_name (not prof_name) because boss_name has initials removed for consistency
        prof_words = set(boss_name.upper().split())
        if len(prof_words) >= 2:
            for lookup_boss_name, lookup_data in self._professor_lookup.items():
                lookup_words = set(lookup_boss_name.split())
                if prof_words.issubset(lookup_words):
                    self._professor_lookup[normalized_prof_name] = lookup_data
                    self._logger.info(f"Subset match: '{prof_name}' ({prof_words}) matched via '{lookup_boss_name}' ({lookup_words})")
                    return 'matched'

        # Strategy 8: New professors match (session deduplication)
        for dto in self._new_professors_dtos:
            # Check if normalized name matches any of the new professor's aliases
            for alias in dto.boss_aliases:
                if normalized_prof_name == alias.upper():
                    self._professor_lookup[normalized_prof_name] = {
                        'database_id': dto.id,
                        'boss_name': boss_name.upper(),
                        'afterclass_name': afterclass_name
                    }
                    return 'matched'
            # Also check if afterclass_name matches
            if afterclass_name.upper() == dto.name.upper():
                self._professor_lookup[normalized_prof_name] = {
                    'database_id': dto.id,
                    'boss_name': boss_name.upper(),
                    'afterclass_name': afterclass_name
                }
                return 'matched'

        # No match found → return None to indicate CREATE
        return None

    def _check_cache_for_professor(self, normalized_name: str, boss_name: str, afterclass_name: str) -> Optional[Dict]:
        """Check professors_cache for match (covers strategies 3-6: direct, boss_name, normalized, boss_aliases)."""
        # Direct cache match
        if normalized_name in self._professors_cache:
            return self._build_lookup_entry(self._professors_cache[normalized_name])

        if boss_name in self._professors_cache:
            return self._build_lookup_entry(self._professors_cache[boss_name])

        # Normalized cache match (remove commas, extra spaces)
        normalized_afterclass = ' '.join(normalized_name.replace(',', ' ').split())
        for cached_name, prof_data in self._professors_cache.items():
            cached_normalized = ' '.join(str(cached_name).replace(',', ' ').split()).upper()
            if normalized_afterclass == cached_normalized:
                return self._build_lookup_entry(prof_data)

            # Also check boss_aliases from the cached professor
            aliases = self.parse_boss_aliases(prof_data.get('boss_aliases', []))
            for alias in aliases:
                alias_normalized = ' '.join(alias.replace(',', ' ').split()).upper()
                if normalized_afterclass == alias_normalized:
                    return self._build_lookup_entry(prof_data)

        return None

    def _build_lookup_entry(self, prof_data: Dict) -> Dict:
        """Build standardized lookup entry from professor data."""
        return {
            'database_id': str(prof_data.get('id')),
            'boss_name': prof_data.get('name', '').upper(),
            'afterclass_name': prof_data.get('name', '')
        }

    # ------------------------------------------------------------------------
    # Create / Save
    # ------------------------------------------------------------------------

    def _create_new_professor(
        self,
        prof_name: str,
        boss_name: str,
        afterclass_name: str,
        variations: set
    ) -> ProfessorDTO:
        """Create new ProfessorDTO with generated UUID and slug."""
        professor_id = str(uuid.uuid4())
        slug = re.sub(r'[^a-zA-Z0-9]+', '-', afterclass_name.lower()).strip('-')

        # Build boss_aliases from boss_name + variations
        # CRITICAL: Store ALL variations found in raw data as aliases for future matching
        aliases_set = {boss_name}

        # Add the original scraped name (if different from boss_name)
        if prof_name.upper() != boss_name:
            aliases_set.add(prof_name.upper())

        # Add ALL variations found in raw data (from professor_variations dict)
        if variations:
            for variation in variations:
                if variation and variation.strip():
                    # Store the raw variation as-is (normalized to upper)
                    aliases_set.add(variation.upper().strip())
                    # Also store the normalized boss_name version
                    var_boss_name, _ = self._normalize_professor_name_fallback(variation.strip())
                    aliases_set.add(var_boss_name)

        boss_aliases_list = sorted(list(aliases_set))

        row = {
            'name': afterclass_name,
            'slug': slug,
            'boss_aliases': json.dumps(boss_aliases_list),
            'original_scraped_name': prof_name
        }

        return ProfessorDTO.from_row(row)

    def _build_lookup_from_cache(self):
        """Build professor_lookup from professors_cache (DB) for resolution.

        This populates self._professor_lookup with all known professor names
        and aliases from the database cache. These are used during the 8-strategy
        resolution chain to match incoming professor names.
        """
        loaded_count = 0
        for prof_name, prof_data in self._professors_cache.items():
            # Add the professor name itself
            if prof_name.upper() not in self._professor_lookup:
                self._professor_lookup[prof_name.upper()] = {
                    'database_id': str(prof_data.get('id')),
                    'boss_name': prof_name.upper(),
                    'afterclass_name': prof_data.get('name', '')
                }
                loaded_count += 1

            # Also add boss_aliases
            aliases = self.parse_boss_aliases(prof_data.get('boss_aliases', []))
            for alias in aliases:
                alias_upper = alias.upper()
                if alias_upper and alias_upper not in self._professor_lookup:
                    self._professor_lookup[alias_upper] = {
                        'database_id': str(prof_data.get('id')),
                        'boss_name': alias_upper,
                        'afterclass_name': prof_data.get('name', '')
                    }
                    loaded_count += 1

        self._logger.info(f"Built lookup from cache: {loaded_count} entries")

    def _save_professor_lookup_csv(self):
        """Save current state of professor_lookup to CSV (for human reference only).

        CSV is saved to script_output/professor_lookup.csv - NOT used for processing,
        only for human reference to review professor data.

        Format: one row per alias variation, mapping to the same afterclass_name and database_id.
        Includes existing, new, and updated professors with a method column.
        """
        lookup_data = []

        # Existing professors: one row per alias key in professor_lookup
        for alias_key, data in self._professor_lookup.items():
            lookup_data.append({
                'boss_name': alias_key,
                'afterclass_name': data['afterclass_name'],
                'database_id': data['database_id'],
                'method': 'exists'
            })

        # Updated professors: one row per alias
        for dto in getattr(self, '_results_updated', []):
            for alias in dto.boss_aliases:
                alias_upper = alias.upper()
                lookup_data.append({
                    'boss_name': alias_upper,
                    'afterclass_name': dto.name,
                    'database_id': dto.id,
                    'method': 'updated'
                })

        # New professors: one row per alias
        for dto in self._new_professors_dtos:
            for alias in dto.boss_aliases:
                alias_upper = alias.upper()
                lookup_data.append({
                    'boss_name': alias_upper,
                    'afterclass_name': dto.name,
                    'database_id': dto.id,
                    'method': 'new'
                })

        df = pd.DataFrame(lookup_data)
        output_path = 'script_output/professor_lookup.csv'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        self._logger.info(f"Saved {len(lookup_data)} entries to {output_path}")

    # ------------------------------------------------------------------------
    # Utility Functions
    # ------------------------------------------------------------------------

    def _extract_unique_professors(self) -> Tuple[set, Dict[str, set]]:
        """Extract unique professor names from raw data (both standalone and multiple sheets)."""
        unique_professors = set()
        professor_variations = defaultdict(set)

        # Load multiple sheet for professor name lookup
        multiple_df = None
        try:
            raw_data = pd.read_excel('script_input/raw_data.xlsx', sheet_name=['multiple'])
            multiple_df = raw_data['multiple']
        except Exception as e:
            self._logger.warning(f"Could not load multiple sheet: {e}")

        # Extract from standalone (legacy - may not have professor_name)
        for _, row in self._raw_data.iterrows():
            self._extract_professor_from_row(row, unique_professors, professor_variations)

        # Also extract from multiple sheet (this is where professor names actually are!)
        if multiple_df is not None and 'professor_name' in multiple_df.columns:
            for _, row in multiple_df.iterrows():
                self._extract_professor_from_row(row, unique_professors, professor_variations)

        return unique_professors, dict(professor_variations)

    def _extract_professor_from_row(
        self,
        row: pd.Series,
        unique_professors: Set[str],
        professor_variations: Dict[str, Set[str]]
    ) -> None:
        """Extract professor names from a row and populate unique_professors and professor_variations."""
        prof_name_raw = row.get('professor_name')
        if prof_name_raw is None or pd.isna(prof_name_raw):
            return

        prof_name = str(prof_name_raw).strip()
        if not prof_name or prof_name.lower() in ['nan', 'tba', 'to be announced']:
            return

        split_professors = self.resolution_service.split_professor_names(prof_name)
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
                else:
                    professor_variations[clean_prof].add(clean_prof)
