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
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set

from src.pipeline.processors.abstract_processor import AbstractProcessor
from src.pipeline.dtos.professor_dto import ProfessorDTO


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

    # Class constants
    _LLM_RATE_LIMIT_SECONDS = 6
    _LLM_BATCH_SIZE = 50

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
                return [ProfessorProcessor._clean_alias(item.strip().strip('"')) for item in content.split(',') if item.strip()]

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
        self._llm_client = None
        self._llm_model_name = "gemini-2.5-flash"
        self._llm_batch_size = self._LLM_BATCH_SIZE
        self._llm_prompt = None
        self._new_professors_dtos: List[ProfessorDTO] = []  # Session deduplication

    def set_llm_client(self, client, model_name: str, batch_size: int, prompt: str):
        """Configure LLM client for batch normalization."""
        self._llm_client = client
        self._llm_model_name = model_name
        self._llm_batch_size = batch_size
        self._llm_prompt = prompt

    def process(self) -> Tuple[List[ProfessorDTO], List[ProfessorDTO]]:
        """Main entry point - returns (new_professors, updated_professors)."""
        self._logger.info("Processing professors...")

        # Step 1: Build internal lookup from professors_cache (DB) for lookups
        self._build_lookup_from_cache()

        # Step 2: Extract unique professor names from raw data
        unique_professors, professor_variations = self._extract_unique_professors()
        self._logger.info(f"Found {len(unique_professors)} unique professor names")

        # Step 3: Normalize all names using LLM (with fallback to rule-based)
        normalized_map = self._normalize_professors_batch(unique_professors)

        # Step 4: For each professor - run FULL resolution chain
        results_new, results_updated = self._resolve_all_professors(
            unique_professors, normalized_map, professor_variations
        )

        # Step 5: Save updated lookup to CSV (for human reference only)
        self._save_professor_lookup_csv()

        self._logger.info(f"Professor processing complete: {len(results_new)} new, {len(results_updated)} updated")
        return results_new, results_updated

    # ------------------------------------------------------------------------
    # Normalization (LLM + fallback)
    # ------------------------------------------------------------------------

    def _normalize_professors_batch(self, names: List[str]) -> Dict[str, Tuple[str, str]]:
        """Normalize batch of names using LLM with rule-based fallback."""
        if not names:
            return {}

        normalized_map = {}

        # Try LLM first
        if self._llm_client and self._llm_prompt:
            try:
                normalized_map = self._call_llm_batch(names)
                self._logger.info("Batch normalization completed using Gemini LLM.")
            except Exception as e:
                self._logger.warning(f"LLM normalization failed ({e}). Falling back to rule-based.")
                normalized_map = {}

        # Fallback to rule-based for ALL names (including any that failed LLM)
        if not normalized_map:
            for name in names:
                normalized_map[name] = self._normalize_professor_name_fallback(name)
            self._logger.info("Used rule-based normalization for all names.")

        return normalized_map

    def _call_llm_batch(self, names: List[str]) -> Dict[str, Tuple[str, str]]:
        """Call LLM to normalize a batch of names."""
        from google import genai
        normalized_map = {}
        total_batches = (len(names) + self._llm_batch_size - 1) // self._llm_batch_size

        self._logger.info(f"Normalizing {len(names)} names in {total_batches} batches using '{self._llm_model_name}'...")

        for i in range(0, len(names), self._llm_batch_size):
            batch_names = names[i:i + self._llm_batch_size]
            batch_num = i // self._llm_batch_size + 1
            self._logger.info(f"  -> Processing batch {batch_num} of {total_batches} ({len(batch_names)} names)...")

            response = self._llm_client.models.generate_content(
                model=self._llm_model_name,
                contents=f"{self._llm_prompt}\n\n{json.dumps(batch_names)}",
                config=genai.GenerateContentConfig(response_mime_type="application/json")
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

        for prof_name in unique_professors:
            boss_name, afterclass_name = normalized_map.get(prof_name, ("UNKNOWN", "Unknown"))

            result = self._resolve_single_professor(prof_name, boss_name, afterclass_name)

            if result is None:
                # No match found → create new
                dto = self._create_new_professor(prof_name, boss_name, afterclass_name, professor_variations.get(prof_name, set()))
                results_new.append(dto)
                self._new_professors_dtos.append(dto)

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
        aliases_set = {boss_name}
        if variations:
            for variation in variations:
                if variation and variation.strip():
                    var_boss_name, _ = self._normalize_professor_name_fallback(variation.strip())
                    aliases_set.add(var_boss_name)
        if boss_name != prof_name.upper():
            aliases_set.add(prof_name.upper())

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
        """
        lookup_data = []
        seen_ids = set()
        for boss_name, data in self._professor_lookup.items():
            # Avoid duplicates - one entry per database_id
            if data['database_id'] in seen_ids:
                continue
            seen_ids.add(data['database_id'])
            lookup_data.append({
                'boss_name': data['boss_name'],
                'afterclass_name': data['afterclass_name'],
                'database_id': data['database_id'],
                'method': 'exists'
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
        """Extract unique professor names from raw data."""
        unique_professors = set()
        professor_variations = defaultdict(set)

        for _, row in self._raw_data.iterrows():
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
                    # Check for comma-separated format where extension is multi-word
                    # e.g., "YUESHEN, BART ZHOU" - we want to keep the full name intact
                    if ', ' in clean_prof:
                        parts = clean_prof.split(', ')
                        if len(parts) == 2:
                            base_name = parts[0].strip()
                            extension = parts[1].strip()
                            # Only track as variation if extension is single word
                            # Multi-word extension (like "BART ZHOU") means full name should be kept
                            if len(extension.split()) == 1:
                                professor_variations[clean_prof].add(base_name)
                                professor_variations[clean_prof].add(clean_prof)
                                if base_name in professor_variations:
                                    professor_variations[base_name].add(clean_prof)
                            else:
                                # Multi-word extension - full name is the real name, no split needed
                                professor_variations[clean_prof].add(clean_prof)
                    else:
                        professor_variations[clean_prof].add(clean_prof)

        return unique_professors, dict(professor_variations)

    def _split_professor_names(self, name: str) -> List[str]:
        """Split professor names using greedy longest-match-first algorithm.

        Uses professor_lookup to identify known professors. Unknown single-word
        parts are combined with the previous known professor, not treated as standalone.
        This prevents single-word names like "Hara" from becoming separate professors.
        """
        if not name:
            return []

        name_str = str(name).strip()

        # Quick return: if entire string is a known professor, return as-is
        # This handles comma-containing names like "LEE, MICHELLE PUI YEE"
        if name_str.upper() in self._professor_lookup:
            return [name_str]

        # No commas means single professor
        if ',' not in name_str:
            return [name_str] if name_str else []

        parts = [p.strip() for p in name_str.split(',') if p.strip()]

        found_professors = []
        i = 0

        while i < len(parts):
            match_found = False

            # Try longest combination first (greedy matching)
            for j in range(len(parts), i, -1):
                candidate = ', '.join(parts[i:j])
                if candidate.upper() in self._professor_lookup:
                    found_professors.append(candidate)
                    i = j
                    match_found = True
                    break

            # No match found - handle unknown parts
            if not match_found:
                unknown_part = parts[i]

                # KEY CONSTRAINT: Single-word unknown parts COMBINE with previous professor
                # This prevents standalone single-word names like "Hara" or "Eileen"
                if found_professors and len(unknown_part.split()) == 1:
                    found_professors[-1] = f"{found_professors[-1]}, {unknown_part}"
                    self._logger.info(f"Combined unknown single word '{unknown_part}' with previous professor -> '{found_professors[-1]}'")
                else:
                    # Multi-word unknown or no previous professor -> standalone
                    found_professors.append(unknown_part)

                i += 1

        return found_professors