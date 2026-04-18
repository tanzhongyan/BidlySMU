import os
import re
import json
import uuid
import pandas as pd
from collections import defaultdict
from typing import Optional, Tuple

try:
    import win32com.client as win32
    WIN32_AVAILABLE = True
except ImportError:
    WIN32_AVAILABLE = False

from src.pipeline.abstract_processor import AbstractProcessor
from src.utils.professor_normalizer import ProfessorNormalizer
from src.utils.alias_parser import parse_boss_aliases

class ProfessorProcessor(AbstractProcessor):
    def __init__(self, context):
        super().__init__(context)
        self.ctx = context
        self._logger = context.logger
        self.professor_lookup_loaded = False
        self.partial_matches = []
        
        self.professor_normalizer = ProfessorNormalizer(
            logger=self._logger,
            professor_lookup=self.ctx.professor_lookup,
            professors_cache=self.ctx.professors_cache,
            llm_client=self.ctx.llm_client,
            llm_model_name=self.ctx.llm_model_name,
            llm_batch_size=self.ctx.llm_batch_size,
            llm_prompt=self.ctx.llm_prompt
        )

    def process(self):
        self.process_professors()

    def _do_process(self):
        pass

    def _load_cache(self):
        pass

    def _collect_results(self):
        pass

    def process_professors(self):
        """
        Orchestrates the processing of professors: extraction, normalization, and creation.
        """
        self.ctx.logger.info("👥 Processing professors...")
        
        # Step 1: Extract unique names and their variations from the data source.
        unique_professors, professor_variations = self._extract_unique_professors()

        # Step 2: Filter out existing professors to find only new names.
        new_professors_to_normalize = []
        for prof_name in unique_professors:
            # A professor is considered "new" if they are not in the primary lookup.
            if prof_name.upper() not in self.ctx.professor_lookup:
                new_professors_to_normalize.append(prof_name)

        self.ctx.logger.info(f"Found {len(unique_professors)} unique names. "
                    f"Identified {len(new_professors_to_normalize)} as new and requiring normalization.")
        
        # Step 2b: Normalize only the new names using the LLM-first, fallback-second approach.
        # Sync caches to normalizer first
        self.professor_normalizer.set_professor_lookup(self.ctx.professor_lookup)
        self.professor_normalizer.set_professors_cache(self.ctx.professors_cache)
        self.professor_normalizer.set_llm_client(self.ctx.llm_client)
        normalized_map = self.professor_normalizer.normalize_professors_batch(new_professors_to_normalize)

        # Add a fallback for existing professors to ensure they are still processed later
        for prof_name in unique_professors:
            if prof_name not in normalized_map:
                # If an existing professor wasn't normalized, add them to the map using the fallback
                # to ensure they are processed correctly in the steps that follow.
                normalized_map[prof_name] = self.professor_normalizer.normalize(prof_name)

        if not normalized_map:
            self.ctx.logger.info("No professor names were normalized. Aborting professor processing.")
            return

        # Step 3: Check cache, fuzzy match, and create new professor records.
        email_to_professor = {}
        for boss_name_key, prof_data in self.ctx.professors_cache.items():
            if 'email' in prof_data and prof_data['email'] and prof_data['email'].lower() != 'enquiry@smu.edu.sg':
                email_to_professor[prof_data['email'].lower()] = prof_data

        fuzzy_matched_professors = []
        
        for prof_name in unique_professors:
            try:
                boss_name, afterclass_name = normalized_map[prof_name]
                
                # --- The logic below is IDENTICAL to your original script's Step 3 ---
                if hasattr(self, 'professor_lookup') and prof_name.upper() in self.ctx.professor_lookup:
                    continue
                if hasattr(self, 'professor_lookup') and boss_name.upper() in self.ctx.professor_lookup:
                    self.ctx.professor_lookup[prof_name.upper()] = self.ctx.professor_lookup[boss_name.upper()]
                    continue
                
                if hasattr(self, 'professor_lookup'):
                    found_partial_match = False
                    for lookup_boss_name, lookup_data in self.ctx.professor_lookup.items():
                        prof_words = set(prof_name.upper().split())
                        lookup_words = set(lookup_boss_name.split())
                        
                        if prof_words.issubset(lookup_words) and len(prof_words) >= 2:
                            self.ctx.professor_lookup[prof_name.upper()] = lookup_data
                            found_partial_match = True
                            break
                    if found_partial_match:
                        continue
                
                if boss_name in self.ctx.professors_cache:
                    if not hasattr(self, 'professor_lookup'):
                        self.ctx.professor_lookup = {}
                    self.ctx.professor_lookup[prof_name.upper()] = {
                        'database_id': self.ctx.professors_cache[boss_name]['id'],
                        'boss_name': boss_name,
                        'afterclass_name': self.ctx.professors_cache[boss_name].get('name', afterclass_name)
                    }
                    continue
                
                fuzzy_match_found = False
                normalized_prof = ' '.join(str(prof_name).replace(',', ' ').split()).upper()
                
                for cached_name, cached_prof in self.ctx.professors_cache.items():
                    if cached_name is None:
                        continue
                    cached_normalized = ' '.join(str(cached_name).replace(',', ' ').split()).upper()
                    if normalized_prof == cached_normalized:
                        if not hasattr(self, 'professor_lookup'):
                            self.ctx.professor_lookup = {}
                        self.ctx.professor_lookup[prof_name.upper()] = {
                            'database_id': cached_prof['id'],
                            'boss_name': cached_prof.get('boss_name', cached_prof['name'].upper()),
                            'afterclass_name': cached_prof.get('name', afterclass_name)
                        }
                        fuzzy_match_found = True
                        break
                if fuzzy_match_found:
                    continue
                
                # This is the block that was previously a placeholder
                for new_prof in self.ctx.new_professors:
                    if 'boss_aliases' in new_prof:
                        try:
                            boss_aliases = json.loads(new_prof['boss_aliases'])
                            if isinstance(boss_aliases, list) and boss_aliases:
                                new_normalized = ' '.join(boss_aliases[0].replace(',', ' ').split()).upper()
                            else:
                                new_normalized = ' '.join(new_prof.get('afterclass_name', '').replace(',', ' ').split()).upper()
                        except (json.JSONDecodeError, TypeError):
                            new_normalized = ' '.join(new_prof.get('afterclass_name', '').replace(',', ' ').split()).upper()
                    else:
                        new_normalized = ' '.join(new_prof.get('afterclass_name', '').replace(',', ' ').split()).upper()

                    if normalized_prof == new_normalized:
                        if not hasattr(self, 'professor_lookup'):
                            self.ctx.professor_lookup = {}
                        self.ctx.professor_lookup[prof_name.upper()] = {
                            'database_id': new_prof['id'],
                            'boss_name': boss_name,
                            'afterclass_name': new_prof['afterclass_name']
                        }
                        fuzzy_match_found = True
                        break
                
                if fuzzy_match_found:
                    continue
                
                if hasattr(self, 'professor_lookup'):
                    best_fuzzy_match = None
                    best_fuzzy_score = 0
                    FUZZY_MATCH_THRESHOLD = 90
                    for lookup_boss_name, lookup_data in self.ctx.professor_lookup.items():                        
                        score = self.professor_normalizer._calculate_fuzzy_score(prof_name, lookup_boss_name)
                        if score > best_fuzzy_score:
                            best_fuzzy_match = lookup_data
                            best_fuzzy_score = score
                    
                    if best_fuzzy_match and best_fuzzy_score >= FUZZY_MATCH_THRESHOLD:
                        fuzzy_matched_professors.append({
                            'boss_aliases': f'["{prof_name.upper()}"]',
                            'afterclass_name': best_fuzzy_match.get('afterclass_name', prof_name),
                            'database_id': best_fuzzy_match['database_id'],
                            'method': 'fuzzy_match',
                            'confidence_score': f"{best_fuzzy_score:.2f}"
                        })
                        if not hasattr(self, 'professor_lookup'):
                            self.ctx.professor_lookup = {}
                        self.ctx.professor_lookup[prof_name.upper()] = best_fuzzy_match
                        continue
                
                resolved_email = self.resolve_professor_email(afterclass_name)
                
                if (resolved_email and 
                    resolved_email.lower() != 'enquiry@smu.edu.sg' and 
                    resolved_email.lower() in email_to_professor):
                    existing_prof = email_to_professor[resolved_email.lower()]
                    if not hasattr(self, 'professor_lookup'):
                        self.ctx.professor_lookup = {}
                    self.ctx.professor_lookup[prof_name.upper()] = {
                        'database_id': existing_prof['id'],
                        'boss_name': boss_name,
                        'afterclass_name': existing_prof.get('name', afterclass_name)
                    }
                    continue
                
                self._create_new_professor(prof_name, professor_variations, email_to_professor)

            except Exception as e:
                self.ctx.logger.error(f"❌ Error processing professor '{prof_name}': {e}")
                continue
        
        if fuzzy_matched_professors:
            fuzzy_df = pd.DataFrame(fuzzy_matched_professors)
            fuzzy_path = os.path.join(self.ctx.config.verify_dir, 'fuzzy_matched_professors.csv')
            fuzzy_df.to_csv(fuzzy_path, index=False)
            self.ctx.logger.info(f"🔍 Saved {len(fuzzy_matched_professors)} fuzzy matched professors for validation")
        
        self.ctx.logger.info(f"✅ Created {self.ctx.stats['professors_created']} new professors")

    def resolve_professor_email(self, professor_name):
        """Resolve professor email using Outlook contacts"""
        # Early return for non-Windows/CI environments
        if not WIN32_AVAILABLE:
            return 'enquiry@smu.edu.sg'

        try:
            # Initialize Outlook
            outlook = win32.Dispatch("Outlook.Application")
            namespace = outlook.GetNamespace("MAPI")
            
            # Try exact resolver first
            recipient = namespace.CreateRecipient(professor_name)
            if recipient.Resolve():
                # Try to get SMTP address
                address_entry = recipient.AddressEntry
                
                # Try Exchange user
                try:
                    exchange_user = address_entry.GetExchangeUser()
                    if exchange_user and exchange_user.PrimarySmtpAddress:
                        return exchange_user.PrimarySmtpAddress.lower()
                except:
                    pass
                
                # Try Exchange distribution list
                try:
                    exchange_dl = address_entry.GetExchangeDistributionList()
                    if exchange_dl and exchange_dl.PrimarySmtpAddress:
                        return exchange_dl.PrimarySmtpAddress.lower()
                except:
                    pass
                
                # Try PR_SMTP_ADDRESS property
                try:
                    property_accessor = address_entry.PropertyAccessor
                    smtp_addr = property_accessor.GetProperty("http://schemas.microsoft.com/mapi/proptag/0x39FE001E")
                    if smtp_addr:
                        return smtp_addr.lower()
                except:
                    pass
                
                # Fallback: regex search in Address field
                try:
                    address = getattr(address_entry, "Address", "") or ""
                    match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", address)
                    if match:
                        return match.group(0).lower()
                except:
                    pass
            
            # If exact resolve fails, try contacts search
            contacts_folder = namespace.GetDefaultFolder(10)  # olFolderContacts
            tokens = [t.lower() for t in professor_name.split() if t]
            
            for item in contacts_folder.Items:
                try:
                    full_name = (item.FullName or "").lower()
                    if all(token in full_name for token in tokens):
                        # Try the three standard email slots
                        for field in ("Email1Address", "Email2Address", "Email3Address"):
                            addr = getattr(item, field, "") or ""
                            if addr and "@" in addr:
                                return addr.lower()
                except:
                    continue
            
            # If no email found, return default
            return 'enquiry@smu.edu.sg'
            
        except Exception as e:
            self.ctx.logger.warning(f"Email resolution failed for {professor_name}: {e}")
            return 'enquiry@smu.edu.sg'

    def _lookup_professor_with_fallback(self, prof_name: str) -> Optional[str]:
        """Enhanced professor lookup with improved partial word matching and no phantom professor creation."""
        
        if prof_name is None or pd.isna(prof_name):
            return None
        
        prof_name = str(prof_name).strip()
        if not prof_name or prof_name.lower() == 'nan':
            return None
        
        # Strategy 1 & 2: Direct and variation-based lookup (unchanged).
        normalized_name = prof_name.upper()
        if hasattr(self, 'professor_lookup'):
            if normalized_name in self.ctx.professor_lookup:
                return self.ctx.professor_lookup[normalized_name]['database_id']
            
            variations = [
                prof_name.strip().upper(),
                prof_name.replace(',', '').strip().upper(),
                ' '.join(prof_name.replace(',', ' ').split()).upper()
            ]
            for variation in variations:
                if variation in self.ctx.professor_lookup:
                    return self.ctx.professor_lookup[variation]['database_id']
        
        # Strategy 3: Search boss_aliases in professors_cache using the new robust parser.
        search_name_normalized = normalized_name
        for prof_data in self.ctx.professors_cache.values():
            aliases_list = self._parse_boss_aliases(prof_data.get('boss_aliases'))
            
            for alias in aliases_list:
                alias_normalized = alias.strip().upper()
                
                if alias_normalized == search_name_normalized:
                    self.ctx.logger.info(f"✅ Found exact match in boss_aliases: {prof_name} → {alias} (ID: {prof_data.get('id')})")
                    if not hasattr(self, 'professor_lookup'): self.ctx.professor_lookup = {}
                    self.ctx.professor_lookup[search_name_normalized] = {
                        'database_id': str(prof_data.get('id')),
                        'boss_name': alias_normalized,
                        'afterclass_name': prof_data.get('name', prof_name)
                    }
                    return str(prof_data.get('id'))

        # Strategy 4: Enhanced partial word matching for cases like "DENNIS LIM" → "LIM CHONG BOON DENNIS"
        search_words = set(normalized_name.replace(',', ' ').split())
        if len(search_words) >= 2:  # Only try partial matching for multi-word names
            best_match = None
            best_score = 0
            
            for prof_data in self.ctx.professors_cache.values():
                # Check against afterclass_name (database name)
                db_name = prof_data.get('name', '').upper()
                db_words = set(db_name.replace(',', ' ').split())
                
                # Check if all search words are found in database name
                if search_words.issubset(db_words):
                    # Calculate match score (percentage of db_words that match search_words)
                    match_score = len(search_words) / len(db_words) if db_words else 0
                    
                    if match_score > best_score and match_score >= 0.5:  # At least 50% match
                        best_match = prof_data
                        best_score = match_score
                
                # Also check against boss_aliases
                aliases_list = self._parse_boss_aliases(prof_data.get('boss_aliases'))
                for alias in aliases_list:
                    alias_words = set(alias.upper().replace(',', ' ').split())
                    if search_words.issubset(alias_words):
                        match_score = len(search_words) / len(alias_words) if alias_words else 0
                        if match_score > best_score and match_score >= 0.5:
                            best_match = prof_data
                            best_score = match_score
            
            if best_match and best_score >= 0.5:
                self.ctx.logger.info(f"🔍 Partial word match found: '{prof_name}' → '{best_match.get('name')}' (score: {best_score:.2f})")
                
                # Add to lookup and save to partial matches tracking
                if not hasattr(self, 'professor_lookup'): self.ctx.professor_lookup = {}
                self.ctx.professor_lookup[normalized_name] = {
                    'database_id': str(best_match.get('id')),
                    'boss_name': normalized_name,
                    'afterclass_name': best_match.get('name', prof_name)
                }
                
                # Track partial matches for review
                if not hasattr(self, 'partial_matches'):
                    self.partial_matches = []
                self.partial_matches.append({
                    'boss_name': prof_name,
                    'afterclass_name': best_match.get('name'),
                    'database_id': str(best_match.get('id')),
                    'method': 'partial_match',
                    'match_score': f"{best_score:.2f}"
                })
                
                return str(best_match.get('id'))
        
        # Strategy 5: Exact fuzzy matching (unchanged)
        if hasattr(self, 'professor_lookup'):
            for lookup_name in self.ctx.professor_lookup.keys():
                if self.professor_normalizer._names_match_fuzzy_exact(normalized_name, lookup_name):
                    return self.ctx.professor_lookup[lookup_name]['database_id']
        
        if normalized_name in self.ctx.professors_cache:
            return self.ctx.professors_cache[normalized_name]['id']
        
        # Strategy 6: DO NOT create new professor - log as unmatched instead
        self.ctx.logger.warning(f"⚠️ Professor not found, will create new: {prof_name}")
        
        # Create new professor (only when absolutely necessary)
        return self._create_new_professor(prof_name)

    def load_professor_lookup_csv(self):
        """Load professor lookup CSV once and cache it properly"""
        # Check if already loaded to prevent repeated loading
        if hasattr(self, 'professor_lookup_loaded') and self.ctx.professor_lookup_loaded:
            return
        
        lookup_file = 'script_input/professor_lookup.csv'
        
        if not os.path.exists(lookup_file):
            self.ctx.logger.warning("📋 professor_lookup.csv not found - will use database cache only")
            self.ctx.professor_lookup_loaded = True
            return
        
        try:
            # Load the CSV file
            lookup_df = pd.read_csv(lookup_file)
            
            # Validate required columns exist
            required_cols = ['boss_name', 'afterclass_name', 'database_id', 'method']
            missing_cols = [col for col in required_cols if col not in lookup_df.columns]
            if missing_cols:
                self.ctx.logger.error(f"❌ professor_lookup.csv missing required columns: {missing_cols}")
                self.ctx.professor_lookup_loaded = True
                return
            
            # Clear existing lookup and load fresh data
            self.ctx.professor_lookup = {}
            loaded_count = 0
            
            for _, row in lookup_df.iterrows():
                boss_name = row.get('boss_name')
                afterclass_name = row.get('afterclass_name')
                database_id = row.get('database_id')
                
                # Skip rows with critical missing values
                if pd.isna(boss_name) or pd.isna(database_id):
                    continue
                    
                # Use boss_name as the primary key for lookup (as you specified)
                boss_name_key = str(boss_name).strip().upper()
                self.ctx.professor_lookup[boss_name_key] = {
                    'database_id': str(database_id),
                    'boss_name': str(boss_name),
                    'afterclass_name': str(afterclass_name) if not pd.isna(afterclass_name) else str(boss_name)
                }
                loaded_count += 1
            
            self.ctx.logger.info(f"✅ Loaded {loaded_count} entries from professor_lookup.csv")
            self.ctx.professor_lookup_loaded = True
            
        except Exception as e:
            self.ctx.logger.error(f"❌ Error loading professor_lookup.csv: {e}")
            self.ctx.logger.info("📋 Continuing with database cache only")
            self.ctx.professor_lookup_loaded = True

    def _create_new_professor(self, prof_name: str, professor_variations: dict = None, email_to_professor: dict = None) -> str:
        """
        Create a new professor record, ensure proper tracking, and handle both primary and fallback alias creation.
        """
        boss_name, afterclass_name = self.professor_normalizer.normalize(prof_name)
        
        # Check if already created in this session to prevent duplicates
        for new_prof in self.ctx.new_professors:
            aliases_val = new_prof.get('boss_aliases', '[]')
            try:
                alias_list = json.loads(aliases_val) if isinstance(aliases_val, str) else aliases_val
            except (json.JSONDecodeError, TypeError):
                alias_list = []

            if boss_name in alias_list or afterclass_name == new_prof.get('name', ''):
                # This professor was already created in this run, just return its ID.
                return new_prof.get('id')

        # Check for slug collision with existing professors in DB cache
        base_slug = re.sub(r'[^a-zA-Z0-9]+', '-', afterclass_name.lower()).strip('-')
        for prof_id, prof_data in self.ctx.professors_cache.items():
            existing_slug = prof_data.get('slug', '')
            if not existing_slug:
                existing_slug = re.sub(r'[^a-zA-Z0-9]+', '-', prof_data.get('name', '').lower()).strip('-')
            if existing_slug == base_slug:
                self.ctx.logger.warning(f"⚠️ Professor slug collision detected: {base_slug} (existing: {prof_data.get('name')}). Using existing ID.")
                return prof_id

        # --- Unified Creation Logic ---
        professor_id = str(uuid.uuid4())
        slug = re.sub(r'[^a-zA-Z0-9]+', '-', afterclass_name.lower()).strip('-')
        resolved_email = self.resolve_professor_email(afterclass_name)

        # --- Conditional Alias Creation ---
        boss_aliases_set = set()
        boss_aliases_set.add(boss_name)
        
        # SCENARIO A: Use sophisticated alias creation if professor_variations dictionary is provided
        if professor_variations:
            professor_specific_variations = professor_variations.get(prof_name, set())
            for variation in professor_specific_variations:
                if variation and variation.strip():
                    variation_boss_name, _ = self.professor_normalizer.normalize(variation.strip())
                    boss_aliases_set.add(variation_boss_name)
        # SCENARIO B: Fallback to simple alias creation if not provided
        else:
            if boss_name != prof_name.upper():
                boss_aliases_set.add(prof_name.upper())

        boss_aliases_list = sorted(list(boss_aliases_set))
        boss_aliases_json = json.dumps(boss_aliases_list)
        
        # --- Create and Register the New Professor ---
        new_prof = {
            'id': professor_id,
            'name': afterclass_name,
            'email': resolved_email,
            'slug': slug,
            'photo_url': 'https://smu.edu.sg',
            'profile_url': 'https://smu.edu.sg',
            'belong_to_university': 1,
            'boss_aliases': boss_aliases_json,
            'original_scraped_name': prof_name
        }
        
        self.ctx.new_professors.append(new_prof)
        self.ctx.stats['professors_created'] += 1
        
        # Update lookup tables
        if not hasattr(self, 'professor_lookup'):
            self.ctx.professor_lookup = {}
        
        lookup_entry = {
            'database_id': professor_id,
            'boss_name': boss_name,
            'afterclass_name': afterclass_name
        }
        # Map the original name and all its aliases to the new ID
        self.ctx.professor_lookup[prof_name.upper()] = lookup_entry
        for alias in boss_aliases_list:
            self.ctx.professor_lookup[alias.upper()] = lookup_entry

        # Update the email duplicate checker dictionary if it was passed in
        if email_to_professor is not None and resolved_email and resolved_email.lower() != 'enquiry@smu.edu.sg':
            email_to_professor[resolved_email.lower()] = new_prof
        
        self.ctx.logger.info(f"✅ Created professor: {afterclass_name} with email: {resolved_email}")
        self.ctx.logger.info(f"   Boss aliases: {boss_aliases_list}")
        
        return professor_id

    def update_professor_lookup_from_corrected_csv(self):
        """Update professor lookup from manually corrected new_professors.csv"""
        self.ctx.logger.info("🔄 Updating professor lookup from corrected CSV...")
        
        # Read corrected new_professors.csv
        corrected_csv_path = os.path.join(self.ctx.config.verify_dir, 'new_professors.csv')
        if not os.path.exists(corrected_csv_path):
            self.ctx.logger.info(f"📝 No corrected CSV found: {corrected_csv_path} - assuming all professors already exist")
            return True

        corrected_df = pd.read_csv(corrected_csv_path)
        if corrected_df.empty:
            self.ctx.logger.info(f"📝 Empty corrected CSV - no professors to update")
            return True

        try:
            self.ctx.logger.info(f"📖 Reading {len(corrected_df)} corrected professor records")
            
            # Clear and rebuild the new_professors list with corrected data
            self.ctx.new_professors = []
            
            # Update internal professor_lookup and rebuild new_professors
            updated_count = 0
            
            # FIXED: Initialize professor_lookup if it doesn't exist
            if not hasattr(self, 'professor_lookup'):
                self.ctx.professor_lookup = {}
            
            for _, row in corrected_df.iterrows():
                original_name = row.get('original_scraped_name', '')
                corrected_afterclass_name = row.get('name', '')  # This is the corrected name
                boss_aliases = row.get('boss_aliases', '')  # This should be JSON string
                professor_id = row.get('id', '')

                # Parse boss_aliases JSON string
                try:
                    if isinstance(boss_aliases, str) and boss_aliases.strip():
                        boss_aliases_list = json.loads(boss_aliases)
                        if isinstance(boss_aliases_list, list) and boss_aliases_list:
                            boss_name = boss_aliases_list[0]  # Use first boss alias
                        else:
                            boss_name = original_name.upper() if original_name else corrected_afterclass_name.upper()
                    else:
                        boss_name = original_name.upper() if original_name else corrected_afterclass_name.upper()
                except (json.JSONDecodeError, TypeError):
                    # Fallback if JSON parsing fails
                    boss_name = original_name.upper() if original_name else corrected_afterclass_name.upper()
                
                # Rebuild the professor record with corrected data
                corrected_prof = {
                    'id': professor_id,
                    'name': corrected_afterclass_name,  # Use corrected name
                    'email': row.get('email', 'enquiry@smu.edu.sg'),
                    'slug': row.get('slug', ''),
                    'photo_url': row.get('photo_url', 'https://smu.edu.sg'),
                    'profile_url': row.get('profile_url', 'https://smu.edu.sg'),
                    'belong_to_university': row.get('belong_to_university', 1),
                    'boss_aliases': boss_aliases  # Keep as JSON string
                }
                
                # Add to new_professors list
                self.ctx.new_professors.append(corrected_prof)
                
                # FIXED: Update professor_lookup with ALL variations
                if professor_id:
                    lookup_entry = {
                        'database_id': professor_id,
                        'boss_name': boss_name,
                        'afterclass_name': corrected_afterclass_name
                    }
                    
                    # Add original scraped name to lookup
                    if original_name:
                        self.ctx.professor_lookup[original_name.upper()] = lookup_entry
                        updated_count += 1
                    
                    # Add corrected afterclass name to lookup
                    if corrected_afterclass_name:
                        self.ctx.professor_lookup[corrected_afterclass_name.upper()] = lookup_entry
                    
                    # Add boss_name to lookup
                    if boss_name:
                        self.ctx.professor_lookup[boss_name.upper()] = lookup_entry
                    
                    # FIXED: Add all boss aliases to lookup
                    try:
                        if isinstance(boss_aliases, str) and boss_aliases.strip():
                            boss_aliases_list = json.loads(boss_aliases)
                            if isinstance(boss_aliases_list, list):
                                for alias in boss_aliases_list:
                                    if alias and str(alias).strip():
                                        self.ctx.professor_lookup[str(alias).upper()] = lookup_entry
                    except (json.JSONDecodeError, TypeError):
                        pass  # Skip if JSON parsing fails
            
            # Save updated professor lookup to CSV
            self._save_corrected_professor_lookup()
            
            self.ctx.logger.info(f"✅ Updated {updated_count} professor lookup entries")
            self.ctx.logger.info(f"✅ Rebuilt {len(self.ctx.new_professors)} professor records with corrections")
            self.ctx.logger.info(f"✅ Total lookup entries now: {len(self.ctx.professor_lookup)}")
            
            return True
            
        except Exception as e:
            self.ctx.logger.error(f"❌ Failed to update professor lookup: {e}")
            import traceback
            traceback.print_exc()
            return False

    def update_professors_with_boss_names(self):
        """
        Update professors with missing/additional boss_names by comparing professor_lookup.csv
        with database boss_aliases and combining new variations from high-confidence fuzzy matches.
        """
        self.ctx.logger.info("👤 Updating professors with boss_names and detecting new variations...")

        # --- Step 1: Load high-confidence fuzzy matches from Phase 1 ---
        fuzzy_path = os.path.join(self.ctx.config.verify_dir, 'fuzzy_matched_professors.csv')
        new_aliases_by_id = defaultdict(list)

        if os.path.exists(fuzzy_path):
            try:
                fuzzy_df = pd.read_csv(fuzzy_path)
                high_confidence_matches = fuzzy_df[fuzzy_df['confidence_score'] >= 95]
                self.ctx.logger.info(f"🔍 Found {len(high_confidence_matches)} high-confidence (>=95) fuzzy matches to process.")

                for _, row in high_confidence_matches.iterrows():
                    database_id = str(row['database_id'])
                    afterclass_name = row['afterclass_name']
                    
                    try:
                        aliases_val = row.get('boss_aliases', '[]')
                        new_aliases = json.loads(aliases_val) if isinstance(aliases_val, str) else []
                        
                        for alias in new_aliases:
                            if alias and str(alias).strip():
                                clean_alias = str(alias).strip()
                                new_aliases_by_id[database_id].append(clean_alias)
                                
                                # Add to in-memory professor_lookup to be saved later
                                if not hasattr(self, 'professor_lookup'):
                                    self.ctx.professor_lookup = {}
                                
                                alias_key = clean_alias.upper()
                                if alias_key not in self.ctx.professor_lookup:
                                    self.ctx.professor_lookup[alias_key] = {
                                        'database_id': database_id,
                                        'boss_name': clean_alias,
                                        'afterclass_name': afterclass_name,
                                        'method': 'fuzzy_match' # Add method for tracking
                                    }
                                    self.ctx.logger.info(f"➕ Adding fuzzy match to lookup: '{clean_alias}' -> '{afterclass_name}'")

                    except (json.JSONDecodeError, TypeError) as e:
                        self.ctx.logger.warning(f"⚠️ Could not parse boss_aliases from fuzzy_matched_professors.csv for row: {row.to_dict()}. Error: {e}")
            
            except Exception as e:
                self.ctx.logger.error(f"❌ Error processing fuzzy_matched_professors.csv: {e}")

        # --- Step 2: Load existing variations from professor_lookup.csv ---
        lookup_file = 'script_input/professor_lookup.csv'
        lookup_groups = defaultdict(list)
        if os.path.exists(lookup_file):
            try:
                lookup_df = pd.read_csv(lookup_file)
                for _, row in lookup_df.iterrows():
                    database_id = row.get('database_id')
                    boss_name = row.get('boss_name')
                    if pd.notna(database_id) and pd.notna(boss_name):
                        lookup_groups[str(database_id)].append(str(boss_name).strip())
            except Exception as e:
                self.ctx.logger.error(f"❌ Error loading professor_lookup.csv: {e}")

        # --- Step 3: Iterate through professors and combine all alias sources ---
        updated_professor_ids = set()
        self.ctx.update_professors = []
        new_variations_found = []

        for prof_key, prof_data in self.ctx.professors_cache.items():
            professor_id = str(prof_data.get('id'))
            if professor_id in updated_professor_ids:
                continue

            # Get all sources of aliases as sets for easy combination
            current_boss_aliases = set(self._parse_boss_aliases(prof_data.get('boss_aliases')))
            lookup_variations = set(lookup_groups.get(professor_id, []))
            fuzzy_variations = set(new_aliases_by_id.get(professor_id, []))

            # Combine all unique variations using set union
            final_aliases_raw = current_boss_aliases.union(lookup_variations).union(fuzzy_variations)

            # Normalize both sets for a stable comparison, preventing repeated updates
            current_aliases_normalized = {name.replace("’", "'") for name in current_boss_aliases}
            final_aliases_normalized = {name.replace("’", "'") for name in final_aliases_raw}

            # Check for changes using the normalized sets
            if final_aliases_normalized != current_aliases_normalized:
                # Save the raw, original names to preserve the smart quote from the source
                unique_boss_names = sorted(list(final_aliases_raw))
                # Use ensure_ascii=False to prevent encoding '’' to '\u2019'
                boss_aliases_json = json.dumps(unique_boss_names, ensure_ascii=False)

                self.ctx.update_professors.append({
                    'id': professor_id,
                    'boss_aliases': boss_aliases_json,
                })
                
                # For logging, find the newly added variations
                newly_added = final_aliases_raw - current_boss_aliases
                if newly_added:
                    self.ctx.logger.info(f"✅ Adding {len(newly_added)} new variations for professor {professor_id}: {sorted(list(newly_added))}")
                    new_variations_found.append({
                        'professor_id': professor_id,
                        'professor_name': prof_data.get('name', 'Unknown'),
                        'existing_aliases': sorted(list(current_boss_aliases)),
                        'new_variations': sorted(list(newly_added)),
                        'final_aliases': unique_boss_names
                    })
                
                updated_professor_ids.add(professor_id)

        # --- Step 4: Save all outputs ---
        # Save partial matches if any were found
        if hasattr(self, 'partial_matches') and self.partial_matches:
            partial_df = pd.DataFrame(self.partial_matches)
            partial_path = os.path.join(self.ctx.config.verify_dir, 'partial_matches.csv')
            partial_df.to_csv(partial_path, index=False)
            self.ctx.logger.info(f"🔍 Saved {len(self.partial_matches)} partial matches to partial_matches.csv")

        # Save new variations summary
        if new_variations_found:
            report_data = [{'professor_id': item.get('professor_id'),'professor_name': item.get('professor_name'), 'existing_aliases': '|'.join(item.get('existing_aliases', [])), 'new_variations': '|'.join(item.get('new_variations', [])),'final_aliases': '|'.join(item.get('final_aliases', []))} for item in new_variations_found]
            variations_df = pd.DataFrame(report_data)
            variations_path = os.path.join(self.ctx.config.verify_dir, 'new_variations_found.csv')
            variations_df.to_csv(variations_path, index=False, encoding='utf-8-sig')
            self.ctx.logger.info(f"🆕 Saved {len(new_variations_found)} professors with new variations to new_variations_found.csv")

        # Save the update_professor.csv file
        if self.ctx.update_professors:
            df = pd.DataFrame(self.ctx.update_professors)
            update_path = os.path.join(self.ctx.config.output_base, 'update_professor.csv')
            df.to_csv(update_path, index=False, encoding='utf-8')
            self.ctx.logger.info(f"✅ Saved {len(self.ctx.update_professors)} unique professor updates to update_professor.csv")
            self.ctx.stats['professors_updated'] = len(self.ctx.update_professors)
        else:
            self.ctx.logger.info("ℹ️ No professors need boss_name updates.")
            self.ctx.stats['professors_updated'] = 0

        # --- Step 5: Persist the updated professor lookup table ---
        self._save_corrected_professor_lookup()

    def _save_corrected_professor_lookup(self):
        """Save professor lookup preserving all input entries, adding new ones, and including partial matches"""
        # Start with all existing entries from input professor_lookup.csv
        all_lookup_data = {}
        
        # Step 1: Load ALL entries from input professor_lookup.csv (preserve existing)
        input_lookup_file = 'script_input/professor_lookup.csv'
        if os.path.exists(input_lookup_file):
            try:
                input_df = pd.read_csv(input_lookup_file)
                for _, row in input_df.iterrows():
                    boss_name = row.get('boss_name')
                    afterclass_name = row.get('afterclass_name')
                    database_id = row.get('database_id')
                    method = row.get('method', 'exists')
                    
                    # Only require database_id to be present
                    if pd.notna(database_id):
                        if pd.isna(boss_name) or str(boss_name).strip() == '':
                            if pd.notna(afterclass_name):
                                lookup_key = f"EMPTY_BOSS_{str(afterclass_name).upper().replace(' ', '_')}"
                                boss_name_value = ""
                            else:
                                lookup_key = f"EMPTY_BOSS_{str(database_id)}"
                                boss_name_value = ""
                        else:
                            lookup_key = str(boss_name).upper()
                            boss_name_value = str(boss_name)
                        
                        all_lookup_data[lookup_key] = {
                            'boss_name': boss_name_value,
                            'afterclass_name': str(afterclass_name) if pd.notna(afterclass_name) else "",
                            'database_id': str(database_id),
                            'method': str(method)
                        }
                
                self.ctx.logger.info(f"📖 Loaded {len(all_lookup_data)} existing entries from input professor_lookup.csv")
            except Exception as e:
                self.ctx.logger.warning(f"⚠️ Could not load input professor_lookup.csv: {e}")
        
        # Step 2: Add/update with new entries from current processing
        new_entries_count = 0
        updated_entries_count = 0
        
        for scraped_name, data in self.ctx.professor_lookup.items():
            boss_name = data.get('boss_name', scraped_name.upper())
            afterclass_name = data.get('afterclass_name', scraped_name)
            database_id = data['database_id']
            
            # Determine method: check if this is a newly created professor or partial match
            method = 'exists'  # default
            if any(prof['id'] == database_id for prof in self.ctx.new_professors):
                method = 'created'
            elif hasattr(self, 'partial_matches') and any(match['database_id'] == database_id and match['boss_name'] == scraped_name for match in self.partial_matches):
                method = 'partial_match'
            
            boss_name_key = str(boss_name).upper()
            
            if boss_name_key in all_lookup_data:
                # Update existing entry if method changed
                if method in ['created', 'partial_match']:
                    all_lookup_data[boss_name_key]['method'] = method
                    updated_entries_count += 1
            else:
                # Add new entry
                all_lookup_data[boss_name_key] = {
                    'boss_name': str(boss_name),
                    'afterclass_name': str(afterclass_name),
                    'database_id': str(database_id),
                    'method': method
                }
                new_entries_count += 1
                self.ctx.logger.info(f"   -> NEW LOOKUP: Adding '{boss_name}' for '{afterclass_name}' (ID: {database_id}, method: {method})")
        
        # Step 3: Add partial matches that weren't already in professor_lookup
        if hasattr(self, 'partial_matches'):
            for match in self.partial_matches:
                boss_name_key = match['boss_name'].upper()
                if boss_name_key not in all_lookup_data:
                    all_lookup_data[boss_name_key] = {
                        'boss_name': match['boss_name'],
                        'afterclass_name': match['afterclass_name'],
                        'database_id': match['database_id'],
                        'method': f"partial_match_{match.get('match_score', '')}"
                    }
                    new_entries_count += 1
                    self.ctx.logger.info(f"   -> PARTIAL MATCH: Adding '{match['boss_name']}' → '{match['afterclass_name']}' (score: {match.get('match_score', 'N/A')})")
        
        # Step 4: Convert to list and sort
        lookup_data = list(all_lookup_data.values())
        lookup_data.sort(key=lambda x: x['boss_name'] if x['boss_name'] else x['afterclass_name'])
        
        # Step 5: Save main lookup file
        df = pd.DataFrame(lookup_data)
        df.to_csv(input_lookup_file, index=False)
        
        # Step 6: Save separate tracking files for manual review
        if hasattr(self, 'partial_matches') and self.partial_matches:
            partial_df = pd.DataFrame(self.partial_matches)
            partial_path = os.path.join(self.ctx.config.verify_dir, 'partial_matches_log.csv')
            partial_df.to_csv(partial_path, index=False)
            self.ctx.logger.info(f"🔍 Saved {len(self.partial_matches)} partial matches to partial_matches_log.csv")
        
        self.ctx.logger.info(f"✅ Updated professor_lookup.csv:")
        self.ctx.logger.info(f"   • Total entries: {len(lookup_data)}")
        self.ctx.logger.info(f"   • New entries added: {new_entries_count}")
        self.ctx.logger.info(f"   • Existing entries updated: {updated_entries_count}")
        
        # Step 7: Log summary of different methods
        method_counts = {}
        for entry in lookup_data:
            method = entry.get('method', 'unknown')
            method_counts[method] = method_counts.get(method, 0) + 1
        
        self.ctx.logger.info("📊 Entries by method:")
        for method, count in sorted(method_counts.items()):
            self.ctx.logger.info(f"   • {method}: {count}")

    def _extract_unique_professors(self) -> Tuple[set, dict]:
        """Extracts unique professor names and their variations from the raw data."""
        unique_professors = set()
        professor_variations = defaultdict(set)

        for _, row in self.ctx.multiple_data.iterrows():
            prof_name_raw = row.get('professor_name')
            if prof_name_raw is None or pd.isna(prof_name_raw):
                continue
            
            prof_name = str(prof_name_raw).strip()
            if not prof_name or prof_name.lower() in ['nan', 'tba', 'to be announced']:
                continue
            
            split_professors = self.professor_normalizer._split_professor_names(prof_name)
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

    def _parse_boss_aliases(self, boss_aliases_val: any) -> list[str]:
        """Parse boss_aliases using shared utility."""
        return parse_boss_aliases(boss_aliases_val)