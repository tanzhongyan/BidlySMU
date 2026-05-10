# BidlySMU Pipeline Documentation

## Overview

BidlySMU is an OOP data pipeline for Singapore Management University (SMU) course bidding prediction. The pipeline follows an ETL (Extract, Transform, Load) pattern with machine learning integration for predicting minimum and median bid prices for university courses.

A central `PipelineCoordinator` orchestrates 12+ specialized processors across three phases — dimension tables, fact tables, and predictions — with results persisted to both CSV and PostgreSQL.

**Two separate workflows:**
1. **Production Pipeline** (`scripts/run_pipeline.sh`): Data collection → Processing → Prediction using pre-trained models
2. **Model Training** (`V4_03_catboost_training.ipynb`): Trains the three CatBoost models from historical data (separate process)

## Architecture

### Directory Structure

```
src/
├── config.py                          # PipelineConfig, bidding window parsing, schedules
├── requirements.txt
├── driver/
│   ├── driver_factory.py              # Chrome WebDriver creation
│   └── authenticator.py               # BOSS automated login (TOTP)
├── db/
│   ├── adapters.py                    # Psycopg2Adapter (PostgreSQL connection)
│   └── database_helper.py             # Bulk INSERT/UPDATE, cache download
├── logging/
│   └── logger.py                      # Structured logging setup
├── pipeline/
│   ├── pipeline_coordinator.py        # Central orchestrator
│   ├── transformer.py                 # SMUBiddingTransformer (CatBoost features)
│   ├── safety_factor_calculator.py    # T-distribution percentile multipliers
│   ├── dtos/                          # Data Transfer Objects
│   │   ├── acad_term_dto.py
│   │   ├── bid_prediction_dto.py      # + SafetyFactorDTO
│   │   ├── bid_result_dto.py
│   │   ├── bid_window_dto.py
│   │   ├── class_availability_dto.py
│   │   ├── class_dto.py
│   │   ├── course_dto.py
│   │   ├── professor_dto.py
│   │   └── timing_dto.py
│   └── processors/
│       ├── abstract_processor.py       # Base class (Template Method pattern)
│       ├── acad_term_processor.py
│       ├── bid_prediction_processor.py
│       ├── bid_result_processor.py
│       ├── bid_window_processor.py
│       ├── class_availability_processor.py
│       ├── class_exam_timing_processor.py
│       ├── class_processor.py
│       ├── class_timing_processor.py
│       ├── course_processor.py
│       ├── professor_processor.py
│       ├── professor_resolution_service.py
│       └── safety_factor_processor.py
├── scraper/
│   ├── abstract_scraper.py            # Base scraper class
│   ├── class_scraper.py               # BOSS class detail scraping
│   ├── html_data_extractor.py         # HTML → Excel extraction
│   ├── overall_results_scraper.py     # BOSS bid results scraping
│   └── scraper_coordinator.py         # Scraper orchestration
scripts/
└── run_pipeline.sh                    # Pipeline entry point
```

### Component Descriptions

#### Data Acquisition Layer (`src/scraper/`)

Scrapes raw data from SMU's BOSS (Bidding Online System for Students) using Selenium. Disabled by default in `run_pipeline.sh` — typically run separately before the pipeline.

| Component | File | Responsibility |
|-----------|------|----------------|
| **Class Scraper** | `class_scraper.py` | Scrapes class detail pages (class numbers 1000-5000) per academic term. Stops after 300 consecutive empty records. Saves HTML files to structured directory hierarchy. |
| **HTML Data Extractor** | `html_data_extractor.py` | Parses scraped HTML files using Selenium DOM parsing. Handles encoding issues (UTF-8 mojibake fixes). Outputs `script_input/raw_data.xlsx` with standalone + multiple sheets. |
| **Overall Results Scraper** | `overall_results_scraper.py` | Scrapes historical bidding results from BOSS OverallResults page. Handles pagination (up to 200 pages), Incoming Freshmen/Exchange window detection, and data deduplication. |
| **Scraper Coordinator** | `scraper_coordinator.py` | Orchestrates scraper execution and login flow. |

#### Pipeline Processing Layer (`src/pipeline/`)

The core data processing engine. All processors inherit from `AbstractProcessor` which enforces a standard `process()` interface and provides `safe_int()`/`safe_float()` type conversion utilities.

**Phase 1 — Dimension Tables:**

| Processor | File | Input | Output | Key Logic |
|-----------|------|-------|--------|-----------|
| **AcadTermProcessor** | `acad_term_processor.py` | standalone sheet | `AcadTermDTO` (new/updated) | Extracts term codes from text (T1, T2, T3A, T3B), generates `AY{year}{suffix}{term}` IDs |
| **CourseProcessor** | `course_processor.py` | standalone sheet, courses/faculties cache | `CourseDTO` (new/updated) | Matches courses to faculties using prefix frequency analysis |
| **ProfessorProcessor** | `professor_processor.py` | multiple sheet, professors cache | `ProfessorDTO` (new/updated) | Resolves professor names via `ProfessorResolutionService`; merges `professor_lookup.csv` as supplementary source |
| **BidWindowProcessor** | `bid_window_processor.py` | standalone sheet, bid_window cache | `BidWindowDTO` (new/updated) | Parses bidding window text ("Round 1 Window 1", "Incoming Freshmen Rnd 1 Win 4") using centralized `parse_bidding_window()` from `config.py` |

**Phase 2 — Fact Tables:**

| Processor | File | Input | Output | Key Logic |
|-----------|------|-------|--------|-----------|
| **ClassProcessor** | `class_processor.py` | standalone sheet, multiple_lookup, course_lookup, existing classes cache | `ClassDTO` (new/updated) | Group-based reconciliation at `(acad_term_id, boss_id)` level. Handles professor transitions (0→1, 1→N, swaps). Soft-deactivates excess records. Creates one class record per professor for multi-professor courses. |
| **ClassTimingProcessor** | `class_timing_processor.py` | multiple sheet, class_lookup, record_key_to_class_ids | `ClassTimingDTO` | Deduplicates against existing timing keys `(class_id, day_of_week, start_time, end_time, venue)` |
| **ClassExamTimingProcessor** | `class_exam_timing_processor.py` | multiple sheet, class_lookup, record_key_to_class_ids | `ClassExamTimingDTO` | Skips classes that already have exam timings in cache |
| **ClassAvailabilityProcessor** | `class_availability_processor.py` | standalone sheet, class/bid_window lookups | `ClassAvailabilityDTO` | Filters to current bidding window only |
| **BidResultProcessor** | `bid_result_processor.py` | standalone sheet, overall results Excel, class/bid_window/course lookups | `BidResultDTO` (new/updated) | Imports bid results from both raw_data.xlsx and scraped overallBossResults |

**Phase 3 — Predictions:**

| Processor | File | Input | Output | Key Logic |
|-----------|------|-------|--------|-----------|
| **BidPredictionProcessor** | `bid_prediction_processor.py` | standalone sheet, class/bid_window lookups, multiple_lookup | `BidPredictionDTO` | Three-model CatBoost inference (classification + median regression + min regression). Entropy-based confidence scoring. Tree-subset uncertainty quantification. |
| **SafetyFactorProcessor** | `safety_factor_processor.py` | validation results CSVs, acad_term_id | `SafetyFactorDTO` | T-distribution fitting on regression residuals. Generates empirical + theoretical multipliers for percentiles 1-99 (396 total entries per term). |

#### Data Transfer Objects (`src/pipeline/dtos/`)

All DTOs use Python `@dataclass` with a consistent serialization pattern:

```python
@dataclass
class ExampleDTO:
    COLUMNS = {'snake_case': 'snake_case'}  # CSV/DB column mapping

    def to_csv_row(self) -> dict: ...   # ISO-format dates, None-safe
    def to_db_row(self) -> dict: ...    # Native datetime, None-safe
    @staticmethod
    def from_row(...) -> 'ExampleDTO': ...  # Factory for CREATE
```

This dual-serialization ensures CSV output (human-readable ISO dates) and DB output (native Python types) are handled consistently across all 10 DTOs.

#### Database Layer (`src/db/`)

| Component | File | Responsibility |
|-----------|------|----------------|
| **Psycopg2Adapter** | `adapters.py` | PostgreSQL connection creation from config dict |
| **DatabaseHelper** | `database_helper.py` | `create_connection()`, `insert_df()` (bulk INSERT via `execute_batch`), `update_df()` (bulk UPDATE via `execute_batch`), `download_cache()` (DB tables → pickle files) |

#### Configuration (`src/config.py`)

| Component | Description |
|-----------|-------------|
| **`PipelineConfig`** | Dataclass holding: `bidding_schedules`, `start_ay_term`, `db_config`, `input_file`, `output_base`, `verify_dir`, `cache_dir`. Constructed via `from_env()`. |
| **`parse_bidding_window()`** | Centralized parser for bidding window text → `(round, window)` tuple. Supports: "Round 1A Window 2", "Incoming Freshmen Rnd 1 Win 4", "Incoming Exchange Rnd 1C Win 1", abbreviated "Rnd 1A Win 2", and generic fallback patterns. |
| **`BIDDING_SCHEDULES`** | Loaded from `script_input/bidding_schedules.json`. Maps term IDs to `[(datetime, window_name, folder_suffix)]` tuples. |
| **`CURRENT_WINDOW_NAME`** | Auto-computed at module load time from schedules and current time. |

## Data Flow

### Phase 1: Data Collection (Parallel, disabled by default)
```
Stream A: BOSS Website → class_scraper → HTML Files → html_data_extractor → raw_data.xlsx
Stream B: BOSS Website → overall_results_scraper → overallBossResults/{term}.xlsx
```

### Phase 2: Pipeline Processing (Sequential via PipelineCoordinator)
```
raw_data.xlsx + db_cache/*.pkl + config
         ↓
  PipelineCoordinator
    ├─ Phase 1 (Dimensions):
    │   AcadTermProcessor → CourseProcessor → ProfessorProcessor → BidWindowProcessor
    │       ↓ (builds lookups for downstream)
    ├─ Phase 2 (Facts):
    │   ClassProcessor → ClassTimingProcessor → ClassExamTimingProcessor
    │   → ClassAvailabilityProcessor → BidResultProcessor
    │       ↓ (class_lookup available for predictions)
    └─ Phase 3 (Predictions):
        BidPredictionProcessor → SafetyFactorProcessor
         ↓
  CSV output (script_output/) + PostgreSQL
```

### Cross-Processor Data Passing

The `PipelineCoordinator` builds lookup dictionaries from upstream processor results and injects them into downstream processors:

| Lookup | Key | Value | Used By |
|--------|-----|-------|---------|
| `course_lookup` | `course_code` | `CourseDTO` | ClassProcessor |
| `professor_resolution_service` | (service) | (resolved IDs) | ClassProcessor |
| `bid_window_lookup` | `(acad_term_id, round, window)` | `BidWindowDTO` | ClassAvailabilityProcessor, BidResultProcessor, BidPredictionProcessor |
| `class_lookup` | `(acad_term_id, boss_id, professor_id)` | `ClassDTO` | ClassTimingProcessor, ClassExamTimingProcessor, ClassAvailabilityProcessor, BidResultProcessor, BidPredictionProcessor |
| `record_key_to_class_ids` | `record_key` | `[class_ids]` | ClassTimingProcessor, ClassExamTimingProcessor |

## Key Design Patterns

### 1. Template Method (AbstractProcessor)
All processors inherit from `AbstractProcessor` which enforces a `process()` method and provides shared utilities (`safe_int`, `safe_float`). This ensures a consistent interface across all 12 processors.

### 2. DTO Serialization
Every data entity has a dedicated DTO with `to_csv_row()` and `to_db_row()` methods. This decouples the pipeline's internal data representation from output format concerns and ensures consistent handling of None values, date formatting, and type conversion.

### 3. Professor Resolution Service
A dedicated `ProfessorResolutionService` serves as the single source of truth for mapping scraped professor names to database IDs. It uses a 7-strategy resolution chain:

| Strategy | Description |
|----------|-------------|
| 1. Direct lookup | Exact match in `boss_name_upper → professor_id` map |
| 2. Variation lookup | Name variations (remove commas, normalize spaces, reorder "LAST, FIRST" ↔ "FIRST LAST") |
| 3. Full lookup | Match against `database_id` in full professor records |
| 4. Boss alias lookup | Match via `boss_aliases` field (alternate scraped name forms) |
| 5. Subset matching | Partial word match (e.g., "JOHN DOE" matches "JOHN DOE SMITH") — requires ≥2 words |
| 6. New professors | Session-created professors not yet in DB cache |
| 7. No match | Returns None — professor will be created as new record |

**Design decision — no LLM surname extraction**: V4 used Gemini 2.5 Flash to identify primary surnames for ambiguous names. The current service relies on DB-based lookups, boss_aliases, and name variations instead. Trade-off: simpler dependency (no `GEMINI_API_KEY` needed) vs. potential duplicate new professors for ambiguous names. The `professor_lookup.csv` file serves as a human-curated fallback for edge cases.

### 4. Group-Based Class Reconciliation
`ClassProcessor` reconciles existing and incoming class state at the `(acad_term_id, boss_id)` group level rather than per-record. This correctly handles professor transitions:

- **0→1 professors**: TBA class gets professor assigned
- **1→0 professors**: Class gets `professor_id=None` (soft deactivation, not deletion)
- **1→N professors**: Additional class records created, one per professor
- **Professor swaps**: Existing class records repurposed with new professor IDs

Excess existing classes (more old records than incoming professors) are marked for soft deactivation rather than hard deletion.

### 5. Cache-First Design
`PipelineCoordinator._load_caches()` loads from `db_cache/*.pkl` first. Only on cache miss does it download from PostgreSQL. This minimizes DB load across pipeline runs. Caches are stored as pickle-serialized DataFrames and converted to dict-of-dict format for processor compatibility.

## Pipeline vs Model Training

### Production Pipeline (`scripts/run_pipeline.sh`)
- **Frequency**: Runs for each bidding window (multiple times per term)
- **Input**: Live BOSS system data (`script_input/raw_data.xlsx`)
- **Output**: Predictions for current courses
- **Models**: Loads pre-trained `.cbm` files
- **Duration**: ~15-30 minutes depending on data volume

### Model Training Workflow (`V4_03_catboost_training.ipynb`)
- **Frequency**: Periodic (once per term or when model performance degrades)
- **Input**: Historical bidding data from database
- **Output**: Three trained model files + validation results
- **Process**: Train three CatBoost models → generate validation metrics → calculate safety factors → save to `models/`

**Critical**: The pipeline does NOT train models. It always loads the latest production models.

## ML Architecture

### Three-Model Approach
1. **Classification Model** (`production_classification_model.cbm`): Predicts probability that a course will receive bids
2. **Median Bid Regression** (`production_regression_median_model.cbm`): Predicts median bid price
3. **Min Bid Regression** (`production_regression_min_model.cbm`): Predicts minimum successful bid

### Feature Engineering (`SMUBiddingTransformer`)
| Feature | Type | Source |
|---------|------|--------|
| `subject_area` | Categorical | Course code prefix (e.g., "FNCE" from "FNCE318") |
| `catalogue_no` | Categorical | Course code number (e.g., 318) |
| `round` | Categorical | Bidding round (1, 1A, 1F, 2, etc.) |
| `window` | Numeric | Bidding window number |
| `before_process_vacancy` | Numeric | Total seats minus enrolled students |
| `acad_year_start` | Numeric | Academic year start (e.g., 2025) |
| `term` | Categorical | Term code (1, 2, 3A, 3B) |
| `start_time` | Categorical | Class start time |
| `course_name` | Categorical | Course name |
| `section` | Categorical | Section identifier |
| `instructor` | Categorical | JSON array of professor names (via professor_lookup mapping) |
| `has_mon`..`has_sun` | Numeric (binary) | Day-of-week one-hot encoding |

### Uncertainty Quantification
- **Classification confidence**: Entropy-based confidence score (1 − normalized entropy of prediction probabilities)
- **Regression uncertainty**: Virtual ensemble method — 10 tree subsets from each CatBoost model, standard deviation of subset predictions
- **Safety factors**: T-distribution fitting on validation residuals. Empirical + theoretical multipliers for percentiles 1-99. 396 entries per academic term (99 percentiles × 2 multiplier types × 2 prediction types).

## Technical Stack

### Core Dependencies
- **Python 3.x**: Primary programming language
- **Selenium 4.31.0**: Web scraping and automation
- **CatBoost 1.2.8**: Gradient boosting for tabular data
- **Pandas 2.2.3**: Data manipulation and analysis
- **Psycopg2**: PostgreSQL adapter (bulk operations via `execute_batch`)
- **PostgreSQL**: Primary database storage

### Supporting Libraries
- **NumPy**: Numerical computations
- **SciPy**: T-distribution fitting for safety factors
- **python-dotenv**: Environment variable loading
- **openpyxl**: Excel file I/O

## Usage Guidelines

### Pipeline Execution
```bash
# Run complete pipeline
./scripts/run_pipeline.sh
```

The pipeline entry point loads `PipelineConfig.from_env()`, constructs a `PipelineCoordinator`, and calls `coordinator.run()` which executes all 12 processors sequentially across 3 phases.

### Configuration
1. Set `ACAD_TERM_ID` in `.env` (e.g., `AY202526T3A`)
2. Set database credentials in `.env` (`DB_HOST`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `DB_PORT`)
3. Update `script_input/bidding_schedules.json` with current term's schedule
4. Place `script_input/raw_data.xlsx` (from scraper) before running
5. Ensure trained models exist in `models/`

### Model Usage
```python
from catboost import CatBoostClassifier, CatBoostRegressor

clf = CatBoostClassifier().load_model('models/production_classification_model.cbm')
median = CatBoostRegressor().load_model('models/production_regression_median_model.cbm')
min_model = CatBoostRegressor().load_model('models/production_regression_min_model.cbm')
```