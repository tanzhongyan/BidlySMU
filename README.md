# **BidlySMU**

**Bidly** is a predictive tool designed to help SMU students navigate the module bidding system with confidence. Leveraging advanced machine learning models, Bidly predicts minimum and median bid prices with sophisticated uncertainty quantification, ensuring students can secure their desired modules with ease. Whether you're trying to optimize your e-credits or plan your bidding strategy, Bidly empowers you with data-driven insights and confidence-calibrated predictions.

---

<div style="background-color:#DFFFD6; padding:12px; border-radius:5px; border: 1px solid #228B22;">
  <h2 style="color:#006400;">✅ Looking to Implement This? ✅</h2>
  <p>🚀 <strong>Use the latest V4 models with <a href="example_prediction.ipynb">example_prediction.ipynb</a>!</strong></p>
  <ul>
    <li>📌 **Three pre-trained CatBoost models (`.cbm`) included for instant predictions.**</li>
    <li>🔧 **Step-by-step instructions with uncertainty quantification available.**</li>
    <li>⚡ **No re-training required—just load and predict with confidence intervals!**</li>
    <li>🎯 **Advanced safety factor system with percentile-based risk assessment.**</li>
  </ul>
  <p>👉 <a href="example_prediction.ipynb"><strong>Go to Example Prediction Notebook</strong></a></p>
</div>

<br>

<div style="background-color:#FFD700; padding:15px; border-radius:5px; border: 2px solid #FF4500;">
    
  <h2 style="color:#8B0000;">⚠️🚨 SCRAPE THIS DATA AT YOUR OWN RISK 🚨⚠️</h2>
  
  <p><strong>📌 If you need the data, please contact me directly.</strong> Only available for **existing students**.</p>

  <h3>🔗 📩 How to Get the Data?</h3>
  <p>📨 <strong>Reach out to me for access</strong> instead of scraping manually.</p>

</div>

---

## **2. Key Differences Between Versions**
- **V4 (Recommended)**: Revolutionary three-model architecture with **classification + median + min bid prediction models**, advanced uncertainty quantification using t-distribution fitting, entropy-based confidence scoring, and sophisticated percentile-based safety factors.
- **V3 (Deprecated)**: Introduced bug fixes, two **pre-trained `.cbm` models** for **minimum and median bid predictions** without re-training.
- **V2 (Deprecated)**: Enhances V1 by incorporating scraped data from the BOSS website, including class timings, for improved predictions.
- **V1 (Deprecated)**: Utilizes readily available data from the OASIS BOSS bidding results.

---

## **3. Project Overview**

### **V4 (Recommended)**
- **Purpose**: Provides a comprehensive three-model system with advanced uncertainty quantification for production-ready bid predictions.
- **Key Features**:
  1. **Three-model architecture**: Classification (bid opportunity detection) + Median + Min regression models
  2. **Advanced uncertainty quantification**: T-distribution-based confidence intervals and entropy-based confidence scoring
  3. **Sophisticated safety factor system**: Percentile-based multipliers (1%-99%) derived from validation error analysis
  4. **Enhanced feature engineering**: Day-of-week boolean flags and optimized categorical encoding
  5. **Production-ready deployment**: All models saved as `.cbm` files with consistent interfaces
  6. **Risk-aware predictions**: Distribution-based uncertainty modeling prevents dangerous under-bidding
  7. **Scientific rigor**: Replaces ad-hoc safety factors with statistically grounded uncertainty measures

### **V3** (Deprecated)
- **Purpose**: Provides a pre-trained `.cbm` model for instant predictions without requiring dataset reprocessing.
- **Key Features**:
  1. Directly **loads pre-trained models** from the repository.
  2. No need for additional feature engineering.
  3. Works out-of-the-box with structured course data.

### **V2** (Deprecated)
- **Purpose**: Improve prediction accuracy using additional features like class timings.
- **Key Features**:
  1. Scraped data enhanced the dataset significantly.
  2. Feature selection was performed to manage diminishing returns.

### **V1** (Deprecated)
- **Purpose**: Evaluate and compare different machine learning models (CatBoost and DNN) using OASIS data.
- **Key Features**:
  1. CatBoost models explored default, tuned, and cleaned versions.
  2. Neural networks were tested but found less suitable for this tabular data.

---

## **4. Key Findings**

### **V4 Breakthrough Improvements**
1. **Multi-model approach**: Separate classification and regression models significantly improve prediction accuracy and reliability
2. **Scientific uncertainty quantification**: T-distribution fitting to validation errors provides statistically grounded confidence intervals
3. **Recall-optimized classification**: Maximizes detection of courses that will receive bids, minimizing missed opportunities
4. **Distribution-aware safety factors**: Percentile-based multipliers (e.g., 90% confidence) replace arbitrary percentage increases
5. **Enhanced feature engineering**: Day-of-week patterns and improved instructor handling boost model performance
6. **Production deployment ready**: Comprehensive model persistence and consistent interfaces for real-world usage

### **V3 Improvements** (Deprecated)
1. No re-training required—models are pre-trained and available as `.cbm` files.
2. Fixed bugs such as dependency on future values and outlier removal.

### **V2 Enhancements** (Deprecated)
1. Adding class timings significantly reduces prediction errors.
2. Additional features lead to diminishing returns beyond a certain point.
3. Adding safety factors or bootstrapping confidence intervals could improve predictions but requires more computation.

### **V1 Summary** (Deprecated)
1. CatBoost models perform better for tabular data with categorical variables.
2. Default CatBoost is nearly as effective as tuned models.
3. Removing outliers (e.g., troll bids) has limited impact on performance.
4. Latest academic year data provides the most predictive power.
5. Combining data from all years yields the best results.

---

## **5. Project Structure**

```
BidlySMU/
├── src/                          # Python pipeline source code
│   ├── config.py                 # PipelineConfig, parse_bidding_window, bidding schedules
│   ├── requirements.txt
│   ├── driver/                   # WebDriver management
│   │   ├── driver_factory.py     # Chrome WebDriver creation
│   │   └── authenticator.py      # BOSS automated login (TOTP)
│   ├── db/                       # Database layer
│   │   ├── adapters.py           # Psycopg2Adapter (PostgreSQL connection)
│   │   └── database_helper.py    # Bulk INSERT/UPDATE, cache download
│   ├── logging/
│   │   └── logger.py             # Structured logging setup
│   ├── pipeline/                 # Core processing engine
│   │   ├── pipeline_coordinator.py  # Central orchestrator (3-phase execution)
│   │   ├── transformer.py           # SMUBiddingTransformer (CatBoost features)
│   │   ├── safety_factor_calculator.py  # T-distribution percentile multipliers
│   │   ├── dtos/                    # Data Transfer Objects
│   │   │   ├── acad_term_dto.py
│   │   │   ├── bid_prediction_dto.py   # + SafetyFactorDTO
│   │   │   ├── bid_result_dto.py
│   │   │   ├── bid_window_dto.py
│   │   │   ├── class_availability_dto.py
│   │   │   ├── class_dto.py
│   │   │   ├── course_dto.py
│   │   │   ├── professor_dto.py
│   │   │   └── timing_dto.py
│   │   └── processors/              # 12 specialized processors
│   │       ├── abstract_processor.py       # Base class (Template Method)
│   │       ├── acad_term_processor.py
│   │       ├── bid_prediction_processor.py
│   │       ├── bid_result_processor.py
│   │       ├── bid_window_processor.py
│   │       ├── class_availability_processor.py
│   │       ├── class_exam_timing_processor.py
│   │       ├── class_processor.py          # Group reconciliation, soft deactivation
│   │       ├── class_timing_processor.py
│   │       ├── course_processor.py
│   │       ├── professor_processor.py
│   │       ├── professor_resolution_service.py  # 7-strategy name resolution
│   │       └── safety_factor_processor.py
│   └── scraper/                  # Data collection (disabled by default)
│       ├── abstract_scraper.py    # Base scraper class
│       ├── class_scraper.py       # BOSS class detail scraping
│       ├── html_data_extractor.py # HTML → Excel extraction
│       ├── overall_results_scraper.py  # BOSS bid results scraping
│       └── scraper_coordinator.py # Scraper orchestration
├── scripts/
│   └── run_pipeline.sh           # Pipeline entry point
├── models/                       # Trained CatBoost models (.cbm)
├── notebooks/                    # Jupyter notebooks
├── assets/                       # Images and static assets
├── docs/                         # Documentation
├── script_input/                 # Input data (raw_data.xlsx, BOSS results, schedules)
├── script_output/                # Output data (CSV predictions, verify/)
├── deprecated/V4/                # Old V4 procedural scripts (deprecated)
├── db_cache/                     # Pickle-serialized DB table cache
├── logs/                         # Pipeline logs
└── README.md
```

### **Key Components**

| Component | File | Description |
|-----------|------|-------------|
| **Pipeline Entry Point** | `scripts/run_pipeline.sh` | Loads config, runs PipelineCoordinator |
| **Configuration** | `src/config.py` | `PipelineConfig.from_env()`, `parse_bidding_window()`, bidding schedules from JSON |
| **Orchestrator** | `src/pipeline/pipeline_coordinator.py` | 3-phase execution, cross-processor lookups, dual output (CSV + DB) |
| **Class Processing** | `src/pipeline/processors/class_processor.py` | Group-based reconciliation at `(acad_term_id, boss_id)` level, professor transitions, soft deactivation |
| **Professor Resolution** | `src/pipeline/processors/professor_resolution_service.py` | 7-strategy name resolution chain (replaces V4's LLM approach) |
| **Predictions** | `src/pipeline/processors/bid_prediction_processor.py` | Three-model CatBoost inference, entropy confidence, tree-subset uncertainty |
| **Feature Engineering** | `src/pipeline/transformer.py` | `SMUBiddingTransformer` — course code decomposition, day-of-week encoding, instructor mapping |
| **Safety Factors** | `src/pipeline/safety_factor_calculator.py` | T-distribution fitting on residuals, percentile multipliers 1-99 |
| **Database Helper** | `src/db/database_helper.py` | Bulk INSERT/UPDATE via `execute_batch`, cache download from PostgreSQL |
| **Data Collection** | `src/scraper/` | BOSS scraping (class details, HTML extraction, overall results) |

---

## **6. V4 Model Architecture**

### **Three-Model System**
| **Model Type** | **Purpose** | **Output** | **Uncertainty Measure** |
|----------------|-------------|------------|--------------------------|
| **Classification** | Predict bid courses | Probability + Confidence Level | Entropy-based confidence score |
| **Median Bid Regression** | Predict median bid price | Price + Uncertainty Multipliers | T-distribution-based multipliers |
| **Min Bid Regression** | Predict minimum bid price | Price + Uncertainty Multipliers | T-distribution-based multipliers |

### **Advanced Safety Factor Formula**
**Traditional approach**: `bid = prediction × 1.7` (arbitrary multiplier)  
**V4 approach**: `bid = prediction + uncertainty × multiplier` (distribution-based)

**Example (90% confidence)**:
- Median bid = `median_predicted + median_uncertainty × 1.584`
- Min bid = `min_predicted + min_uncertainty × 1.533`

---

## **6. Automated Login (For Cloud Deployment)**

The scraper now supports fully automated login with TOTP-based MFA, enabling deployment on cloud containers with cron jobs.

### **Setup**

1. **Install dependencies**:
   ```bash
   pip install pyotp
   ```

2. **Configure environment variables** in `.env`:
   ```bash
   BOSS_EMAIL=your.email.2023@business.smu.edu.sg
   BOSS_PASSWORD=your_password_here
   BOSS_MFA_SECRET=your_mfa_secret_key
   ```

3. **Get your MFA Secret**:
   - Open Microsoft Authenticator app
   - Select your SMU account
   - Tap "Set up" → "Can't scan?"
   - Copy the secret key (Base32 encoded string)

### **Usage**

```python
from src.scraper.step_1c_ScrapeOverallResults import ScrapeOverallResults

# Automated login
scraper = ScrapeOverallResults()
success = scraper.run(
    term='2025-26_T1',
    automated_login=True  # Enable automated login
)
```

Or for the class scraper:

```python
from src.scraper.step_1a_BOSSClassScraper import BOSSClassScraper

scraper = BOSSClassScraper()
success = scraper.run_full_scraping_process(
    automated_login=True  # Enable automated login
)
```

### **Security Notes**
- Store credentials securely using environment variables
- Never commit `.env` files with real credentials
- The MFA secret is as sensitive as your password - keep it secure

---

## **7. Professor Name Resolution**

### **V4: LLM-Based Surname Extraction (Deprecated)**

V4 used **Google Gemini 2.5 Flash** to identify the primary surname from professor names for database matching. The LLM received a JSON list of names and returned a corresponding list of surnames:

- **Prompt**: *"You are an expert in academic name structures from around the world. You will be given a JSON list of professor names. Your task is to identify the primary surname for each name."*
- **Batch size**: 50 names per API call
- **Example**: `"Dr. John Smith Jr."` → `"Smith"`, `"ZHANG WEI"` → `"ZHANG"`
- The extracted surname was then used to match against existing professors in the database
- Required `GEMINI_API_KEY` environment variable and `google-genai` dependency
- Fell back to rule-based normalization (Asian surname databases + Western given names) if LLM was unavailable

### **V5: DB-Based 7-Strategy Resolution Chain (Current)**

V5 replaces the LLM approach entirely with `ProfessorResolutionService` — a deterministic, DB-based resolution chain with no external API dependencies:

| Strategy | Description |
|----------|--------------|
| 1. Direct lookup | Exact match in `boss_name_upper → professor_id` map |
| 2. Variation lookup | Name variations (remove commas, normalize spaces, reorder "LAST, FIRST" ↔ "FIRST LAST") |
| 3. Full lookup | Match against `database_id` in full professor records |
| 4. Boss alias lookup | Match via `boss_aliases` field (alternate scraped name forms) |
| 5. Subset matching | Partial word match (e.g., "JOHN DOE" matches "JOHN DOE SMITH") — requires ≥2 words |
| 6. New professors | Session-created professors not yet in DB cache |
| 7. No match | Returns None — professor will be created as new record |

**Trade-off**: Simpler dependencies (no `GEMINI_API_KEY` or `google-genai` needed) vs. potential duplicate new professors for ambiguous names (e.g., common names like "Rachel Tan"). The `professor_lookup.csv` file serves as a human-curated fallback for edge cases.

---

## **8. Crediting the Author**

If you use this project or its models in your work, please credit **Tan Zhong Yan** in the following way on GitHub:

```markdown
Project by [Tan Zhong Yan](https://github.com/tanzhongyan).  
LinkedIn: [Zhong Yan Tan](https://www.linkedin.com/in/zhong-yan-tan/)