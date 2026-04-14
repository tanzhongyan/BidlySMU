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
│   ├── config.py                 # Central configuration
│   ├── base/                     # Base classes
│   │   └── base_scraper.py       # Abstract BaseScraper with Smart Wait
│   ├── driver/                   # WebDriver management
│   │   ├── driver_factory.py     # ChromeDriverFactory
│   │   └── authenticator.py     # Authenticator interface + ManualLogin/AutomatedLogin
│   ├── scraper/                  # Data collection (Step 1)
│   │   ├── class_scraper.py      # Class information scraper
│   │   ├── html_data_extractor.py # HTML data extractor
│   │   ├── overall_results_scraper.py # Overall results scraper
│   │   ├── coordinator.py        # ScraperCoordinator (wires all components)
│   │   └── deprecated/V4/        # Deprecated scraper implementations
│   ├── pipeline/                 # Processing & prediction (Step 2-3)
│   │   ├── step_2_TableBuilder.py
│   │   └── step_3_BidPrediction.py
│   ├── logging/                  # Logging utilities
│   │   └── logger.py             # LoggerFactory with Sentry integration
│   ├── models/dto/               # Data Transfer Objects
│   │   └── scraping_result.py    # ScrapingResult, ScraperError, ErrorType
│   ├── parser/                   # Data parsing utilities
│   │   ├── excel_writer.py
│   │   └── encoding_handler.py
│   └── utils/                    # Shared utilities
│       ├── term_resolver.py      # Term code mapping utilities
│       └── schedule_resolver.py  # Bidding round schedule utilities
├── scripts/
│   └── run_pipeline.sh           # Pipeline orchestrator
├── tests/                        # Unit tests
│   ├── conftest.py               # Shared pytest fixtures
│   └── unit/                     # Unit tests (116 tests)
├── data/                         # Reference data
│   ├── professor_lookup.csv
│   └── safety_factor_table.csv
├── models/                       # Trained CatBoost models (.cbm)
├── notebooks/                    # Jupyter notebooks
├── assets/                       # Images and static assets
├── docs/                         # Documentation
├── script_input/                 # Input data (raw data, BOSS results)
├── script_output/                # Output data (predictions, logs)
├── deprecated/V4/                 # Old V4 scraper files (deprecated)
├── db_cache/                     # Database cache
├── logs/                         # Pipeline logs
├── requirements.txt
└── README.md
```

### **Key Files (OOP Refactored)**
| File | Description |
|------|-------------|
| `scripts/run_pipeline.sh` | Orchestrates the full pipeline |
| `src/config.py` | Central configuration (terms, rounds, schedules) |
| `src/scraper/coordinator.py` | Wires DriverFactory + Authenticator + Scraper |
| `src/scraper/class_scraper.py` | Scrapes class information from BOSS |
| `src/scraper/html_data_extractor.py` | Extracts data from saved HTML files |
| `src/scraper/overall_results_scraper.py` | Scrapes overall bidding results |
| `src/base/base_scraper.py` | Abstract base with Smart Wait wrapper |
| `src/driver/driver_factory.py` | ChromeDriver factory with headless support |
| `src/driver/authenticator.py` | ManualLogin or AutomatedLogin (TOTP MFA) |
| `src/logging/logger.py` | LoggerFactory with Sentry integration |
| `src/models/dto/scraping_result.py` | ScrapingResult, ScraperError DTOs |
| `tests/unit/` | 116 unit tests with pytest |

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

## **7. AI-Powered Professor Name Normalization**

The pipeline uses **Google Gemini 2.5 Flash** (free tier) to normalize professor names from various formats into standardized surnames for database matching.

### **Setup**

1. **Get a Gemini API Key** (free):
   - Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
   - Create a new API key
   - The free tier includes 1,500 requests/day for Gemini 2.5 Flash

2. **Configure environment variable** in `.env`:
   ```bash
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

3. **Install dependencies** (already in requirements.txt):
   ```bash
   pip install google-genai
   ```

### **How It Works**

- The `src/pipeline/step_2_TableBuilder.py` script uses Gemini LLM to identify surnames from various name formats
- Example: `"Dr. John Smith Jr."` → `"Smith"`
- Processes names in batches of 50 for efficiency
- Falls back to rule-based normalization if LLM is unavailable

### **Configuration**

| Setting | Default | Description |
|---------|---------|-------------|
| `llm_model_name` | `gemini-2.5-flash` | Gemini model version |
| `llm_batch_size` | `50` | Names processed per API call |

---

## **8. Crediting the Author**

If you use this project or its models in your work, please credit **Tan Zhong Yan** in the following way on GitHub:

```markdown
Project by [Tan Zhong Yan](https://github.com/tanzhongyan).  
LinkedIn: [Zhong Yan Tan](https://www.linkedin.com/in/zhong-yan-tan/)