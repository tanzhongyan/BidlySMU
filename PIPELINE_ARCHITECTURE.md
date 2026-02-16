# BidlySMU Pipeline Architecture Documentation

## Overview
BidlySMU is a comprehensive data pipeline for Singapore Management University (SMU) course bidding prediction system. The pipeline follows an ETL (Extract, Transform, Load) pattern with machine learning integration for predicting minimum and median bid prices for university courses.

## Architecture Components

### 1. Data Acquisition Layer (Step 1)
**Purpose**: Extract raw data from SMU's BOSS (Bidding Online System for Students) system.

#### 1a. BOSS Class Scraper (`step_1a_BOSSClassScraper.py`)
- **Function**: Scrapes class details from BOSS system using Selenium WebDriver
- **Key Features**:
  - Full scan of class numbers (1000-5000) for each academic term
  - Manual login handling with Microsoft Authenticator integration
  - Bidding schedule-aware scraping based on configurable timelines
  - HTML file storage in structured directory hierarchy
- **Dependencies**: Selenium, ChromeDriver, webdriver-manager

#### 1b. HTML Data Extractor (`step_1b_HTMLDataExtractor.py`)
- **Function**: Parses scraped HTML files and extracts structured data
- **Key Features**:
  - Selenium-based DOM parsing for reliable data extraction
  - Encoding handling for multilingual content
  - Data validation and cleaning
  - Excel file output generation
- **Output**: `script_input/raw_data.xlsx`

#### 1c. Scrape Overall Results (`step_1c_ScrapeOverallResults.py`)
- **Function**: Scrapes historical bidding results for model training
- **Key Features**:
  - Historical data collection across multiple academic years
  - Parallel execution with other scraping components
  - Error logging and retry mechanisms

### 2. Data Processing Layer (Step 2)
**Purpose**: Transform raw data into structured database format.

#### Table Builder (`step_2_TableBuilder.py`)
- **Function**: Processes extracted data and builds relational database tables
- **Key Features**:
  - Database connection management (PostgreSQL)
  - Entity resolution for professors, courses, and classes
  - Caching system for performance optimization
  - Data validation and integrity checks
  - Multiple output tables generation:
    - `professors`: Instructor information
    - `courses`: Course metadata
    - `classes`: Class instances
    - `class_timings`: Schedule information
    - `class_exam_timings`: Exam schedules
- **Dependencies**: SQLAlchemy, psycopg2, pandas

### 3. Machine Learning Layer (Step 3)
**Purpose**: Generate predictive models for bid price forecasting.

#### Bid Prediction (`step_3_BidPrediction.py`)
- **Function**: Trains and applies CatBoost models for bid prediction
- **Key Features**:
  - **Three-model architecture**:
    1. **Classification Model**: Predicts if a course will receive bids
    2. **Median Bid Regression**: Predicts median bid price
    3. **Minimum Bid Regression**: Predicts minimum successful bid
  - **Advanced feature engineering**:
    - Course code decomposition (school, level, number)
    - Bidding window feature extraction
    - Day-of-week one-hot encoding
    - Instructor categorical encoding
  - **Uncertainty quantification**:
    - T-distribution based confidence intervals
    - Entropy-based confidence scoring
    - Percentile-based safety factors (1%-99%)
  - **Model persistence**: `.cbm` (CatBoost) file format
- **Dependencies**: CatBoost, scikit-learn, numpy, pandas

### 4. Orchestration Layer
**Purpose**: Coordinate pipeline execution and error handling.

#### Pipeline Runner (`run_pipeline.sh`)
- **Function**: Bash script orchestrating the entire pipeline
- **Key Features**:
  - Parallel execution of Step 1 components
  - Sequential execution of Steps 2-3
  - Comprehensive logging with timestamps
  - Error handling and pipeline halting on failures
  - UTF-8 encoding enforcement

#### Configuration Management (`config.py`)
- **Function**: Centralized configuration for the entire pipeline
- **Key Features**:
  - Academic term range configuration
  - Bidding schedule definitions
  - Target round/window settings
  - Date-based folder naming conventions

## Data Flow Architecture

### Phase 1: Data Collection (Parallel)
```
BOSS Website (Selenium) → HTML Files → Excel Data
       ↑                        ↑
    [1a] Scraper           [1b] Extractor
       ↓                        ↓
    Class Details          Structured Data
       └───────────────────────┘
                    ↓
           [1c] Historical Results
                    ↓
            Raw Data Collection
```

### Phase 2: Data Processing (Sequential)
```
Raw Data → [2] Table Builder → Database Tables
                    ↓
           Structured Data Ready
                    ↓
         Feature Engineering Pipeline
```

### Phase 3: Model Training & Prediction (Sequential)
```
Structured Data → [3] Bid Prediction → Trained Models
                    ↓
           Prediction Generation
                    ↓
        Uncertainty Quantification
                    ↓
         Safety Factor Application
```

## Key Architectural Patterns

### 1. Modular Design
- Each component has a single responsibility
- Clear interfaces between components
- Independent testing capabilities

### 2. Error Resilience
- Comprehensive logging at each step
- Pipeline halting on critical failures
- Retry mechanisms for web scraping
- Data validation at transformation boundaries

### 3. Performance Optimization
- Parallel execution for independent tasks
- Database connection pooling
- Caching for entity resolution
- Batch processing for large datasets

### 4. Configuration Management
- Centralized configuration file
- Environment variable support
- Academic calendar awareness
- Schedule-based execution control

## Technical Stack

### Core Dependencies
- **Python 3.x**: Primary programming language
- **Selenium 4.31.0**: Web scraping and automation
- **CatBoost 1.2.8**: Gradient boosting for tabular data
- **Pandas 2.2.3**: Data manipulation and analysis
- **SQLAlchemy**: Database ORM and connection management
- **PostgreSQL**: Primary database storage

### Supporting Libraries
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning utilities
- **Matplotlib/Seaborn**: Data visualization
- **Webdriver-manager**: ChromeDriver management
- **Psycopg2**: PostgreSQL adapter

## Deployment Considerations

### Environment Requirements
1. **Chrome/Chromium Browser**: Required for Selenium automation
2. **ChromeDriver**: Automatically managed by webdriver-manager
3. **PostgreSQL Database**: For structured data storage
4. **Python Virtual Environment**: For dependency isolation
5. **UTF-8 Encoding**: Required for multilingual content handling

### Security Considerations
1. **Database Credentials**: Stored in environment variables
2. **Manual Authentication**: Required for BOSS system access
3. **Data Privacy**: Historical bidding data sensitivity
4. **Rate Limiting**: Respectful scraping practices

## Monitoring and Maintenance

### Logging Strategy
- Timestamped log files for each pipeline run
- Component-specific log separation
- Error tracking with stack traces
- Performance metrics collection

### Model Management
- Version control for trained models
- Performance monitoring over time
- Regular retraining schedules
- A/B testing capabilities

## Evolution and Versioning

### Version History
- **V1 (Deprecated)**: Basic CatBoost models with OASIS data
- **V2 (Deprecated)**: Enhanced with scraped class timing data
- **V3 (Deprecated)**: Pre-trained models with bug fixes
- **V4 (Current)**: Three-model architecture with uncertainty quantification

### Future Improvements
1. **Real-time Processing**: Stream processing for live bidding
2. **Enhanced Monitoring**: Dashboard for pipeline health
3. **Automated Retraining**: CI/CD for model updates
4. **API Integration**: RESTful endpoints for predictions
5. **Containerization**: Docker deployment for reproducibility

## Usage Guidelines

### Pipeline Execution
```bash
# Run complete pipeline
./run_pipeline.sh

# Individual component execution
python step_1a_BOSSClassScraper.py
python step_1b_HTMLDataExtractor.py
python step_1c_ScrapeOverallResults.py
python step_2_TableBuilder.py
python step_3_BidPrediction.py
```

### Configuration
1. Update `config.py` for target academic terms
2. Set database credentials in environment variables
3. Configure bidding schedules as needed
4. Adjust scraping parameters for performance

### Model Usage
```python
# Load pre-trained models
classification_model = CatBoostClassifier().load_model('production_classification_model.cbm')
median_model = CatBoostRegressor().load_model('production_regression_median_model.cbm')
min_model = CatBoostRegressor().load_model('production_regression_min_model.cbm')
```

## Conclusion
The BidlySMU pipeline represents a sophisticated data engineering solution combining web scraping, data processing, and machine learning. Its modular architecture, comprehensive error handling, and advanced prediction capabilities make it a robust system for SMU course bidding analysis.

The three-model approach with uncertainty quantification provides statistically grounded predictions, while the parallel execution and caching mechanisms ensure performance efficiency. The pipeline's design allows for both batch processing and potential real-time extensions.