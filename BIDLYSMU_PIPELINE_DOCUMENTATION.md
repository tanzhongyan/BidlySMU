# BidlySMU Pipeline Documentation

## Overview
BidlySMU is a comprehensive data pipeline for Singapore Management University (SMU) course bidding prediction system. The pipeline follows an ETL (Extract, Transform, Load) pattern with machine learning integration for predicting minimum and median bid prices for university courses.

**Important Note**: The pipeline consists of two separate workflows:
1. **Production Pipeline** (`run_pipeline.sh`): Data collection → Processing → Prediction using pre-trained models
2. **Model Training** (`V4_03_catboost_training.ipynb`): Trains the three CatBoost models from historical data (separate process)

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
**Purpose**: Load pre-trained models and generate predictions for bid price forecasting.

#### Bid Prediction (`step_3_BidPrediction.py`)
- **Function**: Loads pre-trained CatBoost models and applies them for bid prediction
- **Note**: Model training happens separately in `V4_03_catboost_training.ipynb`
- **Key Features**:
  - **Three-model architecture**:
    1. **Classification Model**: Predicts if a course will receive bids
    2. **Median Bid Regression**: Predicts median bid price
    3. **Minimum Bid Regression**: Predicts minimum successful bid
  - **Advanced feature engineering**:
    - Course code decomposition (subject area, catalogue number)
    - Bidding window feature extraction (round, window)
    - Day-of-week one-hot encoding (7 binary features)
    - Instructor categorical encoding (JSON array format)
    - Time and vacancy features
  - **Uncertainty quantification**:
    - Virtual ensemble method (10 tree subsets) for prediction uncertainty
    - T-distribution based confidence intervals
    - Entropy-based confidence scoring for classification
    - Percentile-based safety factors (1%-99%)
  - **Model files**: Loads three `.cbm` (CatBoost) production model files:
    - `production_classification_model.cbm`
    - `production_regression_median_model.cbm`
    - `production_regression_min_model.cbm`
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
Stream A: Class Details Pipeline (Sequential)
BOSS Website → [1a] Scraper → HTML Files → [1b] Extractor → raw_data.xlsx

Stream B: Historical Results (Parallel with Stream A)
BOSS Website → [1c] Overall Results Scraper → Historical Bidding Data

                    ↓ (Both streams complete)
            Combined Raw Data Ready
```

### Phase 2: Data Processing (Sequential)
```
Raw Data (raw_data.xlsx) → [2] Table Builder → Database Tables
                    ↓
           Entity Resolution (professors, courses)
                    ↓
           Create Classes with class_id
                    ↓
        Generate CSV outputs for verification
                    ↓
         Database Ready for ML
```

### Phase 3: Prediction using Pre-trained Models (Sequential)
```
Pre-trained Models + Structured Data → [3] Bid Prediction → Predictions
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

### 2. Critical Execution Order
- **Step 2 MUST complete before Step 3**: Classes with `class_id` must exist before predictions
- **Database constraint**: `BidPrediction.classId` references `Classes.id` (foreign key)
- **Multi-professor handling**: Step 2 creates separate class records for each professor
- **Class mapping**: Step 3 links predictions to `class_id` via `new_classes.csv` + database cache

### 3. Error Resilience
- Comprehensive logging at each step
- Pipeline halting on critical failures
- Retry mechanisms for web scraping
- Data validation at transformation boundaries
- Transaction rollback in database operations

### 4. Performance Optimization
- Parallel execution for independent tasks (Step 1a+1b runs parallel with 1c)
- Database connection pooling
- Caching for entity resolution
- Batch processing for large datasets
- Pre-loading models once per execution

### 5. Configuration Management
- Centralized configuration file
- Environment variable support
- Academic calendar awareness
- Schedule-based execution control

## Pipeline vs Model Training

### Production Pipeline (`run_pipeline.sh`)
**Purpose**: Collect current data, process it, and generate predictions
- **Frequency**: Runs for each bidding window (multiple times per term)
- **Input**: Live BOSS system data
- **Output**: Predictions for current courses
- **Models**: Uses pre-trained models (loads `.cbm` files)
- **Duration**: ~15-30 minutes depending on data volume

### Model Training Workflow (`V4_03_catboost_training.ipynb`)
**Purpose**: Train/retrain prediction models from historical data
- **Frequency**: Periodic (e.g., once per term or when model performance degrades)
- **Input**: Historical bidding data from database
- **Output**: Three trained model files (`.cbm` format) + safety factor table
- **Process**: 
  1. Load historical data
  2. Feature engineering and preprocessing
  3. Train three CatBoost models with hyperparameter tuning
  4. Generate validation metrics
  5. Calculate safety factors via t-distribution fitting
  6. Save production models to `script_output/models/`
- **Duration**: ~1-2 hours depending on data size and hyperparameters

### Critical Distinction
- The **pipeline** (steps 1-3) does NOT train models
- Model training happens **offline** in Jupyter notebooks
- Pipeline always loads the **latest production models**
- Both workflows share the same feature engineering code (`SMUBiddingTransformer`)

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

## Architecture Analysis

### Executive Summary
BidlySMU implements a sophisticated three-phase data pipeline for SMU course bidding prediction. The architecture demonstrates strong engineering principles with modular design, parallel execution, and comprehensive error handling. The system successfully integrates web scraping, data processing, and machine learning into a cohesive pipeline.

### Strengths

#### 1. **Modular Design Excellence**
- **Clear Separation of Concerns**: Each Python script has a single, well-defined responsibility
- **Independent Component Testing**: Components can be tested in isolation
- **Configurable Interfaces**: Clear input/output contracts between components
- **Version Control**: Deprecated versions maintained for reference

#### 2. **Performance Optimization**
- **Parallel Execution**: Step 1 components run concurrently (Stream A and B)
- **Caching Strategy**: Entity resolution caching in TableBuilder reduces database load
- **Batch Processing**: Efficient handling of large datasets
- **Connection Pooling**: Database connections managed effectively

#### 3. **Robust Error Handling**
- **Comprehensive Logging**: Timestamped logs for each pipeline run
- **Pipeline Halting**: Fail-fast approach with clear error messages
- **Retry Mechanisms**: Web scraping includes retry logic
- **Data Validation**: Multiple validation points throughout the pipeline

#### 4. **Advanced ML Architecture**
- **Three-Model Approach**: Classification + dual regression provides comprehensive predictions
- **Uncertainty Quantification**: Statistically grounded confidence intervals
- **Feature Engineering**: Sophisticated feature extraction pipeline
- **Model Persistence**: `.cbm` format for production deployment

### Areas for Improvement

#### 1. **Database Dependency**
- **Current State**: Tight coupling with PostgreSQL
- **Recommendation**: Abstract database layer to support multiple backends
- **Impact**: Would improve deployment flexibility

#### 2. **Configuration Management**
- **Current State**: Mixed approach (config.py + environment variables)
- **Recommendation**: Unified configuration system with validation
- **Impact**: Better configuration error handling

#### 3. **Monitoring and Alerting**
- **Current State**: Basic file-based logging
- **Recommendation**: Integration with monitoring systems (Prometheus, Grafana)
- **Impact**: Proactive issue detection

#### 4. **Testing Coverage**
- **Current State**: Limited evidence of automated testing
- **Recommendation**: Unit tests for each component, integration tests for pipeline
- **Impact**: Increased reliability and easier maintenance

## Technical Debt Analysis

### 1. **Selenium Dependency**
- **Risk**: Browser automation can be fragile and version-dependent
- **Mitigation**: Consider headless browser alternatives or API-based approaches
- **Priority**: Medium - functional but maintenance-heavy

### 2. **Manual Authentication**
- **Risk**: Human intervention required for pipeline execution
- **Mitigation**: Investigate automated authentication options
- **Priority**: High - limits automation potential

### 3. **Large File Storage**
- **Risk**: HTML files and Excel intermediates consume significant storage
- **Mitigation**: Implement cleanup strategies or streaming processing
- **Priority**: Low - storage is cheap but could impact performance

### 4. **Model Version Management**
- **Risk**: Multiple `.cbm` files without explicit version tracking
- **Mitigation**: Model registry with metadata tracking
- **Priority**: Medium - important for production reliability

## Scalability Assessment

### Current Capacity
- **Data Volume**: Handles multiple academic years of course data
- **Processing Speed**: Parallel scraping optimizes data collection
- **Model Complexity**: Three CatBoost models with sophisticated features

### Scaling Limitations
1. **Database Bottleneck**: Single PostgreSQL instance
2. **Memory Constraints**: Large pandas DataFrames in memory
3. **Scraping Rate**: Respectful scraping but limited by website response times
4. **Training Time**: CatBoost training can be resource-intensive

### Scaling Recommendations
1. **Database Scaling**: Read replicas for analytics, connection pooling optimization
2. **Memory Management**: Chunk processing for large datasets
3. **Distributed Scraping**: Multiple IP addresses/instances for parallel scraping
4. **Model Optimization**: Feature importance analysis to reduce dimensionality

## Security Analysis

### Strengths
1. **Credential Management**: Database credentials in environment variables
2. **Data Validation**: Input validation at multiple points
3. **Error Obfuscation**: Limited sensitive information in error messages

### Concerns
1. **Manual Authentication**: Credentials potentially exposed during manual login
2. **Data Sensitivity**: Historical bidding data requires careful handling
3. **Web Scraping Ethics**: Rate limiting but still automated access

### Recommendations
1. **Credential Rotation**: Automated credential management system
2. **Data Encryption**: At-rest encryption for sensitive data
3. **Access Logging**: Comprehensive audit trails for data access

## Reliability Assessment

### Availability
- **Pipeline Success Rate**: High (based on error handling design)
- **Recovery Time**: Moderate (manual intervention may be needed)
- **Data Consistency**: Strong (database transactions and validation)

### Fault Tolerance
1. **Component Isolation**: Failure in one component doesn't corrupt others
2. **Data Recovery**: Intermediate files allow partial re-execution
3. **Error Propagation**: Clear error messages and logging

### Recommendations for Improvement
1. **Checkpointing**: Save intermediate states for faster recovery
2. **Health Checks**: Component-level health monitoring
3. **Circuit Breakers**: Prevent cascading failures

## Maintainability Analysis

### Code Quality
- **Documentation**: Good high-level documentation, limited inline comments
- **Structure**: Clear modular structure with consistent naming
- **Complexity**: Moderate complexity well-managed through separation

### Technical Stack
- **Modern Dependencies**: Up-to-date versions of key libraries
- **Standard Tools**: Industry-standard tools (Selenium, CatBoost, PostgreSQL)
- **Development Experience**: Standard Python development setup

### Maintenance Challenges
1. **Selenium Updates**: Browser and driver version compatibility
2. **Database Schema Evolution**: Schema changes require careful migration
3. **Model Retraining**: Regular updates needed for accuracy

## Performance Metrics Analysis

### Current Performance Characteristics
1. **Scraping Speed**: Limited by website response times and rate limiting
2. **Processing Time**: Efficient batch processing with pandas
3. **Model Inference**: Fast CatBoost predictions
4. **Memory Usage**: Moderate, with potential for optimization

### Optimization Opportunities
1. **Database Indexing**: Additional indexes for common queries
2. **Memory Profiling**: Identify and optimize memory hotspots
3. **Parallel Processing**: Further parallelization opportunities in Step 2
4. **Caching Optimization**: More aggressive caching of processed data

## Future Architecture Directions

### Short-term Improvements (1-3 months)
1. **Automated Testing**: Unit and integration test suite
2. **Configuration Validation**: Schema-based configuration validation
3. **Enhanced Logging**: Structured logging with metrics collection
4. **Documentation**: API documentation and deployment guides

### Medium-term Enhancements (3-6 months)
1. **Containerization**: Docker deployment for reproducibility
2. **CI/CD Pipeline**: Automated testing and deployment
3. **Monitoring Dashboard**: Real-time pipeline monitoring
4. **API Layer**: RESTful API for predictions

### Long-term Vision (6-12 months)
1. **Real-time Processing**: Stream processing for live bidding
2. **Microservices Architecture**: Decompose monolithic components
3. **Machine Learning Pipeline**: Automated model training and deployment
4. **Multi-University Support**: Architecture generalization

## Compliance and Ethics Considerations

### Data Privacy
- **Current State**: Limited data privacy measures
- **Recommendation**: Data anonymization and access controls
- **Impact**: Essential for production deployment

### Web Scraping Ethics
- **Current State**: Respectful scraping with rate limiting
- **Recommendation**: Explicit terms of service compliance
- **Impact**: Legal and ethical compliance

### Model Fairness
- **Current State**: No explicit fairness testing
- **Recommendation**: Bias detection and mitigation
- **Impact**: Ethical AI deployment

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
- Pre-trained models stored as `.cbm` files in `script_output/models/`
- Three production models:
  - `classification/production_classification_model.cbm`
  - `regression_median/production_regression_median_model.cbm`
  - `regression_min/production_regression_min_model.cbm`
- Model training done separately in `V4_03_catboost_training.ipynb`
- Version control for trained models (V1, V2, V3, V4)
- Performance monitoring through validation metrics
- Safety factor table updated after model training

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

## Recommendations Priority

### High Priority
1. Implement automated testing suite
2. Enhance security measures for sensitive data
3. Add comprehensive monitoring

### Medium Priority
1. Containerize deployment
2. Implement CI/CD pipeline
3. Optimize database performance

### Low Priority
1. Refactor for microservices
2. Add real-time processing
3. Generalize for multi-university support

## Conclusion

The BidlySMU pipeline represents a sophisticated data engineering solution combining web scraping, data processing, and machine learning. Its modular architecture, comprehensive error handling, and advanced prediction capabilities make it a robust system for SMU course bidding analysis.

The three-model approach with uncertainty quantification provides statistically grounded predictions, while the parallel execution and caching mechanisms ensure performance efficiency. The pipeline's design allows for both batch processing and potential real-time extensions.

The architecture successfully balances complexity with maintainability, providing a robust platform for SMU course bidding predictions while allowing for future evolution and enhancement.