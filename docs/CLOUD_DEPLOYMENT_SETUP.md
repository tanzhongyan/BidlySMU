# BidlySMU Cloud Deployment Setup Guide

> **Complete step-by-step guide for setting up development environment from scratch.**
> Last updated: 2026-05-18

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Prerequisites](#2-prerequisites)
3. [Part A: Supabase Setup (Database + Storage)](#3-part-a-supabase-setup)
4. [Part B: AWS Setup](#4-part-b-aws-setup)
5. [Part C: Local Development Environment](#5-part-c-local-development-environment)
6. [Part D: Build and Deploy](#6-part-d-build-and-deploy)
7. [Part E: Testing the Pipeline](#7-part-e-testing-the-pipeline)
8. [Cost Estimates](#8-cost-estimates)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           BIDLYSMU CLOUD ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐     │
│  │  EVENTBRIDGE     │     │  LAMBDA          │     │  SUPABASE        │     │
│  │  Scheduler       │────▶│  Monthly         │────▶│  Storage         │     │
│  │  (Cron: 1st/Mo)  │     │  Scheduler       │     │  - schedules/    │     │
│  └──────────────────┘     └────────┬─────────┘     │  - input/        │     │
│                                    │               │  - output/       │     │
│                                    ▼               └──────────────────┘     │
│                         ┌──────────────────┐                               │
│                         │  EVENTBRIDGE     │                               │
│                         │  Dynamic Schedules│                              │
│                         │  (Per Window)    │                               │
│                         └────────┬─────────┘                               │
│                                  │                                          │
│                                  ▼                                          │
│  ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐     │
│  │  ECR             │────▶│  ECS FARGATE     │────▶│  SUPABASE        │     │
│  │  - pipeline img  │     │  Pipeline Task   │     │  PostgreSQL DB   │     │
│  │  - scheduler img │     │  (2 vCPU, 4GB)   │     │  (Staging)       │     │
│  └──────────────────┘     └──────────────────┘     └──────────────────┘     │
│                                    │                                          │
│                                    ▼                                          │
│                         ┌──────────────────┐                               │
│                         │  CLOUDWATCH      │                               │
│                         │  Logs            │                               │
│                         └──────────────────┘                               │
│                                                                              │
│  SECRETS MANAGER: bidlysmu-db-credentials, bidlysmu-boss-credentials,       │
│                   bidlysmu-api-keys                                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Monthly (1st of each month)**: EventBridge triggers Lambda scheduler
2. **Lambda**: Fetches BOSS events from Truba JSON API → Updates `bidding_schedules.json` in Supabase Storage → Creates EventBridge schedules for each bidding window
3. **Per Bidding Window**: EventBridge triggers ECS task with environment variables (`ACAD_TERM_ID`, `CURRENT_WINDOW_NAME`)
4. **ECS Task**: Runs full pipeline (scrape BOSS → process data → predict bids → upload results)

### Key Changes (May 2026)

- **Lambda scheduler** now uses Truba JSON API instead of web scraping (no Selenium required)
- **Lightweight Lambda image** - only needs `requests`, `supabase`, `boto3`
- **Environment variables** passed to ECS task via EventBridge overrides
- **Term format conversion**: Truba format (`2026-27_T1`) → BOSS format (`AY202627T1`)

---

## 2. Prerequisites

### Required Accounts

| Service | Purpose | Free Tier? |
|---------|---------|------------|
| **Supabase** | PostgreSQL database + Storage | Yes (500MB DB, 1GB Storage) |
| **AWS** | Lambda, ECS, ECR, SSM Parameter Store, EventBridge | Yes (12 months) |
| **SMU Account** | BOSS access for scraping | N/A (must be student) |

### Local Tools Required

| Tool | Version | Purpose |
|------|---------|---------|
| **Docker Desktop** | Latest | Build container images |
| **AWS CLI v2** | 2.x | Deploy to AWS |
| **Terraform** | 1.5+ | Infrastructure as Code |
| **Python** | 3.12+ | Local development |
| **Git** | Latest | Version control |

### Verify Installations

```bash
# Check Docker
docker --version

# Check AWS CLI
aws --version

# Check Terraform
terraform version

# Check Python
python --version
```

---

## 3. Part A: Supabase Setup

### Step A1: Create Supabase Account

1. Go to [supabase.com](https://supabase.com)
2. Click "Start your project"
3. Sign up with GitHub (recommended) or email
4. Verify your email

### Step A2: Create Staging Project

1. Click "New Project"
2. Fill in details:
   - **Name**: `bidlysmu-staging`
   - **Database Password**: Generate a strong password (save this!)
   - **Region**: `Southeast Asia (Singapore)` - closest to SMU
   - **Plan**: Free tier is fine for development
3. Click "Create new project"
4. Wait ~2 minutes for project to be provisioned

### Step A3: Get Database Connection Details

1. Go to **Project Settings** (gear icon) → **Database**
2. Note these values:

   | Setting | Value Location | Example |
   |---------|----------------|---------|
   | `DB_HOST` | Connection string → Host | `db.xxxxx.supabase.co` |
   | `DB_NAME` | Database name | `postgres` |
   | `DB_USER` | User | `postgres` |
   | `DB_PASSWORD` | The password you set | (your password) |
   | `DB_PORT` | Port | `5432` |

3. **IMPORTANT**: For connection from AWS, you need to:
   - Go to **Project Settings** → **Database** → **Connection string**
   - Select **"Transaction"** pooler mode (port 6543) for serverless
   - Or use direct connection (port 5432) for long-running tasks

### Step A4: Create Storage Bucket

1. Go to **Storage** in left sidebar
2. Click "Create a new bucket"
3. Configure:
   - **Name**: `bidlysmu-files`
   - **Public bucket**: No (keep private)
4. Click "Create bucket"

### Step A5: Get Supabase API Keys

1. Go to **Project Settings** → **API**
2. Note these values:

   | Key | Location | Purpose |
   |-----|----------|---------|
   | `SUPABASE_URL` | Project URL | `https://xxxxx.supabase.co` |
   | `SUPABASE_SERVICE_KEY` | service_role secret | Full admin access (for backend) |

3. ⚠️ **Security**: Never expose `service_role` key on frontend!

### Step A6: Upload Initial Files to Storage

1. Go to **Storage** → `bidlysmu-files`
2. Create folder structure:

   ```
   bidlysmu-files/
   ├── input/
   │   ├── bidding_schedules.json    # REQUIRED - upload from script_input/
   │   └── raw_data.xlsx             # REQUIRED - shared file across all terms (upload from script_input/)
   └── schedules/
       └── existing_schedules.json    # Auto-created by Lambda (do not upload)
   ```

3. Upload files:
   - `script_input/bidding_schedules.json` → `input/bidding_schedules.json`
   - `script_input/raw_data.xlsx` → `input/raw_data.xlsx`

#### Complete Storage Structure

After the pipeline runs, the full structure will be:

```
bidlysmu-files/
├── input/
│   ├── bidding_schedules.json       # BOSS event windows (uploaded manually, updated by Lambda)
│   ├── raw_data.xlsx                # Shared class data across all terms (uploaded manually, updated by scraper)
│   └── overallBossResults/          # Temporary bid results (auto-created by scraper)
│       └── {term}.xlsx              # e.g., 2025-26_T3B.xlsx
├── output/
│   └── {TERM}/{WINDOW}/             # Pipeline output organized by term/window (auto-created by pipeline)
│       ├── new_classes.csv
│       ├── updated_classes.csv
│       ├── new_professors.csv
│       ├── bid_predictions.csv
│       └── ... (other CSV outputs)
└── schedules/
    └── existing_schedules.json      # Tracks created EventBridge schedules (auto-created by Lambda)
```

**What gets created automatically:**

| File/Folder | Created By | When |
|-------------|------------|------|
| `input/overallBossResults/` | Scraper | First run if doesn't exist |
| `input/overallBossResults/*.xlsx` | Scraper | Each pipeline run |
| `output/{TERM}/{WINDOW}/` | Pipeline | First run for each term/window |
| `output/{TERM}/{WINDOW}/*.csv` | Pipeline | Each pipeline run |
| `schedules/existing_schedules.json` | Lambda | First Lambda run |

**What you must upload manually:**

| File | Source |
|------|--------|
| `input/bidding_schedules.json` | `script_input/bidding_schedules.json` |
| `input/raw_data.xlsx` | `script_input/raw_data.xlsx` |

**Example with real values:**
```
bidlysmu-files/
├── input/
│   ├── bidding_schedules.json
│   ├── raw_data.xlsx
│   └── overallBossResults/
│       └── 2025-26_T3B.xlsx
├── output/
│   └── AY202526T3B/R2W3/
│       ├── new_classes.csv
│       └── bid_predictions.csv
└── schedules/
    └── existing_schedules.json
```

#### How Pipeline Uses Storage (when `USE_SUPABASE_STORAGE=true`)

| Phase | Action | Local Path | Supabase Path |
|-------|--------|------------|---------------|
| **Step 0 (Download)** | Downloads input files | `script_input/bidding_schedules.json` | `input/bidding_schedules.json` |
| | | `script_input/raw_data.xlsx` | `input/raw_data.xlsx` |
| | | `script_input/overallBossResults/*.xlsx` | `input/overallBossResults/*.xlsx` |
| **Step 1 (Scrape)** | Creates/updates data | `script_input/raw_data.xlsx` | (uploaded in Step 3) |
| | | `script_input/overallBossResults/*.xlsx` | (uploaded in Step 3) |
| **Step 2 (Process)** | Generates outputs | `script_output/*.csv` | (uploaded in Step 3) |
| **Step 3 (Upload)** | Uploads all files | `script_input/raw_data.xlsx` | `input/raw_data.xlsx` |
| | | `script_input/bidding_schedules.json` | `input/bidding_schedules.json` |
| | | `script_input/overallBossResults/*.xlsx` | `input/overallBossResults/*.xlsx` |
| | | `script_output/*.csv` | `output/{TERM}/{WINDOW}/*.csv` |

**Key differences from local structure:**
- `raw_data.xlsx` is a **single shared file** in both local and Supabase (not split by term/window)
- `overallBossResults/*.xlsx` are **temporary files** stored flat in `input/overallBossResults/`
- Output CSVs are **organized by term/window** in `output/{TERM}/{WINDOW}/`
- Cache files (`db_cache/*.pkl`) are **NOT uploaded** - always downloaded fresh from database

**Note:** If `USE_SUPABASE_STORAGE=false` (default for local dev), all files stay local.

### Step A7: Configure Database Schema

The pipeline will create tables automatically on first run, but you can optionally set up the schema manually:

1. Go to **SQL Editor**
2. Run the schema from your database migration files (if you have them)

Alternatively, the pipeline's `DatabaseHelper` will create tables via `CREATE TABLE IF NOT EXISTS`.

---

## 4. Part B: AWS Setup

### Step B1: Create AWS Account

1. Go to [aws.amazon.com](https://aws.amazon.com)
2. Click "Create an AWS Account"
3. Follow the signup process (requires credit card for verification)
4. Choose **Basic Support** (free)

### Step B2: Configure AWS CLI

```bash
# Install AWS CLI (if not installed)
# macOS: brew install awscli
# Windows: Download from https://aws.amazon.com/cli/

# Configure credentials
aws configure

# Enter when prompted:
# AWS Access Key ID: (leave blank for now)
# AWS Secret Access Key: (leave blank for now)
# Default region: ap-southeast-1
# Default output format: json
```

### Step B3: Create IAM User for Deployment

1. Go to **IAM** → **Users** → **Create user**
2. User name: `bidlysmu-deploy`
3. Select "Provide user access to the AWS Management Console" (optional)
4. Click "Next"
5. **Attach policies directly**:
   - `AmazonEC2ContainerRegistryFullAccess`
   - `AmazonECS_FullAccess`
   - `AWSLambda_FullAccess`
   - `AmazonEventBridgeSchedulerFullAccess`
   - `SecretsManagerReadWrite`
   - `IAMFullAccess` (for creating roles via Terraform)
   - `AmazonVPCFullAccess` (if creating new VPC)
6. Click "Next" → "Create user"

### Step B4: Create Access Keys

1. Click on the user `bidlysmu-deploy`
2. Go to **Security credentials** tab
3. Under "Access keys", click "Create access key"
4. Select "Command Line Interface (CLI)"
5. Click "Next" → "Create access key"
6. **Download the .csv file** (you won't see these again!)
7. Update `aws configure`:

   ```bash
   aws configure
   # Enter the Access Key ID and Secret Access Key from the CSV
   ```

### Step B5: Configure SSM Parameters

Credentials are managed via **AWS SSM Parameter Store** (free tier), not SSM Parameter Store.
The Terraform configuration (`ssm.tf`) creates 12 namespaced parameters under `/bidlysmu/`.

Set values in your `terraform.tfvars` file before running `terraform apply`:

```hcl
# Database (SecureString)
ssm_db_host     = "db.xxxxx.supabase.co"
ssm_db_name     = "postgres"
ssm_db_user     = "postgres"
ssm_db_password = "your-password-here"
ssm_db_port     = "5432"             # String (plaintext)

# BOSS credentials (SecureString)
ssm_boss_email     = "your.email.2023@business.smu.edu.sg"
ssm_boss_password  = "your-smu-password"
ssm_boss_mfa_secret = "YOUR_BASE32_TOTP_SECRET"

# API keys (SecureString)
ssm_gemini_api_key       = "your-gemini-api-key-or-empty"
ssm_supabase_url          = "https://xxxxx.supabase.co"  # String (plaintext)
ssm_supabase_service_key  = "your-service-role-key"
ssm_sentry_dsn            = "your-sentry-dsn-or-empty"
```

All sensitive parameters use `SecureString` (KMS-encrypted). `DB_PORT` and `SUPABASE_URL` are stored as plain `String` since they are not sensitive.
```

**How to get MFA Secret:**
1. Open Microsoft Authenticator app
2. Select your SMU account
3. Tap the three dots → "Set up"
4. Look for "Can't scan?" or "Show secret key"
5. Copy the Base32-encoded string (looks like `JBSWY3DPEHPK3PXP`)

### Step B6: Set Up Sentry (Error Monitoring)

Sentry provides real-time error tracking and performance monitoring for the pipeline.

#### Create Sentry Account

1. Go to [sentry.io](https://sentry.io)
2. Click "Start for free"
3. Sign up with GitHub (recommended) or email
4. Verify your email

#### Create a Project

1. Click **Projects** → **Create Project**
2. Select **Python** as the platform
3. Configure:
   - **Project name**: `bidlysmu-pipeline`
   - **Team**: Select your team (or create one)
   - **Alert frequency**: "On every new issue" (recommended)
4. Click **Create Project**

#### Get the DSN

1. Go to **Settings** → **Projects** → `bidlysmu-pipeline`
2. Click **Client Keys (DSN)** in the left sidebar
3. Copy the DSN (looks like `https://xxxxx@o123456.ingest.sentry.io/1234567`)

#### Add DSN to SSM Parameter Store

Set the Sentry DSN in your `terraform.tfvars`:

```bash
# Update existing secret (replace with your actual DSN)
aws secretsmanager update-secret \
    --secret-id bidlysmu-api-keys \
    --secret-string '{
        "gemini_api_key": "your-gemini-api-key-or-empty",
        "supabase_url": "https://xxxxx.supabase.co",
        "supabase_service_key": "your-service-role-key",
        "sentry_dsn": "https://xxxxx@o123456.ingest.sentry.io/1234567"
    }'
```

#### Sentry Configuration in Code

The pipeline automatically initializes Sentry if `SENTRY_DSN` is present in environment variables. The initialization is in `src/config.py`:

```python
# src/config.py (already implemented)
import sentry_sdk

if SENTRY_DSN:
    sentry_sdk.init(
        dsn=SENTRY_DSN,
        traces_sample_rate=0.1,  # 10% of transactions
        profiles_sample_rate=0.1,  # 10% of profiles
    )
```

#### Test Sentry Integration

After deployment, test that errors are being captured:

```python
# Test locally first
import sentry_sdk
from src.config import SENTRY_DSN

if SENTRY_DSN:
    sentry_sdk.init(dsn=SENTRY_DSN)
    
    # Trigger a test error
    sentry_sdk.capture_message("Test message from BidlySMU")
    print("Check Sentry dashboard for the test message")
```

Or test in the deployed Lambda/ECS task by checking CloudWatch logs for Sentry-related messages.

#### Sentry Dashboard Features

After setup, you can:

- **View Errors**: Real-time error feed with stack traces
- **Set Alerts**: Configure Slack/email notifications for new errors
- **Performance**: Monitor transaction times (useful for pipeline stages)
- **Releases**: Track which deployment caused an error
- **User Context**: See which academic term/window had errors

#### Cost Considerations

| Plan | Events/Month | Team Members | Cost |
|------|--------------|--------------|------|
| **Developer** | 5,000 | 1 | Free |
| **Team** | 50,000 | Unlimited | $26/month |
| **Business** | 200,000+ | Unlimited | $80+/month |

**Recommendation**: Start with the free Developer plan for staging. The pipeline generates minimal events (~100-500/month during normal operation).

---

## 5. Part C: Local Development Environment

### Step C1: Clone Repository

```bash
git clone https://github.com/tanzhongyan/BidlySMU.git
cd BidlySMU
```

### Step C2: Create Python Virtual Environment

```bash
# Create venv
python -m venv .venv

# Activate
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step C3: Create Local .env File

Create `.env` in project root:

```bash
# .env - DO NOT COMMIT THIS FILE!

# Academic term (format: AY + start_year + end_year + term_code)
ACAD_TERM_ID=AY202526T3A

# Bidding schedules path
BIDDING_SCHEDULES_PATH=script_input/bidding_schedules.json

# Target windows (optional - for testing specific windows)
# TARGET_CURRENT_WINDOW=Round 2 Window 3
# TARGET_PREVIOUS_WINDOW=Round 2 Window 2

# Database (Supabase Staging)
DB_HOST=db.xxxxx.supabase.co
DB_NAME=postgres
DB_USER=postgres
DB_PASSWORD=your-password
DB_PORT=5432

# BOSS Credentials
BOSS_EMAIL=your.email.2023@business.smu.edu.sg
BOSS_PASSWORD=your-smu-password
BOSS_MFA_SECRET=your-base32-totp-secret

# Supabase Storage
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_SERVICE_KEY=your-service-role-key
USE_SUPABASE_STORAGE=false  # Set to true when testing cloud flow

# API Keys (optional)
GEMINI_API_KEY=your-gemini-key-or-empty
SENTRY_DSN=your-sentry-dsn-or-empty
```

### Step C4: Verify Local Setup

```bash
# Test database connection
python -c "
from src.db.adapters import Psycopg2Adapter
from src.config import DB_CONFIG
adapter = Psycopg2Adapter(DB_CONFIG)
adapter.connect()
print('Database connection successful!')
adapter.disconnect()
"

# Test Supabase Storage
python -c "
from supabase import create_client
from src.config import SUPABASE_URL, SUPABASE_SERVICE_KEY
client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
print('Supabase connection successful!')
"

# Test Truba API (for Lambda scheduler)
python -c "
from src.scraper.trumba_client import TrubaClient, TrubaConfig
client = TrubaClient(TrubaConfig())
events = client.fetch_boss_events()
print(f'Fetched {len(events)} BOSS events from Truba API')
"
```

---

## 6. Part D: Build and Deploy

### Step D1: Configure Terraform Variables

```bash
cd deploy/terraform

# Copy example file
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars`:

```hcl
# AWS Configuration
aws_region   = "ap-southeast-1"
environment  = "staging"
project_name = "bidlysmu"

# Use existing default VPC (free!)
create_vpc          = false
vpc_name            = "default"
security_group_name = "default"

# ECS Configuration
ecs_cpu           = 2048
ecs_memory        = 4096
pipeline_image_tag = "v1.0.0"
ecs_container_name = "bidlysmu-pipeline"

# Lambda Configuration (lightweight - no Selenium)
lambda_memory       = 512
lambda_timeout      = 300
scheduler_image_tag = "v1.0.0"

# Truba API Configuration
truba_api_url = "https://www.trumba.com/calendars/SMU_RO_Acad.json"
months_ahead  = 12

# Monthly schedule (1st of each month at 8am SGT)
monthly_schedule_cron = "cron(0 0 1 * ? *)"

# Secrets (already created in Step B5)
db_secret_name       = "bidlysmu-db-credentials"
boss_secret_name     = "bidlysmu-boss-credentials"
api_keys_secret_name = "bidlysmu-api-keys"

# Supabase
supabase_url = "https://xxxxx.supabase.co"
```

### Step D2: Initialize Terraform

```bash
cd deploy/terraform

# Initialize
terraform init

# Preview changes
terraform plan
```

### Step D3: Apply Terraform (Create Infrastructure)

```bash
# Apply (this will create ECR, ECS, Lambda, IAM roles, etc.)
terraform apply

# Type 'yes' when prompted
```

This creates:
- 2 ECR repositories (`bidlysmu-pipeline`, `bidlysmu-scheduler`)
- 1 ECS cluster (`bidlysmu-cluster`)
- 1 ECS task definition (`bidlysmu-pipeline-task`)
- 1 Lambda function (`bidlysmu-scheduler`)
- IAM roles for ECS execution, Lambda, EventBridge
- CloudWatch log groups
- EventBridge schedule for monthly Lambda trigger

### Step D4: Get ECR Repository URLs

```bash
# Get outputs
terraform output

# Note the ECR repository URLs, e.g.:
# ecr_pipeline_url = "123456789012.dkr.ecr.ap-southeast-1.amazonaws.com/bidlysmu-pipeline"
# ecr_scheduler_url = "123456789012.dkr.ecr.ap-southeast-1.amazonaws.com/bidlysmu-scheduler"
```

### Step D5: Build and Push Docker Images

```bash
# From project root
cd ../..

# Login to ECR
aws ecr get-login-password --region ap-southeast-1 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.ap-southeast-1.amazonaws.com

# ============================================
# IMAGE 1: Pipeline (for ECS Fargate)
# Contains Selenium for BOSS web scraping
# ============================================

# Build
docker build -t bidlysmu-pipeline:v1.0.0 .

# Tag
docker tag bidlysmu-pipeline:v1.0.0 123456789012.dkr.ecr.ap-southeast-1.amazonaws.com/bidlysmu-pipeline:v1.0.0

# Push
docker push 123456789012.dkr.ecr.ap-southeast-1.amazonaws.com/bidlysmu-pipeline:v1.0.0

# ============================================
# IMAGE 2: Scheduler (for Lambda)
# Lightweight - uses Truba JSON API (no Selenium)
# ============================================

# Build (from lambda directory)
cd lambda/monthly_scheduler

# Build with Lambda base image
docker build -t bidlysmu-scheduler:v1.0.0 .

# Tag
docker tag bidlysmu-scheduler:v1.0.0 123456789012.dkr.ecr.ap-southeast-1.amazonaws.com/bidlysmu-scheduler:v1.0.0

# Push
docker push 123456789012.dkr.ecr.ap-southeast-1.amazonaws.com/bidlysmu-scheduler:v1.0.0
```

### Step D6: Update Lambda with New Image

After pushing the scheduler image, update the Lambda function:

```bash
aws lambda update-function-code \
    --function-name bidlysmu-scheduler \
    --image-uri 123456789012.dkr.ecr.ap-southeast-1.amazonaws.com/bidlysmu-scheduler:v1.0.0
```

### Step D7: Update ECS Task Definition

If you need to update the ECS task with a new image:

```bash
# Get current task definition
aws ecs describe-task-definition --task-definition bidlysmu-pipeline-task --query 'taskDefinition' > task-def.json

# Edit the image URL in task-def.json if needed

# Register new revision
aws ecs register-task-definition --cli-input-json file://task-def.json
```

---

## 7. Part E: Testing the Pipeline

### Step E1: Test Lambda Scheduler Locally

```bash
# Test Truba API fetch
python -c "
from src.scraper.truba_client import TrubaClient, TrubaConfig

config = TrubaConfig(months_ahead=12)
client = TrubaClient(config)
events = client.fetch_boss_events()

print(f'Fetched {len(events)} BOSS events')
for event in events[:5]:
    print(f'  {event.term} | {event.abbrev} | {event.datetime}')

# Test term conversion
from lambda.monthly_scheduler.lambda_function import convert_term_to_acad_term_id
print(f'Term conversion: 2026-27_T1 -> {convert_term_to_acad_term_id(\"2026-27_T1\")}')
"
```

### Step E2: Test Lambda in AWS Console

1. Go to **Lambda** → `bidlysmu-scheduler`
2. Click **Test** tab
3. Create new event:
   ```json
   {
     "trigger": "manual-test"
   }
   ```
4. Click **Test**
5. Check **Execution result** and **CloudWatch Logs**

Expected output:
```json
{
  "statusCode": 200,
  "body": {
    "message": "Successfully updated schedules",
    "events_found": 14,
    "schedules_created": ["bidlysmu-pipeline-2026-27_T1-R1W1", ...],
    "total_windows": 14
  }
}
```

### Step E3: Manually Trigger ECS Pipeline

```bash
# Run ECS task manually
aws ecs run-task \
    --cluster bidlysmu-cluster \
    --task-definition bidlysmu-pipeline-task \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[subnet-xxxxx],securityGroups=[sg-xxxxx],assignPublicIp=DISABLED}"
```

Get subnet and security group IDs from:
```bash
aws ec2 describe-subnets --filters "Name=vpc-id,Values=vpc-xxxxx"
aws ec2 describe-security-groups --filters "Name=vpc-id,Values=vpc-xxxxx"
```

### Step E4: Check CloudWatch Logs

1. Go to **CloudWatch** → **Log groups**
2. Check:
   - `/ecs/bidlysmu-pipeline` - Pipeline execution logs
   - `/aws/lambda/bidlysmu-scheduler` - Lambda logs

### Step E5: Verify Data in Supabase

1. Go to **Supabase Dashboard** → **Table Editor**
2. Check tables were created and populated:
   - `acad_terms`
   - `courses`
   - `professors`
   - `classes`
   - `bid_windows`
   - `bid_results`
   - `bid_predictions`

3. Go to **Storage** → `bidlysmu-files`
4. Check output files in `output/{term}/{window}/`

---

## 8. Cost Estimates

### AWS Monthly Costs (Staging)

| Service | Usage | Cost |
|---------|-------|------|
| **Lambda** | 1 run/month, 5 min | ~$0.01 |
| **ECS Fargate** | 4 windows × 3hr = 12hr | ~$1.00 |
| **ECR** | 2 repos, <1GB | ~$0.10 |
| **SSM Parameter Store** | 12 params | ~$0.00 (free tier) |
| **CloudWatch Logs** | 1GB | ~$0.50 |
| **EventBridge Scheduler** | Free tier | $0.00 |
| **NAT Gateway** | If creating new VPC | ~$30.00 |
| **Total (existing VPC)** | | **~$3.11/month** |
| **Total (new VPC)** | | **~$33.11/month** |

### Supabase Costs

| Plan | Limits | Cost |
|------|--------|------|
| **Free** | 500MB DB, 1GB Storage | $0 |
| **Pro** | 8GB DB, 100GB Storage | $25/month |

**Recommendation**: Start with Free tier for staging, upgrade to Pro for production.

---

## 9. Troubleshooting

### Common Issues

#### Issue: Lambda timeout (300s exceeded)

**Cause**: Truba API request taking too long.

**Solution**:
- Increase Lambda timeout in `terraform.tfvars`:
  ```hcl
  lambda_timeout = 600  # 10 minutes
  ```

#### Issue: ECS task fails to start

**Cause**: Missing subnet/security group configuration.

**Solution**:
- Verify VPC has internet access (NAT Gateway or public subnet)
- Check security group allows outbound HTTPS (443) and PostgreSQL (5432)

#### Issue: Cannot connect to Supabase database

**Cause**: Supabase firewall blocking AWS IP.

**Solution**:
1. Go to Supabase → **Project Settings** → **Database**
2. Disable "Enable IPv4 blocking" or add AWS IP range to allowlist

#### Issue: SSM Parameter Store access denied

**Cause**: IAM role missing permissions.

**Solution**:
- Verify ECS task role has `ssm:GetParameter` and `ssm:GetParameters` permissions
- Check secret ARN matches in task definition

#### Issue: Docker build fails (Chrome installation)

**Cause**: Chrome for Testing version mismatch.

**Solution**:
- Update Chrome version in Dockerfile to match latest stable
- Check [Chrome for Testing JSON](https://googlechromelabs.github.io/chrome-for-testing/known-good-versions-with-downloads.json) for available versions

#### Issue: ACAD_TERM_ID not passed to ECS task

**Cause**: EventBridge schedule not configured with environment overrides.

**Solution**:
- Verify Lambda creates schedules with `Overrides.ContainerOverrides.Environment`
- Check `ECS_CONTAINER_NAME` matches the container name in task definition

### Useful Commands

```bash
# View ECS task logs
aws logs tail /ecs/bidlysmu-pipeline --follow

# View Lambda logs
aws logs tail /aws/lambda/bidlysmu-scheduler --follow

# List all EventBridge schedules
aws scheduler list-schedules

# Manually trigger Lambda
aws lambda invoke --function-name bidlysmu-scheduler response.json

# Force Lambda update
aws lambda update-function-code \
    --function-name bidlysmu-scheduler \
    --image-uri 123456789012.dkr.ecr.ap-southeast-1.amazonaws.com/bidlysmu-scheduler:v1.0.0

# Destroy all Terraform resources
cd deploy/terraform
terraform destroy
```

---

## Quick Reference Checklist

### Supabase Setup
- [ ] Create Supabase account
- [ ] Create staging project (`bidlysmu-staging`)
- [ ] Note database credentials (host, user, password, port)
- [ ] Create storage bucket (`bidlysmu-files`)
- [ ] Get API keys (URL, service_role key)
- [ ] Upload `bidding_schedules.json` to `schedules/`

### AWS Setup
- [ ] Create AWS account
- [ ] Install and configure AWS CLI
- [ ] Create IAM user with required policies
- [ ] Create access keys and configure CLI
- [ ] Create 3 secrets in SSM Parameter Store:
  - [ ] `bidlysmu-db-credentials`
  - [ ] `bidlysmu-boss-credentials`
  - [ ] `bidlysmu-api-keys`
- [ ] Set up Sentry (optional but recommended):
  - [ ] Create Sentry account
  - [ ] Create Python project (`bidlysmu-pipeline`)
  - [ ] Copy DSN and add to `bidlysmu-api-keys` secret

### Local Development
- [ ] Clone repository
- [ ] Create Python virtual environment
- [ ] Install dependencies
- [ ] Create `.env` file with all credentials
- [ ] Test database connection
- [ ] Test Supabase Storage connection
- [ ] Test Truba API fetch

### Deployment
- [ ] Configure `terraform.tfvars`
- [ ] Run `terraform init`
- [ ] Run `terraform apply`
- [ ] Build and push Docker images (2 images)
- [ ] Update Lambda function code
- [ ] Test Lambda manually
- [ ] Test ECS task manually
- [ ] Verify data in Supabase

---

## Next Steps

1. **Set up CI/CD**: Use GitHub Actions to automate Docker builds and deployments
2. **Create production environment**: Separate Supabase project and AWS resources
3. **Set up alerts**: CloudWatch Alarms for failed tasks
4. **Configure Sentry alerts**: Set up Slack/email notifications in Sentry dashboard
5. **Document API**: Create API documentation if exposing predictions

---

**Questions?** Check the [Troubleshooting](#9-troubleshooting) section or create an issue on GitHub.