# =============================================================================
# INPUT VARIABLES
# =============================================================================

variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "ap-southeast-1"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "prod"
}

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "bidlysmu"
}

# =============================================================================
# VPC / NETWORKING
# =============================================================================

variable "create_vpc" {
  description = "Whether to create a new VPC (false = use existing)"
  type        = bool
  default     = false
}

variable "vpc_cidr" {
  description = "CIDR block for VPC (if creating new)"
  type        = string
  default     = "10.0.0.0/16"
}

variable "security_group_name" {
  description = "Name of existing security group (if create_vpc = false)"
  type        = string
  default     = "default"
}

# =============================================================================
# ECS CONFIGURATION
# =============================================================================

variable "ecs_cpu" {
  description = "CPU units for ECS task (2048 = 2 vCPU, needed for dual headless Chrome)"
  type        = number
  default     = 2048
}

variable "ecs_memory" {
  description = "Memory (MB) for ECS task (4096 prevents Chrome OOM with parallel scrapers)"
  type        = number
  default     = 4096
}

variable "pipeline_image_tag" {
  description = "Docker image tag for pipeline (defaults to 'latest')"
  type        = string
  default     = "latest"
}

variable "ecs_container_name" {
  description = "Container name in ECS task definition (for environment variable overrides)"
  type        = string
  default     = "bidlysmu-pipeline"
}

variable "acad_term_id" {
  description = "Academic term ID in BOSS format (e.g., AY202526T3A). Changes each term."
  type        = string
}

variable "use_supabase_storage" {
  description = "Whether to download/upload files from Supabase Storage during pipeline runs"
  type        = bool
  default     = true
}

variable "log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 30
}

variable "ecs_ephemeral_storage" {
  description = "Ephemeral storage (GiB) for ECS Fargate task (21-200)"
  type        = number
  default     = 30
}

variable "ecr_image_retention_count" {
  description = "Number of images to keep in ECR before expiration"
  type        = number
  default     = 10
}

# =============================================================================
# LAMBDA CONFIGURATION
# =============================================================================

variable "lambda_memory" {
  description = "Memory (MB) for Lambda function"
  type        = number
  default     = 1769
}

variable "lambda_timeout" {
  description = "Timeout (seconds) for Lambda function"
  type        = number
  default     = 600
}

variable "scheduler_image_tag" {
  description = "Docker image tag for Lambda scheduler (defaults to 'latest')"
  type        = string
  default     = "latest"
}

variable "lambda_deploy_id" {
  description = "Change this to any new value to force Lambda to pull the latest image (e.g., git SHA or timestamp)"
  type        = string
  default     = ""
}

# =============================================================================
# TRUBA API CONFIGURATION
# =============================================================================

variable "truba_api_url" {
  description = "Truba JSON API URL for SMU calendar"
  type        = string
  default     = "https://www.trumba.com/calendars/SMU_RO_Acad.json"
}

variable "months_ahead" {
  description = "Number of months ahead to fetch events for"
  type        = number
  default     = 12
}

# =============================================================================
# SSM PARAMETER STORE VALUES (replaces Secrets Manager, free tier)
# =============================================================================

variable "ssm_db_host" {
  description = "Database host for SSM parameter"
  type        = string
  sensitive   = true
}

variable "ssm_db_name" {
  description = "Database name for SSM parameter"
  type        = string
  sensitive   = true
}

variable "ssm_db_user" {
  description = "Database user for SSM parameter"
  type        = string
  sensitive   = true
}

variable "ssm_db_password" {
  description = "Database password for SSM parameter"
  type        = string
  sensitive   = true
}

variable "ssm_db_port" {
  description = "Database port for SSM parameter"
  type        = string
  default     = "5432"
}

variable "ssm_boss_email" {
  description = "BOSS email for SSM parameter"
  type        = string
  sensitive   = true
}

variable "ssm_boss_password" {
  description = "BOSS password for SSM parameter"
  type        = string
  sensitive   = true
}

variable "ssm_boss_mfa_secret" {
  description = "BOSS MFA secret for SSM parameter"
  type        = string
  sensitive   = true
}

variable "ssm_gemini_api_key" {
  description = "Gemini API key for SSM parameter"
  type        = string
  sensitive   = true
}

variable "ssm_supabase_url" {
  description = "Supabase URL for SSM parameter"
  type        = string
  sensitive   = true
}

variable "ssm_supabase_service_key" {
  description = "Supabase service key for SSM parameter"
  type        = string
  sensitive   = true
}

variable "ssm_sentry_dsn" {
  description = "Sentry DSN for SSM parameter"
  type        = string
  sensitive   = true
}

# =============================================================================
# EVENTBRIDGE SCHEDULER
# =============================================================================

variable "monthly_schedule_cron" {
  description = "Cron expression for monthly scheduler (default: 1st of each month at 8am SGT / 00:00 UTC)"
  type        = string
  default     = "cron(0 0 1 * ? *)"
}

# =============================================================================
# SUPABASE (passed via variables or secrets)
# =============================================================================

variable "supabase_url" {
  description = "Supabase project URL (can also be stored in secrets)"
  type        = string
  default     = ""
  sensitive   = true
}

variable "supabase_service_key" {
  description = "Supabase service_role key (for DB + Storage access)"
  type        = string
  default     = ""
  sensitive   = true
}

