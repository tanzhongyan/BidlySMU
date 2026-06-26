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

variable "vpc_name" {
  description = "Name of existing VPC to use (if create_vpc = false)"
  type        = string
  default     = "default"
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
  description = "CPU units for ECS task"
  type        = number
  default     = 2048
}

variable "ecs_memory" {
  description = "Memory (MB) for ECS task"
  type        = number
  default     = 4096
}

variable "pipeline_image_tag" {
  description = "Docker image tag for pipeline"
  type        = string
  default     = "v1.0.0"
}

variable "ecs_container_name" {
  description = "Container name in ECS task definition (for environment variable overrides)"
  type        = string
  default     = "bidlysmu-pipeline"
}

# =============================================================================
# LAMBDA CONFIGURATION
# =============================================================================

variable "lambda_memory" {
  description = "Memory (MB) for Lambda function"
  type        = number
  default     = 1024
}

variable "lambda_timeout" {
  description = "Timeout (seconds) for Lambda function"
  type        = number
  default     = 600
}

variable "scheduler_image_tag" {
  description = "Docker image tag for Lambda scheduler"
  type        = string
  default     = "v1.0.0"
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
# SECRETS MANAGER
# =============================================================================

variable "db_secret_name" {
  description = "Name of Secrets Manager secret for database credentials"
  type        = string
  default     = "bidlysmu-db-credentials"
}

variable "boss_secret_name" {
  description = "Name of Secrets Manager secret for BOSS credentials"
  type        = string
  default     = "bidlysmu-boss-credentials"
}

variable "api_keys_secret_name" {
  description = "Name of Secrets Manager secret for API keys"
  type        = string
  default     = "bidlysmu-api-keys"
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
}

variable "ecs_cluster_arn" {
  description = "ECS cluster ARN for EventBridge target (set after ECS is created)"
  type        = string
  default     = ""
}

variable "task_definition_arn" {
  description = "ECS task definition ARN for EventBridge target (set after task is created)"
  type        = string
  default     = ""
}
