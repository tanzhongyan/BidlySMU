# =============================================================================
# BidlySMU AWS Deployment - Terraform Configuration
# =============================================================================
# Deploys:
# - ECR repositories for ECS pipeline and Lambda scheduler
# - ECS Fargate cluster and task definition for pipeline execution
# - Lambda function (container image) for monthly SharePoint calendar scraping
# - EventBridge Scheduler for monthly cron trigger
# - IAM roles with appropriate permissions
# =============================================================================

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Backend configuration - uncomment and configure for production
  # backend "s3" {
  #   bucket         = "bidlysmu-terraform-state"
  #   key            = "terraform.tfstate"
  #   region         = "ap-southeast-1"
  #   encrypt        = true
  #   dynamodb_table = "bidlysmu-terraform-locks"
  # }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "BidlySMU"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

# =============================================================================
# DATA SOURCES - Reference existing AWS resources
# =============================================================================

# Get current AWS account ID
data "aws_caller_identity" "current" {}

# Get available AZs in the region
data "aws_availability_zones" "available" {
  state = "available"
}

# Reference existing VPC (or create new one if var.create_vpc = true)
data "aws_vpc" "existing" {
  count = var.create_vpc ? 0 : 1

  # Match by VPC ID, Name tag, or isDefault
  filter {
    name   = "isDefault"
    values = ["true"]
  }
}

# Reference existing subnets (note: default VPC subnets are public)
data "aws_subnets" "existing" {
  count = var.create_vpc ? 0 : 1
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.existing[0].id]
  }
}

# KMS key for SSM SecureString decryption
data "aws_kms_alias" "ssm" {
  name = "alias/aws/ssm"
}

# Reference existing security group
data "aws_security_group" "existing" {
  count = var.create_vpc ? 0 : 1

  vpc_id = data.aws_vpc.existing[0].id

  # Match by security group ID or group name
  filter {
    name   = substr(var.security_group_name, 0, 3) == "sg-" ? "group-id" : "group-name"
    values = [var.security_group_name]
  }
}