# =============================================================================
# OUTPUT VALUES
# =============================================================================

# =============================================================================
# ECR REPOSITORIES
# =============================================================================

output "pipeline_ecr_repository_url" {
  description = "URL of the pipeline ECR repository"
  value       = aws_ecr_repository.pipeline.repository_url
}

output "scheduler_ecr_repository_url" {
  description = "URL of the scheduler ECR repository"
  value       = aws_ecr_repository.scheduler.repository_url
}

# =============================================================================
# ECS
# =============================================================================

output "ecs_cluster_name" {
  description = "Name of the ECS cluster"
  value       = aws_ecs_cluster.main.name
}

output "ecs_cluster_arn" {
  description = "ARN of the ECS cluster"
  value       = aws_ecs_cluster.main.arn
}

output "ecs_task_definition_arn" {
  description = "ARN of the ECS task definition"
  value       = aws_ecs_task_definition.pipeline.arn
}

output "ecs_task_definition_family" {
  description = "Family of the ECS task definition"
  value       = aws_ecs_task_definition.pipeline.family
}

# =============================================================================
# LAMBDA
# =============================================================================

output "lambda_function_name" {
  description = "Name of the Lambda function"
  value       = aws_lambda_function.scheduler.function_name
}

output "lambda_function_arn" {
  description = "ARN of the Lambda function"
  value       = aws_lambda_function.scheduler.arn
}

output "lambda_function_invoke_arn" {
  description = "Invoke ARN of the Lambda function"
  value       = aws_lambda_function.scheduler.invoke_arn
}

# =============================================================================
# EVENTBRIDGE SCHEDULER
# =============================================================================

output "monthly_schedule_name" {
  description = "Name of the monthly EventBridge schedule"
  value       = aws_scheduler_schedule.monthly.name
}

# =============================================================================
# IAM ROLES
# =============================================================================

output "ecs_task_execution_role_arn" {
  description = "ARN of the ECS task execution role"
  value       = aws_iam_role.ecs_execution.arn
}

output "ecs_task_role_arn" {
  description = "ARN of the ECS task role"
  value       = aws_iam_role.ecs_task.arn
}

output "lambda_execution_role_arn" {
  description = "ARN of the Lambda execution role"
  value       = aws_iam_role.lambda_execution.arn
}

output "scheduler_role_arn" {
  description = "ARN of the EventBridge scheduler role"
  value       = aws_iam_role.scheduler.arn
}

# =============================================================================
# NETWORKING
# =============================================================================

output "vpc_id" {
  description = "ID of the VPC"
  value       = try(aws_vpc.main[0].id, data.aws_vpc.existing[0].id)
}

output "subnet_ids" {
  description = "IDs of subnets used by ECS tasks"
  value       = try(aws_subnet.private[*].id, data.aws_subnets.existing[0].ids)
}

output "security_group_id" {
  description = "ID of the security group"
  value       = try(aws_security_group.main[0].id, data.aws_security_group.existing[0].id)
}

# =============================================================================
# CLOUDWATCH LOGS
# =============================================================================

output "ecs_log_group_name" {
  description = "Name of the ECS CloudWatch log group"
  value       = aws_cloudwatch_log_group.ecs.name
}

output "lambda_log_group_name" {
  description = "Name of the Lambda CloudWatch log group"
  value       = aws_cloudwatch_log_group.lambda.name
}

# =============================================================================
# DOCKER PUSH COMMANDS
# =============================================================================

output "docker_push_commands" {
  description = "Commands to build and push Docker images"
  value = <<-EOT

    # Build and push pipeline image:
    docker build -t ${aws_ecr_repository.pipeline.repository_url}:${var.pipeline_image_tag} .
    aws ecr get-login-password --region ${var.aws_region} | docker login --username AWS --password-stdin ${aws_ecr_repository.pipeline.repository_url}
    docker push ${aws_ecr_repository.pipeline.repository_url}:${var.pipeline_image_tag}

    # Build and push scheduler image:
    cd lambda/monthly_scheduler
    docker build -t ${aws_ecr_repository.scheduler.repository_url}:${var.scheduler_image_tag} .
    aws ecr get-login-password --region ${var.aws_region} | docker login --username AWS --password-stdin ${aws_ecr_repository.scheduler.repository_url}
    docker push ${aws_ecr_repository.scheduler.repository_url}:${var.scheduler_image_tag}
  EOT
}
