# =============================================================================
# LAMBDA FUNCTION (Container Image)
# =============================================================================

# CloudWatch Log Group for Lambda
resource "aws_cloudwatch_log_group" "lambda" {
  name              = "/aws/lambda/${var.project_name}-scheduler"
  retention_in_days = 30

  tags = {
    Name = "${var.project_name}-lambda-logs"
  }
}

# Lambda Function
resource "aws_lambda_function" "scheduler" {
  function_name = "${var.project_name}-scheduler"
  role          = aws_iam_role.lambda_execution.arn
  package_type  = "Image"

  image_uri = "${aws_ecr_repository.scheduler.repository_url}:${var.scheduler_image_tag}"

  memory_size = var.lambda_memory
  timeout     = var.lambda_timeout

  environment {
    variables = {
      SUPABASE_URL        = var.supabase_url
      ECS_CLUSTER_ARN     = aws_ecs_cluster.main.arn
      TASK_DEF_ARN        = aws_ecs_task_definition.pipeline.arn
      SUBNET_IDS          = jsonencode(var.create_vpc ? aws_subnet.private[*].id : data.aws_subnets.private[0].ids)
      SECURITY_GROUP_IDS  = jsonencode([var.create_vpc ? aws_security_group.main[0].id : data.aws_security_group.existing[0].id])
      SCHEDULER_ROLE_ARN  = aws_iam_role.scheduler.arn
    }
  }

  depends_on = [
    aws_cloudwatch_log_group.lambda,
    aws_iam_role_policy_attachment.lambda_logs,
    aws_iam_role_policy_attachment.lambda_secrets,
    aws_iam_role_policy_attachment.lambda_scheduler,
  ]

  tags = {
    Name = "${var.project_name}-scheduler"
  }
}

# =============================================================================
# EVENTBRIDGE SCHEDULER - MONTHLY CRON
# =============================================================================

# EventBridge Schedule for monthly Lambda trigger
resource "aws_scheduler_schedule" "monthly" {
  name       = "${var.project_name}-monthly-scheduler"
  group_name = "default"

  flexible_time_window {
    mode = "OFF"
  }

  schedule_expression = var.monthly_schedule_cron

  target {
    arn      = aws_lambda_function.scheduler.arn
    role_arn = aws_iam_role.scheduler_invoke.arn

    input = jsonencode({
      trigger = "monthly-schedule"
    })
  }
}

# =============================================================================
# LAMBDA PERMISSION FOR EVENTBRIDGE
# =============================================================================

resource "aws_lambda_permission" "scheduler_invoke" {
  statement_id  = "AllowExecutionFromScheduler"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.scheduler.function_name
  principal     = "scheduler.amazonaws.com"
  source_arn    = aws_scheduler_schedule.monthly.arn
}
