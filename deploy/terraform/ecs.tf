# =============================================================================
# ECS FARGATE CLUSTER AND TASK DEFINITION
# =============================================================================

# ECS Cluster
resource "aws_ecs_cluster" "main" {
  name = "${var.project_name}-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = {
    Name = "${var.project_name}-cluster"
  }
}

# CloudWatch Log Group for ECS
resource "aws_cloudwatch_log_group" "ecs" {
  name              = "/ecs/${var.project_name}-pipeline"
  retention_in_days = var.log_retention_days

  tags = {
    Name = "${var.project_name}-ecs-logs"
  }
}

# ECS Task Definition
resource "aws_ecs_task_definition" "pipeline" {
  family                   = "${var.project_name}-pipeline-task"
  task_role_arn            = aws_iam_role.ecs_task.arn
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.ecs_cpu
  memory                   = var.ecs_memory

  ephemeral_storage {
    size_in_gib = var.ecs_ephemeral_storage
  }

  runtime_platform {
    cpu_architecture        = "X86_64"
    operating_system_family = "LINUX"
  }

  container_definitions = jsonencode([
    {
      name      = "${var.project_name}-pipeline"
      image     = "${aws_ecr_repository.pipeline.repository_url}:${var.pipeline_image_tag}"
      essential = true

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.ecs.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "ecs"
        }
      }

      environment = [
        {
          name  = "ACAD_TERM_ID"
          value = var.acad_term_id
        },
        {
          name  = "PYTHONUTF8"
          value = "1"
        },
        {
          name  = "PYTHONIOENCODING"
          value = "utf-8"
        },
        {
          name  = "USE_SUPABASE_STORAGE"
          value = tostring(var.use_supabase_storage)
        }
      ]

      secrets = [
        {
          name      = "DB_HOST"
          valueFrom = aws_ssm_parameter.db_host.arn
        },
        {
          name      = "DB_NAME"
          valueFrom = aws_ssm_parameter.db_name.arn
        },
        {
          name      = "DB_USER"
          valueFrom = aws_ssm_parameter.db_user.arn
        },
        {
          name      = "DB_PASSWORD"
          valueFrom = aws_ssm_parameter.db_password.arn
        },
        {
          name      = "DB_PORT"
          valueFrom = aws_ssm_parameter.db_port.arn
        },
        {
          name      = "BOSS_EMAIL"
          valueFrom = aws_ssm_parameter.boss_email.arn
        },
        {
          name      = "BOSS_PASSWORD"
          valueFrom = aws_ssm_parameter.boss_password.arn
        },
        {
          name      = "BOSS_MFA_SECRET"
          valueFrom = aws_ssm_parameter.boss_mfa_secret.arn
        },
        {
          name      = "GEMINI_API_KEY"
          valueFrom = aws_ssm_parameter.gemini_api_key.arn
        },
        {
          name      = "SUPABASE_URL"
          valueFrom = aws_ssm_parameter.supabase_url.arn
        },
        {
          name      = "SUPABASE_SERVICE_KEY"
          valueFrom = aws_ssm_parameter.supabase_service_key.arn
        },
        {
          name      = "SENTRY_DSN"
          valueFrom = aws_ssm_parameter.sentry_dsn.arn
        }
      ]

      linuxParameters = {
        initProcessEnabled = true
      }
    }
  ])

  tags = {
    Name = "${var.project_name}-pipeline-task"
  }
}
