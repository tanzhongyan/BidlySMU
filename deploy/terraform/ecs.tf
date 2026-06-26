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
  retention_in_days = 30

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
          value = "AY202526T3A"
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
          value = "true"
        }
      ]

      secrets = [
        {
          name      = "DB_HOST"
          valueFrom = "${data.aws_secretsmanager_secret.db.arn}:DB_HOST::"
        },
        {
          name      = "DB_NAME"
          valueFrom = "${data.aws_secretsmanager_secret.db.arn}:DB_NAME::"
        },
        {
          name      = "DB_USER"
          valueFrom = "${data.aws_secretsmanager_secret.db.arn}:DB_USER::"
        },
        {
          name      = "DB_PASSWORD"
          valueFrom = "${data.aws_secretsmanager_secret.db.arn}:DB_PASSWORD::"
        },
        {
          name      = "DB_PORT"
          valueFrom = "${data.aws_secretsmanager_secret.db.arn}:DB_PORT::"
        },
        {
          name      = "BOSS_EMAIL"
          valueFrom = "${data.aws_secretsmanager_secret.boss.arn}:email::"
        },
        {
          name      = "BOSS_PASSWORD"
          valueFrom = "${data.aws_secretsmanager_secret.boss.arn}:password::"
        },
        {
          name      = "BOSS_MFA_SECRET"
          valueFrom = "${data.aws_secretsmanager_secret.boss.arn}:mfa_secret::"
        },
        {
          name      = "GEMINI_API_KEY"
          valueFrom = "${data.aws_secretsmanager_secret.api_keys.arn}:gemini_api_key::"
        },
        {
          name      = "SUPABASE_URL"
          valueFrom = "${data.aws_secretsmanager_secret.api_keys.arn}:supabase_url::"
        },
        {
          name      = "SUPABASE_SERVICE_KEY"
          valueFrom = "${data.aws_secretsmanager_secret.api_keys.arn}:supabase_service_key::"
        },
        {
          name      = "SENTRY_DSN"
          valueFrom = "${data.aws_secretsmanager_secret.api_keys.arn}:sentry_dsn::"
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

# =============================================================================
# DATA SOURCES FOR SECRETS
# =============================================================================

data "aws_secretsmanager_secret" "db" {
  name = var.db_secret_name
}

data "aws_secretsmanager_secret" "boss" {
  name = var.boss_secret_name
}

data "aws_secretsmanager_secret" "api_keys" {
  name = var.api_keys_secret_name
}
