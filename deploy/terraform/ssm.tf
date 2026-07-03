# =============================================================================
# SSM PARAMETER STORE (Free tier, replaces Secrets Manager)
# =============================================================================
# Standard parameters are free (up to 10,000). Advanced are $0.05/param/month.
# We use Standard tier with SecureString for sensitive values.
# =============================================================================

# --- Database credentials ---
resource "aws_ssm_parameter" "db_host" {
  name  = "/${var.project_name}/DB_HOST"
  type  = "SecureString"
  value = var.ssm_db_host

  tags = { Name = "${var.project_name}-db-host" }
}

resource "aws_ssm_parameter" "db_name" {
  name  = "/${var.project_name}/DB_NAME"
  type  = "SecureString"
  value = var.ssm_db_name

  tags = { Name = "${var.project_name}-db-name" }
}

resource "aws_ssm_parameter" "db_user" {
  name  = "/${var.project_name}/DB_USER"
  type  = "SecureString"
  value = var.ssm_db_user

  tags = { Name = "${var.project_name}-db-user" }
}

resource "aws_ssm_parameter" "db_password" {
  name  = "/${var.project_name}/DB_PASSWORD"
  type  = "SecureString"
  value = var.ssm_db_password

  tags = { Name = "${var.project_name}-db-password" }
}

resource "aws_ssm_parameter" "db_port" {
  name  = "/${var.project_name}/DB_PORT"
  type  = "String"
  value = var.ssm_db_port

  tags = { Name = "${var.project_name}-db-port" }
}

# --- BOSS credentials ---
resource "aws_ssm_parameter" "boss_email" {
  name  = "/${var.project_name}/BOSS_EMAIL"
  type  = "SecureString"
  value = var.ssm_boss_email

  tags = { Name = "${var.project_name}-boss-email" }
}

resource "aws_ssm_parameter" "boss_password" {
  name  = "/${var.project_name}/BOSS_PASSWORD"
  type  = "SecureString"
  value = var.ssm_boss_password

  tags = { Name = "${var.project_name}-boss-password" }
}

resource "aws_ssm_parameter" "boss_mfa_secret" {
  name  = "/${var.project_name}/BOSS_MFA_SECRET"
  type  = "SecureString"
  value = var.ssm_boss_mfa_secret

  tags = { Name = "${var.project_name}-boss-mfa-secret" }
}

# --- API keys ---
resource "aws_ssm_parameter" "gemini_api_key" {
  name  = "/${var.project_name}/GEMINI_API_KEY"
  type  = "SecureString"
  value = var.ssm_gemini_api_key

  tags = { Name = "${var.project_name}-gemini-api-key" }
}

resource "aws_ssm_parameter" "supabase_url" {
  name  = "/${var.project_name}/SUPABASE_URL"
  type  = "String"
  value = var.ssm_supabase_url

  tags = { Name = "${var.project_name}-supabase-url" }
}

resource "aws_ssm_parameter" "supabase_service_key" {
  name  = "/${var.project_name}/SUPABASE_SERVICE_KEY"
  type  = "SecureString"
  value = var.ssm_supabase_service_key

  tags = { Name = "${var.project_name}-supabase-service-key" }
}

resource "aws_ssm_parameter" "sentry_dsn" {
  name  = "/${var.project_name}/SENTRY_DSN"
  type  = "SecureString"
  value = var.ssm_sentry_dsn

  tags = { Name = "${var.project_name}-sentry-dsn" }
}
