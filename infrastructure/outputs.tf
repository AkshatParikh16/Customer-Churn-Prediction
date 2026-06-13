output "ec2_public_ip" {
  description = "Elastic IP of the API server — use as EC2_HOST GitHub secret"
  value       = aws_eip.churn_api.public_ip
}

output "ecr_registry" {
  description = "ECR registry URL (without the repo) — use as ECR_REGISTRY GitHub secret"
  value       = "${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com"
}

output "ecr_repository_url" {
  description = "Full ECR repository URL (registry + repo name)"
  value       = aws_ecr_repository.churn.repository_url
}

output "iam_role_arn" {
  description = "GitHub Actions OIDC role ARN — use as AWS_IAM_ROLE_ARN GitHub secret"
  value       = aws_iam_role.github_oidc.arn
}

output "aws_region" {
  description = "Deployed region — use as AWS_REGION GitHub secret"
  value       = var.aws_region
}

output "github_secrets_summary" {
  description = "Copy these values into GitHub Settings → Secrets → Actions"
  value = <<-EOT
  ┌──────────────────────────────────────────────────────────────────┐
  │  GitHub Actions Secrets                                          │
  ├────────────────────────┬─────────────────────────────────────────┤
  │  AWS_IAM_ROLE_ARN      │  ${aws_iam_role.github_oidc.arn}
  │  AWS_REGION            │  ${var.aws_region}
  │  ECR_REGISTRY          │  ${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com
  │  EC2_HOST              │  ${aws_eip.churn_api.public_ip}
  │  EC2_SSH_KEY           │  (paste your private key PEM)
  └────────────────────────┴─────────────────────────────────────────┘
  EOT
}
