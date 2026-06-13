variable "aws_region" {
  description = "AWS region to deploy into"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Short identifier used in all resource names"
  type        = string
  default     = "customer-churn"
}

variable "github_org" {
  description = "GitHub organisation (or user) that owns the repo"
  type        = string
}

variable "github_repo" {
  description = "GitHub repository name (without the org prefix)"
  type        = string
  default     = "customer-churn-prediction"
}

variable "ec2_instance_type" {
  description = "EC2 instance type for the API server"
  type        = string
  default     = "t2.micro"   # free tier eligible (12 months on new accounts)
}

variable "ec2_key_name" {
  description = "Name of an existing EC2 key pair for SSH access (leave empty to skip SSH)"
  type        = string
  default     = ""
}

variable "allowed_ssh_cidr" {
  description = "CIDR block allowed SSH access (your IP). Use 0.0.0.0/0 only for testing."
  type        = string
  default     = "0.0.0.0/0"
}

variable "api_port" {
  description = "Port the FastAPI container listens on"
  type        = number
  default     = 8000
}

variable "ecr_image_tag_mutability" {
  description = "ECR image tag mutability (MUTABLE or IMMUTABLE)"
  type        = string
  default     = "MUTABLE"
}

variable "environment" {
  description = "Deployment environment tag"
  type        = string
  default     = "production"
}
