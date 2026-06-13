# AWS Infrastructure — Terraform

Provisions everything needed to run the Churn API on AWS:

| Resource | Details |
|---|---|
| ECR repository | Stores Docker images, lifecycle policy keeps last 10 |
| EC2 t3.medium | Amazon Linux 2023, Docker pre-installed via user_data |
| Elastic IP | Stable public IP (set once as GitHub secret) |
| IAM OIDC role | GitHub Actions authenticates via OIDC — no stored AWS keys |
| Security group | Port 8000 (API) + optional port 22 (SSH) |

---

## Prerequisites

1. [Terraform ≥ 1.5](https://developer.hashicorp.com/terraform/install)
2. AWS CLI configured: `aws configure` (or use env vars `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`)
3. The GitHub OIDC provider already registered in your AWS account — run once:

```bash
aws iam create-open-id-connect-provider \
  --url https://token.actions.githubusercontent.com \
  --client-id-list sts.amazonaws.com \
  --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1
```

---

## 5-Step Bootstrap

### Step 1 — Set variables

Create `infrastructure/terraform.tfvars`:

```hcl
github_org        = "your-github-username"
github_repo       = "customer-churn-prediction"
aws_region        = "us-east-1"
ec2_key_name      = "my-keypair"   # existing EC2 key pair name (optional)
allowed_ssh_cidr  = "1.2.3.4/32"  # your IP for SSH
```

### Step 2 — Init & plan

```bash
cd infrastructure
terraform init
terraform plan -out=tfplan
```

### Step 3 — Apply

```bash
terraform apply tfplan
```

After ~2 minutes you will see the `github_secrets_summary` output with all four values.

### Step 4 — Copy outputs to GitHub

Go to `https://github.com/<you>/customer-churn-prediction/settings/secrets/actions` and add:

| Secret | Value (from terraform output) |
|---|---|
| `AWS_IAM_ROLE_ARN` | `iam_role_arn` |
| `AWS_REGION` | `aws_region` |
| `ECR_REGISTRY` | `ecr_registry` |
| `EC2_HOST` | `ec2_public_ip` |
| `EC2_SSH_KEY` | Paste the private key of the key pair set in `ec2_key_name` |

### Step 5 — Copy trained model to EC2 & push

```bash
# Copy model artifacts to EC2
scp -i ~/.ssh/my-keypair.pem \
  models/churn_*_prod.joblib \
  models/preprocessor.joblib \
  models/feature_names.joblib \
  models/model_metadata.joblib \
  models/outlier_caps.joblib \
  ec2-user@<EC2_HOST>:/opt/churn/models/

# Push code → CI/CD builds Docker image, pushes to ECR, deploys to EC2
git push origin main
```

After the GitHub Actions workflow succeeds (check the Actions tab), visit:

```
http://<EC2_HOST>:8000/docs
```

---

## Tear Down

```bash
terraform destroy
```

---

## Cost Estimate

| Resource | Monthly Cost (us-east-1) |
|---|---|
| t3.medium (on-demand) | ~$30 |
| Elastic IP (attached) | Free |
| ECR storage (< 1 GB) | ~$0.10 |
| Data transfer | < $1 |
| **Total** | **~$31 / month** |

Use a t3.small (~$15/mo) for dev/demo workloads.
