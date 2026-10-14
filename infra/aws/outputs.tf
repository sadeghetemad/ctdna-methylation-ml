output "bucket" {
  value = aws_s3_bucket.data.bucket
}

output "job_queue" {
  value = aws_batch_job_queue.main.name
}

output "job_role_arn" {
  value = aws_iam_role.job.arn
}

output "head_public_ip" {
  value = var.create_head_node ? aws_instance.head[0].public_ip : null
}

output "head_instance_id" {
  value = var.create_head_node ? aws_instance.head[0].id : null
}

output "ssm_command" {
  description = "Open a shell on the head node."
  value       = var.create_head_node ? "aws ssm start-session --region ${var.region} --target ${aws_instance.head[0].id}" : null
}

# terraform output -raw env_exports > ~/ctdna_aws.env && source ~/ctdna_aws.env
output "env_exports" {
  description = "Environment variables read by scripts/03_process_nfcore.py --aws-batch and scripts/aws_stage.py."
  value       = <<-EOT
    export AWS_REGION=${var.region}
    export CTDNA_S3_BUCKET=${aws_s3_bucket.data.bucket}
    export CTDNA_BATCH_QUEUE=${aws_batch_job_queue.main.name}
    export CTDNA_BATCH_JOB_ROLE=${aws_iam_role.job.arn}
  EOT
}
