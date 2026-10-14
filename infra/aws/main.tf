data "aws_caller_identity" "current" {}

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

data "aws_ssm_parameter" "al2023" {
  name = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
}

locals {
  bucket     = var.bucket_name != "" ? var.bucket_name : "${var.project}-${data.aws_caller_identity.current.account_id}-${var.region}"
  subnet_ids = length(var.subnet_ids) > 0 ? var.subnet_ids : data.aws_subnets.default.ids
}

# ------------------------------------------------------------
# S3
# ------------------------------------------------------------

resource "aws_s3_bucket" "data" {
  bucket = local.bucket
}

resource "aws_s3_bucket_public_access_block" "data" {
  bucket                  = aws_s3_bucket.data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "data" {
  bucket = aws_s3_bucket.data.id

  rule {
    id     = "expire-nextflow-work"
    status = "Enabled"
    filter { prefix = "work/" }
    expiration { days = var.work_expiry_days }
  }

  rule {
    id     = "abort-incomplete-uploads"
    status = "Enabled"
    filter {}
    abort_incomplete_multipart_upload { days_after_initiation = 3 }
  }
}

data "aws_iam_policy_document" "s3_access" {
  statement {
    actions   = ["s3:ListBucket", "s3:GetBucketLocation", "s3:ListBucketMultipartUploads"]
    resources = [aws_s3_bucket.data.arn]
  }

  statement {
    actions = [
      "s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:AbortMultipartUpload",
      "s3:ListMultipartUploadParts", "s3:GetObjectTagging", "s3:PutObjectTagging",
    ]
    resources = ["${aws_s3_bucket.data.arn}/*"]
  }
}

# ------------------------------------------------------------
# IAM: Batch service, EC2 instances, job containers
# ------------------------------------------------------------

data "aws_iam_policy_document" "batch_trust" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["batch.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "batch_service" {
  name               = "${var.project}-batch-service"
  assume_role_policy = data.aws_iam_policy_document.batch_trust.json
}

resource "aws_iam_role_policy_attachment" "batch_service" {
  role       = aws_iam_role.batch_service.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSBatchServiceRole"
}

data "aws_iam_policy_document" "ec2_trust" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "ecs_instance" {
  name               = "${var.project}-batch-instance"
  assume_role_policy = data.aws_iam_policy_document.ec2_trust.json
}

resource "aws_iam_role_policy_attachment" "ecs_instance" {
  role       = aws_iam_role.ecs_instance.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"
}

resource "aws_iam_role_policy" "ecs_instance_s3" {
  name   = "s3"
  role   = aws_iam_role.ecs_instance.id
  policy = data.aws_iam_policy_document.s3_access.json
}

resource "aws_iam_instance_profile" "ecs_instance" {
  name = "${var.project}-batch-instance"
  role = aws_iam_role.ecs_instance.name
}

data "aws_iam_policy_document" "ecs_tasks_trust" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

# Role of the containers: reads and writes the bucket (the aws cli staging the task files)
resource "aws_iam_role" "job" {
  name               = "${var.project}-batch-job"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_trust.json
}

resource "aws_iam_role_policy" "job_s3" {
  name   = "s3"
  role   = aws_iam_role.job.id
  policy = data.aws_iam_policy_document.s3_access.json
}

# ------------------------------------------------------------
# Batch
# ------------------------------------------------------------

resource "aws_security_group" "batch" {
  name        = "${var.project}-batch"
  description = "Batch compute: outbound only"
  vpc_id      = data.aws_vpc.default.id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# Larger root disk, and the aws cli that Nextflow needs on the host to stage
# files to and from S3 (the nf-core containers do not have it).
resource "aws_launch_template" "batch" {
  name = "${var.project}-batch"

  block_device_mappings {
    device_name = "/dev/xvda"
    ebs {
      volume_size           = var.job_disk_gb
      volume_type           = "gp3"
      throughput            = 250
      delete_on_termination = true
      encrypted             = true
    }
  }

  user_data = base64encode(<<-EOT
    MIME-Version: 1.0
    Content-Type: multipart/mixed; boundary="==BOUNDARY=="

    --==BOUNDARY==
    Content-Type: text/x-shellscript; charset="us-ascii"

    #!/bin/bash
    set -e
    curl -sSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -f -p /home/ec2-user/miniconda
    /home/ec2-user/miniconda/bin/conda install -q -y -c conda-forge awscli
    chmod -R a+rX /home/ec2-user/miniconda
    rm -f /tmp/miniconda.sh

    --==BOUNDARY==--
  EOT
  )

  metadata_options {
    http_tokens                 = "required"
    http_put_response_hop_limit = 2
  }
}

resource "aws_batch_compute_environment" "main" {
  compute_environment_name = "${var.project}-methylseq"
  type                     = "MANAGED"
  service_role             = aws_iam_role.batch_service.arn

  compute_resources {
    type                = var.use_spot ? "SPOT" : "EC2"
    allocation_strategy = var.use_spot ? "SPOT_PRICE_CAPACITY_OPTIMIZED" : "BEST_FIT_PROGRESSIVE"
    min_vcpus           = 0
    desired_vcpus       = 0
    max_vcpus           = var.max_vcpus
    instance_type       = var.instance_types
    subnets             = local.subnet_ids
    security_group_ids  = [aws_security_group.batch.id]
    instance_role       = aws_iam_instance_profile.ecs_instance.arn

    launch_template {
      launch_template_id = aws_launch_template.batch.id
      version            = aws_launch_template.batch.latest_version
    }
  }

  depends_on = [aws_iam_role_policy_attachment.batch_service]

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_batch_job_queue" "main" {
  name     = "${var.project}-methylseq"
  state    = "ENABLED"
  priority = 1

  compute_environment_order {
    order               = 1
    compute_environment = aws_batch_compute_environment.main.arn
  }
}

# ------------------------------------------------------------
# Head node: runs Nextflow, the FASTQ staging and steps 04-07
# ------------------------------------------------------------

data "aws_iam_policy_document" "head" {
  statement {
    actions   = ["batch:*"]
    resources = ["*"]
  }

  statement {
    actions = [
      "ecs:DescribeContainerInstances", "ecs:DescribeTasks", "ec2:DescribeInstances",
      "logs:GetLogEvents", "logs:DescribeLogStreams",
    ]
    resources = ["*"]
  }

  statement {
    actions   = ["iam:PassRole"]
    resources = [aws_iam_role.job.arn]
  }
}

resource "aws_iam_role" "head" {
  count              = var.create_head_node ? 1 : 0
  name               = "${var.project}-head"
  assume_role_policy = data.aws_iam_policy_document.ec2_trust.json
}

resource "aws_iam_role_policy" "head" {
  count  = var.create_head_node ? 1 : 0
  name   = "batch-and-s3"
  role   = aws_iam_role.head[0].id
  policy = data.aws_iam_policy_document.head.json
}

resource "aws_iam_role_policy" "head_s3" {
  count  = var.create_head_node ? 1 : 0
  name   = "s3"
  role   = aws_iam_role.head[0].id
  policy = data.aws_iam_policy_document.s3_access.json
}

# Shell access through Session Manager: no SSH key, no open port
resource "aws_iam_role_policy_attachment" "head_ssm" {
  count      = var.create_head_node ? 1 : 0
  role       = aws_iam_role.head[0].name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "head" {
  count = var.create_head_node ? 1 : 0
  name  = "${var.project}-head"
  role  = aws_iam_role.head[0].name
}

resource "aws_key_pair" "head" {
  count      = var.create_head_node && var.ssh_public_key != "" ? 1 : 0
  key_name   = "${var.project}-head"
  public_key = var.ssh_public_key
}

resource "aws_security_group" "head" {
  count       = var.create_head_node ? 1 : 0
  name        = "${var.project}-head"
  description = "Head node: outbound, SSH only from ssh_allowed_cidrs"
  vpc_id      = data.aws_vpc.default.id

  dynamic "ingress" {
    for_each = var.ssh_public_key != "" && length(var.ssh_allowed_cidrs) > 0 ? [1] : []
    content {
      from_port   = 22
      to_port     = 22
      protocol    = "tcp"
      cidr_blocks = var.ssh_allowed_cidrs
    }
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "head" {
  count                  = var.create_head_node ? 1 : 0
  ami                    = data.aws_ssm_parameter.al2023.value
  instance_type          = var.head_instance_type
  subnet_id              = local.subnet_ids[0]
  vpc_security_group_ids = [aws_security_group.head[0].id]
  iam_instance_profile   = aws_iam_instance_profile.head[0].name
  key_name               = var.ssh_public_key != "" ? aws_key_pair.head[0].key_name : null

  root_block_device {
    volume_size = var.head_disk_gb
    volume_type = "gp3"
    encrypted   = true
  }

  metadata_options {
    http_tokens = "required"
  }

  user_data = file("${path.module}/head_user_data.sh")

  tags = {
    Name = "${var.project}-head"
  }
}
