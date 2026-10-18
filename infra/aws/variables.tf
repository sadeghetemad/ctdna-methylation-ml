variable "region" {
  description = "AWS region. Keep the S3 bucket, Batch and the head node in the same region."
  type        = string
  default     = "eu-west-2"
}

variable "project" {
  description = "Name prefix of every resource."
  type        = string
  default     = "ctdna"
}

variable "bucket_name" {
  description = "S3 bucket for FASTQ, reference, Nextflow work files and results. Empty: <project>-<account id>-<region>."
  type        = string
  default     = ""
}

variable "work_expiry_days" {
  description = "Nextflow work files under work/ are deleted after this many days (resume stops working after it)."
  type        = number
  default     = 30
}

variable "max_vcpus" {
  description = "Most vCPUs Batch may run at once. Also limited by your account's Spot vCPU quota (see README)."
  type        = number
  default     = 256
}

variable "ondemand_max_vcpus" {
  description = "Most vCPUs of the on-demand fallback queue (tasks reclaimed on Spot). 64 = 4 jobs at once."
  type        = number
  default     = 64
}

variable "instance_types" {
  description = "Instance types for the jobs. Only 16-vCPU sizes: one alignment (12 vCPU / 56 GB) per instance, so concurrent alignments cannot fill one disk."
  type        = list(string)
  default     = ["m6i.4xlarge", "m7i.4xlarge", "m6a.4xlarge", "m5.4xlarge", "m5a.4xlarge", "r6i.4xlarge", "r6a.4xlarge", "r5.4xlarge"]
}

variable "use_spot" {
  description = "Spot (about 70% cheaper, jobs are retried when reclaimed) or on-demand."
  type        = bool
  default     = true
}

variable "job_disk_gb" {
  description = "Root EBS volume (gp3) of every Batch instance. Bismark writes several converted copies of the reads; the largest samples need well over 100 GB."
  type        = number
  default     = 500
}

variable "subnet_ids" {
  description = "Subnets (with internet access) for Batch and the head node. Empty: all subnets of the default VPC."
  type        = list(string)
  default     = []
}

variable "create_head_node" {
  description = "Create a small EC2 instance that runs Nextflow and the feature steps 04-07. Not needed if you launch from another machine with AWS credentials."
  type        = bool
  default     = true
}

variable "head_instance_type" {
  description = "Instance type of the head node."
  type        = string
  default     = "m6i.xlarge"
}

variable "head_disk_gb" {
  description = "Head node disk. Holds the feature-step inputs synced from S3 (BAMs are about 1 GB per sample)."
  type        = number
  default     = 1000
}

variable "ssh_public_key" {
  description = "Public key (contents of id_ed25519.pub) to log in to the head node with SSH. Empty: no SSH key (use Session Manager)."
  type        = string
  default     = ""
}

variable "ssh_allowed_cidrs" {
  description = "CIDRs allowed to SSH to the head node, e.g. [\"203.0.113.7/32\"]. Required when ssh_public_key is set."
  type        = list(string)
  default     = []
}
