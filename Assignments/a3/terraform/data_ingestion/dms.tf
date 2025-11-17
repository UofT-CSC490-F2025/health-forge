
#Migration Logic
locals {
  # Minimal example: include everything; refine as you add per-table mappings.
  dms_table_mappings = {
    rules = [
      {
        "rule-type"      = "selection",
        "rule-id"        = "1",
        "rule-name"      = "include-all",
        "object-locator" = { "schema-name" = "%", "table-name" = "%" },
        "rule-action"    = "include"
      }
    ]
  }

  etd = file("${path.module}/mimiciv_hosp_external_table_definition.json")
}

# Allow DMS to read from the raw bucket
resource "aws_iam_role" "dms_s3_access" { // Define a role that will be able to access the s3
  name = "dms-s3-access-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Effect    = "Allow",
      Principal = { Service = "dms.amazonaws.com" },
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "dms_s3_read" { // Apply the perms themselves to the role
  role = aws_iam_role.dms_s3_access.id
  policy = jsonencode({
    Version : "2012-10-17",
    Statement : [
      # Needed to list the bucket and discover objects (and region)
      {
        Effect : "Allow",
        Action : [
          "s3:ListBucket",
          "s3:GetBucketLocation"
        ],
        Resource : [
          "${aws_s3_bucket.raw.arn}",
          "${aws_s3_bucket.raw.arn}/*"
        ]
      },
      # Needed to read the source files and the external table definition JSON
      {
        Effect : "Allow",
        Action : [
          "s3:GetObject",
          "s3:GetObjectVersion"
        ],
        Resource : [
          "${aws_s3_bucket.raw.arn}",
          "${aws_s3_bucket.raw.arn}/*"
        ]
      }
    ]
  })
}

resource "aws_iam_service_linked_role" "dms_slr" {
  aws_service_name = "dms.amazonaws.com"
}

resource "aws_iam_role" "dms-vpc-role" {
  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Effect    = "Allow",
      Principal = { Service = "dms.amazonaws.com" },
      Action    = "sts:AssumeRole"
    }]
  })
  name = "dms-vpc-role"
}

resource "aws_iam_role_policy_attachment" "dms-vpc-role-AmazonDMSVPCManagementRole" {
  role       = aws_iam_role.dms-vpc-role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonDMSVPCManagementRole"
}

# Security group used by DMS replication instance
resource "aws_security_group" "dms_sg" { //The DMS instance has to live inside a security group
  name   = "dms-sg"
  vpc_id = data.aws_vpc.default.id
}

resource "aws_vpc_security_group_egress_rule" "dms_egress" {
  security_group_id = aws_security_group.dms_sg.id
  cidr_ipv4         = "0.0.0.0/0"
  ip_protocol       = "-1"

}

# Allow DMS -> RDS on 5432
resource "aws_security_group_rule" "rds_ingress_from_dms" {
  type                     = "ingress"
  security_group_id        = aws_security_group.rds_sg.id # your existing RDS SG
  from_port                = 5432
  to_port                  = 5432
  protocol                 = "tcp"
  source_security_group_id = aws_security_group.dms_sg.id
}

resource "aws_dms_replication_instance" "ri" {
  replication_instance_id    = "s3-to-rds-ri"
  replication_instance_class = "dms.t3.small"
  allocated_storage          = 50
  vpc_security_group_ids     = [aws_security_group.dms_sg.id]
  depends_on                 = [aws_iam_role_policy_attachment.dms-vpc-role-AmazonDMSVPCManagementRole, aws_iam_role_policy.dms_s3_read, aws_iam_role.dms-vpc-role, aws_iam_role_policy_attachment.dms-vpc-role-AmazonDMSVPCManagementRole, aws_iam_service_linked_role.dms_slr]
  publicly_accessible        = true
}

# Source: S3 endpoint
resource "aws_dms_s3_endpoint" "src_s3" {
  endpoint_id             = "src-s3"
  endpoint_type           = "source"
  bucket_name             = aws_s3_bucket.raw.bucket
  service_access_role_arn = aws_iam_role.dms_s3_access.arn

  # Choose the correct format for your files:
  data_format               = "csv"
  csv_delimiter             = ","
  csv_row_delimiter         = "\n"
  compression_type          = "NONE"
  ignore_header_rows                          = 1
  external_table_definition = file("${path.module}/mimiciv_hosp_external_table_definition.json")
}

# Target: RDS Postgres endpoint
resource "aws_dms_endpoint" "tgt_rds" {
  endpoint_id   = "tgt-rds"
  endpoint_type = "target"
  engine_name   = "postgres"

  server_name   = aws_db_instance.raw_data_store.address
  port          = 5432
  database_name = "raw_training_data"
  username      = local.db_creds.username
  password      = local.db_creds.password
  ssl_mode      = "require"
}



resource "aws_dms_replication_task" "s3_to_rds" {
  replication_task_id      = "s3-to-rds-task"
  migration_type           = "full-load" # or "full-load-and-cdc" later
  replication_instance_arn = aws_dms_replication_instance.ri.replication_instance_arn
  source_endpoint_arn      = aws_dms_s3_endpoint.src_s3.endpoint_arn
  target_endpoint_arn      = aws_dms_endpoint.tgt_rds.endpoint_arn

  table_mappings = jsonencode(local.dms_table_mappings)

  replication_task_settings = <<SETTINGS
{
  "TargetMetadata": {
    "TargetSchema": "public",
    "SupportLobs": true,
    "FullLobMode": false
  },
  "FullLoadSettings": {
    "TargetTablePrepMode": "DO_NOTHING",
    "MaxFullLoadSubTasks": 8
  },
  "Logging": {
    "EnableLogging": true,
    "LogComponents": [
      { "Id": "SOURCE_UNLOAD",  "Severity": "LOGGER_SEVERITY_DEFAULT" },
      { "Id": "SOURCE_CAPTURE", "Severity": "LOGGER_SEVERITY_DEFAULT" },
      { "Id": "TARGET_LOAD",    "Severity": "LOGGER_SEVERITY_DEFAULT" },
      { "Id": "TARGET_APPLY",   "Severity": "LOGGER_SEVERITY_DEFAULT" },
      { "Id": "TASK_MANAGER",   "Severity": "LOGGER_SEVERITY_DEFAULT" }
    ]
  },
  "ErrorBehavior": {
    "DataErrorPolicy": "LOG_ERROR",
    "ApplyErrorDeletePolicy": "IGNORE_RECORD"
  }
}
SETTINGS



}

# Get your task ARN easily
locals {
  dms_task_arn = aws_dms_replication_task.s3_to_rds.replication_task_arn
}




