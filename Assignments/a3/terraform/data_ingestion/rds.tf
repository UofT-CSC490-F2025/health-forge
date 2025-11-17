# RDS SECRETS
data "aws_secretsmanager_secret" "raw_db_creds" {
  name = "raw_data_rds_credentials"
}

data "aws_secretsmanager_secret_version" "raw_db_creds_current" {
  secret_id = data.aws_secretsmanager_secret.raw_db_creds.id
}

locals {
  db_creds = jsondecode(data.aws_secretsmanager_secret_version.raw_db_creds_current.secret_string)
}

# MAIN VPC
data "aws_vpc" "default" {
  default = true
}

# LOOK UP EXISTING SUBNETS
data "aws_subnets" "default_vpc_subnets" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# RDS SUBNET

resource "aws_db_subnet_group" "default_vpc_db" {
  name        = "default-vpc-db-subnets"
  subnet_ids  = data.aws_subnets.default_vpc_subnets.ids
  description = "DB subnets in default VPC"
}

# #RDS SECURITY GROUP

resource "aws_security_group" "rds_sg" {
  vpc_id = data.aws_vpc.default.id
}

resource "aws_vpc_security_group_egress_rule" "rds_egress" {
  security_group_id = aws_security_group.rds_sg.id
  cidr_ipv4         = "0.0.0.0/0"
  ip_protocol       = "-1"
}

# RDS
resource "aws_db_instance" "raw_data_store" {
  allocated_storage = 1000
  db_name           = "raw_training_data"
  engine            = "postgres"
  engine_version    = "15"
  instance_class    = "db.t3.micro"

  # Fetch credentials from AWS Secrets Manager
  username            = local.db_creds.username
  password            = local.db_creds.password
  skip_final_snapshot = true

  # Networking
  db_subnet_group_name   = aws_db_subnet_group.default_vpc_db.name
  vpc_security_group_ids = [aws_security_group.rds_sg.id]
  publicly_accessible    = false

}
