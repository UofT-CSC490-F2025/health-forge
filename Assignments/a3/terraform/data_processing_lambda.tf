# LAMBDA SET UP
data "aws_iam_policy_document" "assume_role" {
  statement {
    effect = "Allow"

    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }

    actions = ["sts:AssumeRole"]
  }
}

resource "aws_iam_role" "data_processing_handler_role" {
  name               = "lambda_execution_role"
  assume_role_policy = data.aws_iam_policy_document.assume_role.json
}

resource "aws_security_group" "lambda_sg" { # Security group for the lambda
  vpc_id = data.aws_vpc.default.id
}

# Lambda Role Permissions

resource "aws_iam_role_policy_attachment" "lambda_basic_logs" {
  role       = aws_iam_role.data_processing_handler_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy_attachment" "lambda_vpc_access" {
  role       = aws_iam_role.data_processing_handler_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaVPCAccessExecutionRole"
}

data "aws_iam_policy_document" "s3_write" { #Allows the lambda to write to the data lakehouse
  statement {
    effect = "Allow"
    actions = [
      "s3:PutObject"
    ]
    resources = [
      "arn:aws:s3:::my-bucket-name/*"  # <-- only write to objects in this bucket
    ]
  }
}
resource "aws_iam_role_policy" "allow_s3_write" {
  name   = "allow-s3-write"
  role   = aws_iam_role.data_processing_handler_role.id
  policy = data.aws_iam_policy_document.s3_write.json
}

# --- Package Lambda code into a ZIP ---
data "archive_file" "lambda_zip" {
  type        = "zip"
  source_file = "${path.module}/lambda/MIMC-data-processing.py"
  output_path = "${path.module}/lambda/mimic.zip"
}

# --- Lambda function in the VPC ---
resource "aws_lambda_function" "data_processing_handler" {
  function_name = "data_handler_MIMIC"
  role          = aws_iam_role.data_processing_handler_role.arn
  runtime       = "python3.11"

  filename         = data.archive_file.lambda_zip.output_path
  handler          = "MIMC-data-processing.handler" # module.function in the ZIP

  vpc_config {
    subnet_ids         = data.aws_subnets.default_vpc_subnets.ids
    security_group_ids = [aws_security_group.lambda_sg.id]
  }

  environment {
    variables = {
      DB_HOST = aws_db_instance.raw_data_store.address
      DB_USER = local.db_creds.username
      DB_PASS = local.db_creds.password
      DB_NAME = "raw_training_data"
    }
  }
}