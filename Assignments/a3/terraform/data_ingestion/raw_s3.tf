# S3 BUCKET FOR INITIAL DATA UPLOAD
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

resource "aws_s3_bucket" "raw" {
  bucket = "raw-bucket-${data.aws_caller_identity.current.account_id}-${data.aws_region.current.name}"
}
