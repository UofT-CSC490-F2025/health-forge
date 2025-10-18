# S3 BUCKET FOR INITIAL DATA UPLOAD

resource "aws_s3_bucket" "raw" {
  bucket = "raw_bucket"
}
