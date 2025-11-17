project_name = "health-forge-ehr-diff"
region = "us-east-2" 

## Change instance types amd volume size for SageMaker if desired
training_instance_type = "ml.g5.xlarge"
inference_instance_type = "ml.g5.xlarge"
volume_size_sagemaker = 5

## Should not be changed with the current folder structure
handler_path  = "../../src/lambda_function"
handler       = "config_lambda.lambda_handler"

