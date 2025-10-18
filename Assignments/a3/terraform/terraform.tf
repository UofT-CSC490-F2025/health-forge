
#This file configures terraform itself
terraform {
  required_providers {
    aws = { #Define AWS parameter configs
      source  = "hashicorp/aws"
      version = "~> 5.92"
    }
  }

  required_version = ">= 1.2"
}



