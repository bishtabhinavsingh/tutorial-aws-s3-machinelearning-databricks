# Setup an S3 bucket for Machine Learning on a remote / local instance.
This should help you setup an S3 bucket to train models on a databricks (or local) instance.

## Assumptions
- You know how to create an S3 bucket and configure the IAM on AWS.
- You know where to find your access keys, secret key, document key and bucket name.
- You have a zipped file that contains the test and train folders on S3.


## Method
This tutorial follows a batch method to download your file from S3. Where contents of the Zip file contain the entire batch of test-train files. We extract these files on the local (or remote) instance where a machine learning model will be trained. 

So basically, we just downloaded all files from S3 (in Zip) and then extract them anywhere we want. This could be an EC2 instance, Databricks, Google Collaboratory instance or local. A universal method meant to simply make use of S3 for training a model.

## Configuring connect()
This function handles the connection to S3 bucket, please note the 5 check points to configure the function to work with your S3 bucket.
<img width="1245" alt="1" src="https://user-images.githubusercontent.com/52534177/166829616-491d7cbb-2f10-49b9-9429-65d9226f1cba.png">

The 5 points are:
  - BUCKET_NAME
  - KEY
  - aws_accss_key
  - aws_secret_access_key
