### Files Used to Create Sagemaker Instance and API

#### Container
Used to create a Docker container which was then uploaded to AWS and used to create a Sagemaker endpoint. The only things missing here are the model artifacts created from model development.

#### Local Test
After the container is created this builds a local copy of the container and allows for testing the model.

#### API Creation
Contains two of the files needed to create an API endpoint using chalice. For security reasons, some files had to be excluded. To get the complete set of files, 
