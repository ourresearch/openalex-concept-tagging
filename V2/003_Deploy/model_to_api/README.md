### Files Used to Create Sagemaker Instance and API

#### Container
Used to create a Docker container which was then uploaded to AWS and used to create a Sagemaker endpoint. The only things missing here are the model artifacts created from model development.

#### Local Test
After the container is created this builds a local copy of the container and allows for testing the model.

#### API Creation
Contains two of the files needed to create an API endpoint using chalice. In order to create a new chalice project, make sure chalice and boto3 are installed on your environment (can be done using pip). After, type "chalice new-project" and it will set up a new chalice directory where you can name your project. The app.py and the config.json file are the edited versions for this project. Once those files are created/configured, the AWS Lambda and the REST API are automatically created after typing "chalice deploy".
