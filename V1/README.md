### V1 of the OpenAlex Concept Tagger Model

This page serves as a guide for how to use this part of the repository. You should only have to make minor changes to the code (adding API key files, changing file paths, updating configuration files, etc.) in order to make this code functional. The python package requirements file can be used for all python notebooks in this directory.

#### 001 Exploration

Use the notebook in this directory to see how the data was queried and explored.

#### 002 Model

Use the notebooks in this directory if you would like to train a model from scratch using the same methods as OpenAlex. These notebooks using both Spark (Databricks) and Jupyter notebooks so make sure you are in the right environment. The notebooks progress sequentially so start at notebook 001 and go in order from there.

#### 003 Deploy

Use the notebooks in this directory if you would like to deploy the model locally or in AWS. The model artifacts will need to be downloaded into the appropriate folder before the model can be deployed, so make sure to follow the instructions in the "NOTES" section below to get the model artifacts.


### NOTES
#### Model Artifacts

In order to deploy the model/container in AWS or deploy the model/container locally, you will need the model artifacts. These can be downloaded from our S3 bucket using the AWS CLI. Once the AWS CLI is installed and set up for your account (aws configure), run the following command:

```bash
aws s3 cp s3://openalex-concept-tagger-model-files/ . --recursive
```

The files in this repository can be used to transform the input data correctly and load the model files. Both the V1 and V2 model files are kept at this location so make sure to use the V2 model if you want the latest (best) model.
