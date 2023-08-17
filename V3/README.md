### V3 of the OpenAlex Concept Tagger Model

This page serves as a guide for how to use this part of the repository. You should only have to make minor changes to the code (adding API key files, changing file paths, updating configuration files, etc.) in order to make this code functional. The python package requirements file can be used for all python notebooks in this directory.

#### PLEASE READ BEFORE CONTINUING: 

This model is a carbon copy of the V2 model. The only change that was made in this model iteration is to make sure that every concept that was tagged also has a parent tagged so that there is a direct line from a single concept to top level of our concept graph. The exact logic is provided in 003_Deploy/model_to_api/container/mag_model/predictor.py but here is a high level overview:

For each concept that meets the probability score threshold in our model output, we look at all possible "chains" of concepts from that concept all the way to the top of the tree. For example, 'cancer' (level 2 concept), 'internal medicine' (level 1 concept) and 'medicine' (level 0 concept) are a chain for the 'cancer' concept. So if our model tagged 'cancer', this would be one of the chains that are possible to connect to the top level of the graph (medicine). For some concepts there may only be one chain but for a lot concepts, there will be many chains to get to the top level of the graph since many concepts have multiple parents. For each concept, the chain with the highest aggregate probability score (adding up all of the probability scores of all concepts within the chain) is chosen and all of the concepts within that chain are also tagged to the work. To see how the "ancestor chains" were created, see 002_Model/006_ancestor_chains.ipynb.

Because we are working with the MAG hierarchy for concepts, some of the concepts tagged through this extra "chain" process are very wrong. If this type of chaining is not important in your work, we recommend setting a threshold for the concept score of around 0.32 in order to filter out bad concepts that were tagged. In the future, we are hoping to improve our model and hierarchy together so that we have less errors in the concept tagging. However, by setting a threshold, you should see good results, similar to the results we found for the V2 model.


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
