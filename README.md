# openalex-concept-tagging

This repository contains all of the code for getting the concept tagger up and running. To learn more about where this model is being used, feel free to visit: https://openalex.org/

### Model Development
You can find an explanation of the modeling process at the following link:
[OpenAlex: End-to-End Process for Concept Tagging](https://docs.google.com/document/d/1q3jBlEexskCZaSafFDMEEY3naTeyd7GS/edit?usp=sharing&ouid=112616748913247881031&rtpof=true&sd=true)

If you would like to download the model files that are used in concept tagging, you are able to copy the files from our S3 location using the AWS CLI (must have an AWS account to access):

```bash
aws s3 cp s3://openalex-concept-tagger-model-files/ . --recursive
```

The files in this repository can be used to transform the input data correctly and load the model files. Both the V1 and V2 model files are kept at this location so make sure to use the V2 model if you want the latest (best) model.

### Model Iterations
* V1 (complete)
* V2 (current)
