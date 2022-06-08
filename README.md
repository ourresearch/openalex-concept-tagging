# openalex-concept-tagging

This repository contains all of the code for getting the concept tagger up and running. Go into the model iteration directory (V1 or V2) to find a more detailed explanation of how to use this repository. To learn more about where this model is being used, feel free to visit: https://openalex.org/

### Model Iterations
* V1 (complete)
* V2 (current)

Both a V1 and a V2 model were created but as of right now, the V2 model is being used in OpenAlex. Initially, abstract data was not available for the model so we went with a V1 model that only looked at paper titles and a few other features. Once paper abstract data became available, a V2 model was created and we saw a substantial increase in performance.

### Model Development
You can find an explanation of the modeling and deployment process at the following link:
[OpenAlex: End-to-End Process for Concept Tagging](https://docs.google.com/document/d/1q3jBlEexskCZaSafFDMEEY3naTeyd7GS/edit?usp=sharing&ouid=112616748913247881031&rtpof=true&sd=true)
