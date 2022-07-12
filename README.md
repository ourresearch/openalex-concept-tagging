# openalex-concept-tagging

This repository contains all of the code for getting the [OpenAlex](https://openalex.org) concept tagger up and running. Go into the model iteration directory (V1 or V2) to find a more detailed explanation of how to use this repository. To learn more about concepts in OpenAlex, check out [the docs](https://docs.openalex.org/about-the-data/concept). 

### Model Iterations
* V1 (no longer used)
* V2 (currently used)

Both a V1 and a V2 model were created but as of right now, the V2 model is being used in OpenAlex. Initially, abstract data was not available for the model so we went with a V1 model that only looked at paper titles and a few other features. Once paper abstract data became available, a V2 model was created and we saw a substantial increase in performance.

### Model Development
You can find an explanation of the modeling and deployment process at the following link:
[OpenAlex: End-to-End Process for Concept Tagging](https://docs.google.com/document/d/1q3jBlEexskCZaSafFDMEEY3naTeyd7GS/edit?usp=sharing&ouid=112616748913247881031&rtpof=true&sd=true)

### Concepts
Input can be tagged with one or more of about 65,000 concepts, [listed here](https://docs.google.com/spreadsheets/d/1LBFHjPt4rj_9r0t0TTAlT68NwOtNH8Z21lBMsJDMoZg/edit?usp=sharing). Concepts are part of a hierarchical tree, with levels 0 (e.g., Mathematics) through 5 (e.g., Generalized inverse Gaussian distribution). 
