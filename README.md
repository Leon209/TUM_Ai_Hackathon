# TUM_Ai_Hackathon

# Preprocessing
The time series heart rate data was resampled and transformed into features for model training.

# Model
Two approaches were used. One is based on statistical distribution (eliptic envelope) and the other one is based on classical maschine learning techniques (Decision Trees).

# Deployment
The model is deployed in a Microsoft Azure Pipeline with an REST access point.
The health tech app, which use the model API can be used on this [website](https://webflow.com/design/health-ai).
