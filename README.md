# MSDHGNN-Master




## Code Structure
checkpoint folder: contains the trained models, one of which is MSDHGNN (only low-frequency component modeling) and the other is a variant of MSDHGNN (high- and low-frequency component modeling)

dataset folder: contains ETH and UCY datasets

model.py: the code of MSDHGNN (modeling only low-frequency components)

LowHighmodel.py: the variant of MSDHGNN (modeling both high- and low-frequency components)

test.py: for testing the MSDHGNN (modeling only low-frequency components) and the variant of MSDHGNN (modeling both high- and low-frequency components)

utils.py: general utils used by the code

metrics.py: Measuring tools used by the code


## Model Evaluation
You can easily run the model！ To use the pretrained models at checkpoint/ and evaluate the models performance run:  test.py
