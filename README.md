# MSDHGNN-Master

The code and weights have been released, enjoy it！ You can easily run the model！ To use the pretrained models at checkpoint/ and evaluate the model prediction and real-time performance, **run:  test.py!**

**Run test.py to test the average displacement error (ADE), final displacement error (FDE), and model parameters of the MSDHGNN (modeling only low-frequency components) and the variant of MSDHGNN (modeling both high- and low-frequency components) for various scenarios on the ETH/UCY dataset.**

If you can't run test.py correctly, please contact me in time! Email: chenwangxing@smail.sut.edu.cn


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
