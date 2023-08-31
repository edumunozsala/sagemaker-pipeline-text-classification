# And end-to-end AWS Sagemaker Pipeline to prepare data, train, evaluate and conditional deployment of a Keras Text Classification model

This repository contains the Jupyter notebooks and Python scripts to build a SageMaker Pipeline that includes the following stages:
- Data download and preprocessing
- Train a Keras model for text classification
- Evaluate the model on the validation dataset to get the accuracy
- A condition step to check if the accuracy is acceptable
- Register and create the model in Sagemaker
- Create a Lambda function to deploy the model to an endpoint for future prediction
- If accuracy is worse, then a failed step to notify the result

We are not interested in the model architecture, we are not searching for the "best model" what we want to show here is how to build a Sagemaker pipeline to automate this complex process. MLOps is one of the challenges in the machine learning community, this is a simple example of how can you achieve automation in an ML project.

Most of the inspiration and code examples can be downloaded from the sagemaker examples repository, more specifically from the notebook [SageMaker Pipelines California Housing - Taking different steps based on model performance](https://github.com/aws/amazon-sagemaker-examples/blob/main/sagemaker-pipelines/tabular/tensorflow2-california-housing-sagemaker-pipelines-deploy-endpoint/tensorflow2-california-housing-sagemaker-pipelines-deploy-endpoint.ipynb).

## The dataset

We will be using the popular dataset [IMDB movie review sentiment](https://ai.stanford.edu/~amaas/data/sentiment/).

This is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets. We provide a set of 25,000 highly polar movie reviews for training, and 25,000 for testing. There is additional unlabeled data for use as well. Raw text and already processed bag of words formats are provided.


## Problem description

Our goal is to define, build, and run a machine learning pipeline, end-to-end, that we can run periodically to prepare the data, upload it to AWS S3, build, and train a keras model, evaluate the model on the validation dataset and finally if the performance or accuracy of the model is greater than a defined threshold deploy the model in an endpoint for future predictions.

**Note:** In this simple example the training and validation dataset is not going to be modify so probably no retraining is needed. But you can adjust this example to your own problem and dataset that requires retraining.

## The model

The code for model building and training has been extracted from a Keras tutorial ["Text classification from scratch"](https://keras.io/examples/nlp/text_classification_from_scratch/). That's all, it is a simple text classifier with the layers: Embedding layer, two Convolutional 1D layers, a MaxPooling layer and a final Dense layer. To tokenize the text, the model defines a custom vectorize layer and how to save this layer to build the model in a sagemaker container was the only remarkable point.

We use this simple model to drastically reduce the training time and compute power required, for the sake of saving money and time.

## Content

- text-classification-keras-local notebook: Here we download, prepare, build and train the model locally withour using sagemaker capabilities.
- sm-pipeline-text-classification-keras-batch: the pipeline prepare, train, evluate and run a Batch Transform job
- sm-pipeline-text-classification-keras-deploy: This pipeline deploy the model to an endpoint instead of running a Batch Transform job

## Contributing
If you find some bug or typo, please let me know or fixit and push it to be analyzed. 

## License

These notebooks are under a public GNU License.