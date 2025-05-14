# LLM Project

## Project Task
For this project, I decided to focus on sentiment analysis. The idea was to use the model to determine whether a movie review was positive or negative based on the text.

## Dataset
The dataset used was the [IMBD Dataset](https://huggingface.co/datasets/stanfordnlp/imdb) by Standford NLP.

## Pre-trained Model
The base model chosen for the task was [Distilbert](https://huggingface.co/distilbert/distilbert-base-uncased)

## Performance Metrics
Performance was measured using accuracy, precision, F1 score and ROC-AUC score.

The following are the results:
- Loss: 0.1963
- Accuracy: 0.9257
- Precision: 0.9243
- F1: 0.9258
- Roc Auc: 0.9257

## Hyperparameters
I used Optuna to finetune the hyperparameters for the trained model. To keep compute requirements down I optimised learning rate, batch size and number of epochs. Learning rate and number of epochs were the most important out of the three I tested.

## Model
The [model](https://huggingface.co/Gur212/LHL_LLM_Project) is available to use on HuggingFace.
