# LLM Project

## Project Task
For this project, I decided to focus on sentiment analysis. The idea was to use the model to determine whether a movie review was positive or negative based on the text.

### Steps:
- preprocessed and analysed reviews with a normal machine learning algorithm, in this case XGBoost, for comparison 
- used a pre-trained model with no fine-tuning to get a baseline result
- performed transfer learning to fine-tune the model and compare how it performs against baseline
- performed hyperparamter tuning to further increase model scores
- retrained model with optimal hyperparameters

## Dataset
The dataset used was the [IMBD Dataset](https://huggingface.co/datasets/stanfordnlp/imdb) by Standford NLP.

## Pre-trained Model
The base model chosen for the task was [Distilbert](https://huggingface.co/distilbert/distilbert-base-uncased)

## Performance Metrics
Performance for hyperparameter tuning and the final model was measured using training loss, accuracy, precision, F1 score and ROC-AUC score. The previous models were measured using ROC-AUC score alone.

| Metric        | XGBoost | Pretrained Model | After Transfer Learning | After HP Tuning* |
|---------------|:-------:|:----------------:|:-----------------------:|:----------------:|
| ROC-AUC Score |  0.8532 |      0.5055      |         0.736487        |      0.6680      |
| Accuracy      |    -    |         -        |            -            |      0.9250      |
| Precision     |    -    |         -        |            -            |      0.9247      |
| F1 Score      |    -    |         -        |            -            |      0.9282      |
| Loss          |    -    |         -        |         0.05830         |      0.1959      |

*There was an error in the code to compute ROC-AUC scores. Due to time constraints, hyperparameter optimization was not run again after the fix yet, so final results will vary from this.

##

We can see from the results that pre-trained models can benefit greatly in classification tasks from transfer learning and hyperparameter tuning, and in this case was necessary to get any kind of result better than random guessing. What's also interesting is how well XGBoost fared in comparison. When combined with the much quicker and less intensive process to run, it shows that more traditional machine learning algorithms still have an important place.


## Hyperparameters
I used Optuna to finetune the hyperparameters for the trained model. To keep compute requirements down I optimised learning rate, batch size and number of epochs. Learning rate and number of epochs were the most important out of the three I tested, causing a larger variance of scores. A higher number of epochs always resulted in better scores, but came at the cost of taking much longer to complete. Within the range I tested, a higher learning rate also yielded better results when all else was equivalent. This means that I could likely improve results by expanding the search space further.

## Model
The [model](https://huggingface.co/Gur212/LHL_LLM_Project) is available to use on HuggingFace.

