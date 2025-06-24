# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is a RandomForestClassifier that is trained to rpedict whether a person's income exceeds 50k/year based on census data featuers. It uses demographic and employment attributes such as education, workclass, occupation.

## Intended Use
This model is intended to assist in income prediction tasks based on U.S. census data. 
This can be used for socio-econimic studies, targeted marketing, or demographic analysis. It should not be used for making critical decisions without further validation.

## Training Data 
The model was trained on the U.S. cencus income dataset (Adult dataset).
This dataset includes demographic, educational, occupational andrealationship attributes collected from the 1994 census database. The training data consist of approximately 80% of the total dataset, randomly split while preserving the distribution of the income label. Categorical features were processed using one-hot encoding, and the labels were binarized for classification.

## Evaluation Data
Evaluation was performed on the remaining 20% held out as a test set.This split ensured that the model's performance was assesed on data it had not seen during traininig. Additionaly, performance was analyzed on slices of the data grouped by various categorical features such as education, level, race, and sex, to assess model behaviour across different demographic groups.

## Metrics
The models performance was evaluated using the following metrics on the test dataset:
- Precision: The ratio of correctly predicted positive observations to total predicted positives. Overall precision: 0.7453
- Recall: The ratio of correctly predicted positive observation to all actual positives. Overall recall: 0.6307
-F1 Score (F-beta with beta=1): The harmonic mean of precision and recall to balance both metrics. Overall F1 score: 0.6832

## Ethical Considerations
The model uses sensitive demographic attributes, including sex, race, and education, which may reflect historical biases present in the training data. There is a risk of perpetuating or amplifying these biases in predictions, which could lead to unfair treatment to certain groups. It is important to use this model with caution, continously monitor for bias, and consider fairness audits before deployment.
## Caveats and Recommendations
- The models predictions are based on historical census data and may not generalize perfectly to current or future populations.
- Performance varies across demographics slices, highlighting the need for ongoing evaluation on protected groups.
- Users should combine model outputs with domain knowledge and human judgement, especially for high-stake decisions.
-Retraining with updated and more representative data is recommended to maintain fairness and accuracy over time.
