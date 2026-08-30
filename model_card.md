# Model Card

## Model Details
The machine learning model is a Random Forest Classifier trained on publicly available Census Bureau data. It was built using the scikit-learn library and is designed to be deployed as a web application utilizing the FastAPI package.

## Intended Use
The primary goal of this project is to develop a complete classification pipeline that predicts whether an individual's income exceeds $50,000 annually, monitors model performance across various demographic data slices, and serves those predictions via a FastAPI backend. This model serves as a technical demonstration of machine learning operations and is not intended for real-world automated decision-making in financial or employment contexts.

## Training Data
The model was trained on the Adult Census Income dataset using an 80% data split. The data processing pipeline handles continuous variables alongside categorical features, such as workclass, education, and occupation, which were transformed using One-Hot Encoding. The target variable was processed using a Label Binarizer.

## Evaluation Data
The model was evaluated using the remaining 20% test split from the Adult Census Income dataset. To fulfill the project requirements, code was implemented to monitor and evaluate model performance not just on the overall evaluation data, but also across specific categorical data slices to check for disparities.

## Metrics
Model performance was evaluated using Precision, Recall, and the F1-score. On the overall test set, the model achieved a Precision of 0.7419, a Recall of 0.6384, and an F1-score of 0.6863. To fulfill the project requirements, code was implemented to monitor performance on specific data slices. This slice monitoring showed that model accuracy varies; for example, predictions for individuals with a Bachelor's degree yielded an F1-score of 0.7404, while predictions for Master's degree holders achieved an F1-score of 0.8409.

## Ethical Considerations
Monitoring the data slices revealed notable performance disparities across protected demographic classes. Specifically, the model is noticeably less effective at identifying high-earning females, achieving a Recall of 0.5150 compared to 0.6599 for males.  

## Caveats and Recommendations
Several data slices contain insufficient sample sizes to provide statistically reliable metrics, such as the "Without-pay" workclass, which included only 4 test samples. Future iterations should address data imbalance and explore removing sensitive attributes prior to training to mitigate potential bias.