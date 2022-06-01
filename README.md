# Machine Learning Projects

1) Breast Cancer Prediction
    - We use logistic regression to classify breast cells into malignant or benign, based on features extracted from breast mass images.
    - Skills and software used: scikit-learn, pandas, seaborn, matplotlib, numpy, Jupyter Notebook, Python3
    - Dataset source: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
    - Results on the testing set:
        - Accuracy = 96.5% and F-score = 95.0%

2) Credit Card Fraud Detection
    - We train a logistic regression model, to classify credit card transactions into normal or fraudulent.
    - PySpark and MLlib have been used due to the large size of the dataset.
    - Due to the class imbalance (only ~0.172% cases are frauds), Area under ROC Curve (AUC) has been used as a metric.
    - Skills and software used: PySpark, MLlib, Jupyter Notebook, Python3
    - Dataset source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    - Results on the testing set: 
        - AUC (Area under ROC Curve) = 0.81

3) House Price Prediction
    - We use linear regression, support vector regression, and random forest regression to predict house prices in King County USA.
    - Skills and software used: scikit-learn, pandas, seaborn, matplotlib, numpy, Jupyter Notebook, Python3
    - Dataset source: https://www.kaggle.com/datasets/harlfoxem/housesalesprediction
    - Results on the testing set:
        - Mean Absolute Percentage Error = 13.4% for Random Forest Regression, 15.0% for Support Vector Regression, and 25.2% for Linear Regression
