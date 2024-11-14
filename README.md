# Heart Disease Prediction using Machine Learning

## Introduction

Heart disease is a leading cause of death worldwide, and early detection and prevention are crucial for improving patient outcomes. Machine learning has emerged as a powerful tool for predicting heart disease risk based on patient characteristics and medical history. This project explores the application of various machine learning models, including hybrid approaches, to predict the presence of heart disease.

## Motivation

The motivation behind this project stems from the significant impact of heart disease on individuals and public health. Early and accurate prediction of heart disease risk can enable timely intervention and preventive measures, potentially saving lives and reducing healthcare costs. By leveraging machine learning, we aim to develop a predictive model that can assist medical professionals in identifying individuals at high risk and guiding them towards appropriate care.

## Overview

This project aims to predict the presence of heart disease using a variety of machine learning models, including hybrid approaches. It utilizes the "Heart Disease UCI" dataset, available on Kaggle and the UCI Machine Learning Repository, to train and evaluate various classification algorithms. The project explores data preprocessing, exploratory data analysis, model building, and performance evaluation to identify the most effective model for predicting heart disease.

## Dataset

The "Heart Disease UCI" dataset contains information about patients, including age, sex, chest pain type, resting blood pressure, and other relevant features. These features are used to predict the presence or absence of heart disease, represented as a binary target variable.

## Methodology

The project follows these steps:

1. **Data Loading and Preprocessing:**
   - Loading the dataset using Pandas.
   - Handling missing values by imputing with the median for numerical features.
   - Encoding categorical features using mapping or one-hot encoding.

2. **Exploratory Data Analysis (EDA):**
   - Visualizing the distribution of features using histograms, box plots, and count plots.
   - Examining correlations between variables using a correlation matrix.

3. **Data Splitting and Scaling:**
   - Splitting the data into training and testing sets using an 80/20 split.
   - Standardizing the features using StandardScaler to ensure consistent scaling.

4. **Model Training and Evaluation:**
   - Implementing a range of classification models, including Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors, Naive Bayes, Support Vector Machine, and hybrid models (Voting and Stacking).
   - Evaluating model performance using accuracy, precision, recall, F1-score, and ROC-AUC.
   - Visualizing results using confusion matrices and ROC curves.

5. **Feature Importance Analysis:**
   - Analyzing feature importance using Random Forest to identify the most influential features.
   - Visualizing feature importance using bar plots.

## Results

The project compares the performance of different models based on the evaluation metrics. Hybrid models, combining the strengths of individual classifiers, demonstrate promising results. Detailed results are presented in tables and visualizations within the Jupyter Notebook.

## Conclusion

This project provides insights into the application of machine learning for heart disease prediction. It demonstrates the effectiveness of various models and identifies the key factors contributing to heart disease risk. The results can be valuable for medical professionals and researchers in improving early diagnosis and treatment strategies.

## Usage

To run this project:

1. **Environment:** Use Python 3.x with Jupyter Notebook or Google Colab.
2. **Dependencies:** Install the necessary libraries using:
```Bash
pip install pandas numpy scikit-learn xgboost seaborn matplotlib
```
3. **Data:** Download the "Heart Disease UCI" dataset and place it in the project directory.
4. **Execution:** Open the Jupyter Notebook and run the code cells sequentially.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

This project was inspired by various resources, including the UCI Machine Learning Repository, Kaggle datasets, and online tutorials. We acknowledge the contributions of the dataset creators and the open-source community.
