# AICTE_Employee-Salary-Prediction
This project aims to build a machine learning model that accurately predicts employee salaries based on features such as job title, education level, gender, age and years of experience. It uses regression algorithms and provides insights into salary trends for HR analytics and recruitment tools.

## Overview
Accurate salary prediction is valuable for HR planning, budgeting, and ensuring fair compensation. 
In this project, we:
- Collected and preprocessed a structured employee dataset
- Encoded categorical variables such as education level, job title, and gender
- Applied one-hot encoding and label encoding where appropriate
- Explored and compared multiple regression models for salary prediction
- Evaluated model performance using R² score, MAE, and RMSE
- Selected the best-performing model (Gradient Boosting Regressor) for final deployment

## Repository Structure
```bash
├── encoders/ # Saved encoders and feature metadata
│ ├── education_encoder.pkl
│ ├── feature_names.pkl
│ ├── gender_encoder.pkl
│ └── job_title_columns.pkl
├── EmployeeSalaryPrediction.ipynb # Main Jupyter notebook with full workflow
├── app.py # Streamlit app (deployment) 
├── best_gradient_boosting_model.pkl # Best-performing model
├── salary_dataset.csv # Dataset used for training and evaluation
├── requirements.txt # Python dependencies
└── README.md # Project documentation (this file)
```

## Dataset
The dataset contains the following features
- Age
- Years of Experience
- Education Level
- Job Title
- Gender
- Salary (target variable)

## Technologies Used
- Python
- Pandas, Numpy
- Scikit-Learn
- Matplotlib, Seaborn (for visualisation)
- Jupyter Notebook
- Streamlit (for deployment)

## Models Explored
- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree 
- Random Forest
- Gradient Boosting (Best Model)
- Support Vector Regression (SVR)
- KNeighbors Regression (KNN)

## Best Model Configuration
```python
GradientBoostingRegressor(
    learning_rate=0.2,
    max_depth=5,
    n_estimators=150
)
```

## Evaluation Metrics
- R² Score (Coefficient of determination)
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)

## How to Run
1. Clone the repository
```bash
git clone https://github.com/sbhat05/AICTE_Employee-Salary-Prediction.git
cd AICTE_Employee-Salary-Prediction
```
2. Setup the environment
Create a new python environment(python>=3.8). Navigate to the downloaded folder where requirements.txt is present.
```bash
pip install -r requirements.txt
```
3. Check the paths and run the cells in the jupyter notebook.
4. To run the app, go to the directory where app.py is present on the terminal, then
```bash
streamlit run app.py
```
