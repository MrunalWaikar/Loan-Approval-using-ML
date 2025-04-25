# Loan-Approval-using-ML
# 🏦 Loan Approval Prediction using Machine Learning

An end-to-end machine learning project to automate the loan approval process using Python. The project utilizes classification algorithms to predict whether a loan should be approved based on applicant data.

---

## 📌 Objective

To build a predictive model that helps financial institutions identify eligible loan applicants by analyzing various features such as:

- Applicant's income and employment details
- Credit history
- Loan amount and tenure
- Marital status, education, and property area

---

## 🧰 Tools & Technologies

- **Programming Language**: Python
- **Libraries**:  
  - `Pandas` – for data manipulation  
  - `NumPy` – for numerical operations  
  - `Matplotlib` & `Seaborn` – for visualization  
  - `Scikit-learn` – for machine learning models and evaluation  
- **Environment**: Jupyter Notebook

---

## 📁 Dataset

- **Source**: [Kaggle - Loan Prediction Dataset](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset)
- **Features**:
  - `Gender`, `Married`, `Education`, `Self_Employed`
  - `ApplicantIncome`, `CoapplicantIncome`
  - `LoanAmount`, `Loan_Amount_Term`, `Credit_History`
  - `Property_Area`, `Loan_Status` (target variable)

---

## 🧠 Key Steps

1. **Data Preprocessing**:
   - Handle missing values
   - Encode categorical variables
   - Normalize and clean data

2. **Exploratory Data Analysis (EDA)**:
   - Understand feature distributions
   - Visualize correlations and key insights

3. **Model Building & Evaluation**:
   - Train multiple classification models:
     - Logistic Regression
     - Decision Tree
     - Random Forest
     - Naive Bayes
     - K-Nearest Neighbors
   - Evaluate using accuracy, precision, recall, and F1-score

4. **Model Selection**:
   - Choose the best-performing model based on evaluation metrics

---

## 📈 Insights & Results

- 📊 **Best Model**: Naive Bayes  
- ✅ **Accuracy Achieved**: ~82%  
- 🔍 **Important Features**: Credit history, income, loan amount

*Note: You can replace these values with your actual results.*

---

## 🚀 Getting Started

### 📦 Install Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
