# **Medical Insurance Cost Analysis and Prediction System**

*Exploratory Data Analysis (EDA), Regression Models & Interactive Dashboard using Kaggle Dataset*

---

## **Overview**

Health insurance companies collect large amounts of customer information such as age, gender, body mass index (BMI), number of children, smoking status, residential region, and medical insurance charges. Manually analyzing these records to estimate insurance premiums and identify high-cost customers is difficult, time-consuming, and error-prone.

This project focuses on building a **Medical Insurance Cost Analysis and Prediction System** that performs **Exploratory Data Analysis (EDA)**, applies **Linear Regression**, **Multiple Linear Regression**, and **Logistic Regression**, and provides an **Interactive Dashboard** to understand factors influencing insurance costs and predict future medical insurance charges.

The workflow includes:

* Data Collection from Kaggle Insurance Dataset
* Data Cleaning and Preprocessing
* Descriptive Statistical Analysis
* Insurance Cost Analysis
* Group-Based Analysis
* Relationship Analysis
* Data Visualization
* Risk Classification
* Linear Regression
* Multiple Linear Regression
* Logistic Regression
* Model Evaluation
* Interactive Dashboard Development
* Business Insights and Conclusions

---

## **Dataset**

**Source:** Kaggle

**Dataset Link:** [https://www.kaggle.com/datasets/mirichoi0218/insurance](https://www.kaggle.com/datasets/mirichoi0218/insurance)

**Dataset Name:** Medical Cost Personal Dataset

### **Description**

The dataset contains real-world medical insurance records and includes demographic, lifestyle, and insurance-related information used to analyze and predict medical insurance charges.

### **Selected Columns**

| Column Name | Description                  |
| ----------- | ---------------------------- |
| `age`       | Age of policyholder          |
| `sex`       | Gender of policyholder       |
| `bmi`       | Body Mass Index              |
| `children`  | Number of dependents covered |
| `smoker`    | Smoking status               |
| `region`    | Residential region           |
| `charges`   | Medical insurance charges    |

---

## **Objectives**

### **1. Data Collection**

* Load insurance dataset from Kaggle
* Explore dataset structure
* Identify independent and dependent variables
* Analyze customer demographic information

### **2. Data Cleaning & Preprocessing**

* Check missing values
* Remove duplicate records
* Encode categorical variables (`sex`, `smoker`, `region`)
* Verify data types
* Prepare data for modeling

### **3. Descriptive Statistics**

* Average insurance charges
* Average age
* Average BMI
* Average number of children
* Statistical summary of numerical variables

### **4. Insurance Cost Analysis**

* Smokers vs Non-Smokers
* Male vs Female customers
* Age group comparisons
* BMI category comparisons
* High-cost customer identification

### **5. Group-Based Analysis**

* Age vs Insurance Charges
* BMI vs Insurance Charges
* Children vs Insurance Charges
* Region vs Insurance Charges
* Smoking Status vs Insurance Charges
* Gender vs Insurance Charges

### **6. Relationship Analysis**

* Age vs Charges correlation
* BMI vs Charges correlation
* Children vs Charges correlation
* Smoking Status vs Charges correlation
* Identification of major premium-driving factors

### **7. Data Visualization**

* Bar Charts
* Histograms
* Scatter Plots
* Box Plots
* Heatmaps
* Pie Charts

### **8. Risk Analysis**

Classify customers into:

* Low Cost Customers
* Medium Cost Customers
* High Cost Customers

Identify customers likely to generate higher medical expenses and insurance claims.

### **9. Linear Regression**

Predict insurance charges using:

* Independent Variable: `age`
* Dependent Variable: `charges`

### **10. Multiple Linear Regression**

Predict insurance charges using:

* `age`
* `bmi`
* `children`
* `smoker`

Target:

* `charges`

### **11. Logistic Regression**

Classify customers into:

* Low Cost Customer (0)
* High Cost Customer (1)

Target variable:

* `insurance_category`

Features:

* `age`
* `bmi`
* `children`
* `smoker`

### **12. Model Evaluation**

#### Linear Regression

* R² Score
* MAE
* MSE
* RMSE

#### Multiple Linear Regression

* R² Score
* MAE
* MSE
* RMSE

#### Logistic Regression

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Classification Report

### **13. Dashboard Creation**

Develop an interactive Plotly Dash dashboard with:

* Insurance Charges Distribution
* Age vs Charges Scatter Plot
* BMI vs Charges Scatter Plot
* Region-wise Insurance Charges
* Smoker vs Non-Smoker Comparison
* Gender-wise Insurance Charges
* Correlation Heatmap
* Cost Category Distribution

#### Dashboard Filters

* Age
* Region
* Gender
* Smoker Status
* Number of Children

### **14. Insights & Conclusion**

* Identify major factors affecting insurance premiums
* Analyze the impact of smoking on insurance costs
* Understand the influence of age and BMI
* Compare Linear Regression and Multiple Linear Regression performance
* Classify customers into cost categories
* Support premium estimation and risk assessment
* Enable data-driven decision-making

---

## **Project Highlights**

### **1. Data Collection & Cleaning**

* Dataset loaded from Kaggle Insurance Dataset
* Duplicate records removed
* Categorical variables encoded using Label Encoding
* Data types verified and standardized
* Dataset prepared for analysis and machine learning

### **2. Exploratory Data Analysis (EDA)**

* Insurance charges distribution analyzed
* Demographic characteristics explored
* High-cost customer segments identified
* Insurance charge trends studied across multiple customer groups
* Key premium-driving factors discovered

### **3. Visualization**

The project includes:

* **Bar Charts** – Age-wise and insurance-wise charge comparisons
<img width="1362" height="557" alt="image" src="https://github.com/user-attachments/assets/b3441893-2476-443c-9f19-c0e6b595f8d6" />

* **Histograms** – Age distributions
<img width="1361" height="552" alt="image" src="https://github.com/user-attachments/assets/de11cb03-a320-438f-a670-83a2f7f59b1b" />

* **Scatter Plots** – Age vs Charges
<img width="1363" height="557" alt="image" src="https://github.com/user-attachments/assets/dad88ba2-6160-4d3f-bfa5-708d0a28a436" />

* **Box Plots** – Outlier detection across smoking status and regions
<img width="1361" height="557" alt="image" src="https://github.com/user-attachments/assets/97bcecfe-c859-4678-8a93-a296c86344b8" />

* **Heatmaps** – Correlation analysis
<img width="1362" height="557" alt="image" src="https://github.com/user-attachments/assets/31e697d2-17d1-4e4f-bacd-055d75b58b7a" />

* **Pie Charts** – Smoker and non-smoker distributions
<img width="1363" height="556" alt="image" src="https://github.com/user-attachments/assets/d2383aa5-53ef-4abd-a00d-8adf2a55a944" />


### **4. Risk Classification**

Customers are categorized into:

| Category       | Description                |
| -------------- | -------------------------- |
| 🟢 Low Cost    | Lower insurance charges    |
| 🟡 Medium Cost | Moderate insurance charges |
| 🔴 High Cost   | Higher insurance charges   |

This helps insurance providers identify customers likely to generate significant healthcare expenses.

### **5. Predictive Modeling**

#### Linear Regression

Predicts insurance charges using age alone.

#### Multiple Linear Regression

Predicts insurance charges using age, BMI, children, and smoking status.

#### Logistic Regression

Predicts whether a customer belongs to a low-cost or high-cost insurance category.

### **6. Interactive Dashboard**

Built using Plotly Dash and includes:

* Dynamic filtering
* Insurance cost exploration
* Regional analysis
* Smoker impact analysis
* Correlation visualization
* Cost category distribution

---

## **Tools and Technologies**

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Plotly
* Plotly Dash
* Scikit-learn
* Jupyter Notebook
* Kaggle Dataset

---

## **Results**

### **Key Findings**

* Smoking status is the strongest factor affecting insurance charges.
* Insurance costs generally increase with age.
* Higher BMI values are associated with increased insurance expenses.
* Multiple Linear Regression performs better than Simple Linear Regression.
* High-cost customers can be effectively classified using Logistic Regression.
* Customer segmentation supports better premium planning and risk assessment.

### **Interpretation**

* Insurance companies can improve premium estimation accuracy.
* High-risk customers can be identified earlier.
* Data-driven pricing strategies become more effective.
* Automated analysis reduces manual effort and improves decision-making.

---

## **Conclusion**

The **Medical Insurance Cost Analysis and Prediction System** successfully analyzes real-world insurance data through **EDA, Regression Modeling, Risk Classification, and Interactive Dashboard Visualization**.

By identifying the factors that influence insurance premiums and predicting future charges, the system provides valuable insights for premium estimation, customer segmentation, risk assessment, and strategic decision-making in the insurance industry.

---

## **Author**

**Shree Pranava Ganesh**
Student at Kamaraj College
Thoothukudi, Tamil Nadu, India
