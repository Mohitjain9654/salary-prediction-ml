# 💼 Employee Salary Prediction App

This project is a **Machine Learning classification app** that predicts whether an employee earns **>50K or ≤50K** based on various demographic and professional features. The model is trained on the [Adult Income Dataset (UCI)](https://github.com/Mohitjain9654/salary-prediction-ml/blob/main/adult.csv), and deployed using **Streamlit**.

---

## 🔍 Features

- Built using **scikit-learn**, **XGBoost**, and **Pandas**
- User-friendly **Streamlit web interface**
- Supports **single** and **batch** prediction via CSV upload
- Clean, dark-mode UI with input sliders and dropdowns

---

## 🧠 Model Info

- Preprocessing:
  - Categorical encoding with `OneHotEncoder`
  - Numerical scaling with `StandardScaler`
  - Missing value handling with `SimpleImputer`

- Algorithms tested:
  - Logistic Regression
  - Random Forest
  - K-Nearest Neighbors
  - SVM
  - Gradient Boosting
  - XGBoost ✅ (Final selected model)

---

## 📁 Files in this Repo

| File | Description |
|------|-------------|
| `app.py` | Streamlit frontend for user interaction |
| `salary_prediction_model.joblib` | Trained machine learning model |
| `README.md` | Project overview and instructions |
| `notebook.ipynb` | Jupyter notebook for EDA, preprocessing, training, and evaluation |
| `requirements.txt` | Dependencies to run the app |

---

## 🚀 How to Run the App Locally

### 1️⃣ Clone the repo
```bash
git clone https://github.com/Mohitjain9654/salary-prediction-ml
cd salary-prediction-ml
```

### 2️⃣ Install requirements
```bash
pip install -r requirements.txt
```
### 3️⃣ Run the app
```bash
streamlit run app.py
```

---
### 🖼️ Example Input Features
```text
Age
Workclass
Education
Educational Number
Occupation
Hours per Week
Gender
Marital Status
Relationship
Native Country
Capital Gain / Loss
fnlwgt
Race
```
---
### 📂 Batch Prediction

You can upload a CSV file with multiple entries. The app will return predictions and let you download the results.

CSV must match model input columns.

---
## 🙋‍♂️ Author

Built with ❤️ by Mohit Jain


Feel free to connect or collaborate!

- 🔗 [LinkedIn](https://www.linkedin.com/in/mohit-jain-dev/)  
- 💻 [GitHub](https://github.com/Mohitjain9654)  
- 🌐 [Portfolio Website](https://mohitjain-portfolio.vercel.app/)  
- 📧 Email: mohitjain965405@gmail.com

---
