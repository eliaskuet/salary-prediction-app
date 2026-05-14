# Salary Prediction Web App

A Machine Learning based Salary Prediction Web Application built with Flask, Scikit-learn, and XGBoost.

This project predicts estimated salary based on:

- Age
- Gender
- Education Level
- Job Title
- Years of Experience

The application includes:

- Machine Learning model training pipeline
- Data preprocessing
- Flask backend
- HTML/CSS frontend
- Salary prediction API
- Model serialization using Joblib

---

# Project Preview

The application provides a user-friendly web interface where users can enter employee information and get an estimated salary prediction instantly.

---

# Technologies Used

## Backend
- Python
- Flask

## Machine Learning
- Scikit-learn
- XGBoost
- Pandas
- NumPy
- Joblib

## Frontend
- HTML5
- CSS3

---

# Project Structure

```bash
salary-prediction-app/
│
├── salary_model.py
├── salary_app.py
├── salary_prediction_pipeline.pkl
├── salary_data.csv
├── requirements.txt
├── README.md
│
├── templates/
│   └── index.html
│
├── static/
│   └── css/
│       └── style.css
│
└── screenshots/
    └── predict_salary_app_preview.png.png
```

---

# Features

- Machine Learning salary prediction
- Full preprocessing pipeline
- OneHotEncoding for categorical features
- XGBoost regression model
- Flask web application
- REST API support
- Responsive UI
- Model saving/loading using Joblib

---

# Dataset Features

The model uses the following input features:

| Feature | Type |
|---|---|
| age | Numeric |
| gender | Categorical |
| education_level | Categorical |
| job_title | Categorical |
| years_experience | Numeric |

# Target variable:

- salary

---

# Machine Learning Workflow

## 1. Load Dataset

The dataset is loaded using Pandas.

```python
df = pd.read_csv("salary_data.csv")
```

---

## 2. Data Cleaning

- Column renaming
- Text normalization
- Handling missing values

```python
df.dropna(inplace=True)
```

---

## 3. Feature Engineering

Categorical features:

```python
categorical_features = [
    'gender',
    'education_level',
    'job_title'
]
```

Numeric features:

```python
numeric_features = [
    'age',
    'years_experience'
]
```

---

## 4. Preprocessing

OneHotEncoder is used for categorical variables.

```python
OneHotEncoder(handle_unknown='ignore')
```

---

## 5. Model Training

The project uses:

```python
XGBRegressor(random_state=42)
```

inside a Scikit-learn Pipeline.

---

## 6. Model Evaluation

Evaluation metrics:

- Mean Squared Error (MSE)
- R² Score

---

## 7. Save Model

The trained pipeline is saved using Joblib.

```python
joblib.dump(pipeline, "salary_prediction_pipeline.pkl")
```

---

# Installation Guide

## Step 1 — Clone Repository

```bash
git clone https://github.com/eliaskuet/salary-prediction-app.git
```

---

## Step 2 — Move to Project Directory

```bash
cd salary-prediction-app
```

---

## Step 3 — Create Virtual Environment (Recommended)

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux/Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Step 4 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Requirements File

Create a `requirements.txt` file with:

```txt
Flask
numpy
pandas
scikit-learn
xgboost
joblib
```

---

# Train the Model

Run the following command:

```bash
python salary_model.py
```

This will:

- Train the ML model
- Evaluate the model
- Save the pipeline as:

```bash
salary_prediction_pipeline.pkl
```

---

# Run the Flask Application

Start the Flask server:

```bash
python salary_app.py
```

The application will run at:

```bash
http://127.0.0.1:5000/
```

---

# Web Application Usage

## Input Fields

Users need to provide:

- Age
- Gender
- Education Level
- Job Title
- Years of Experience

## Output

The application predicts:

- Estimated Salary

---

# API Usage

## Endpoint

```bash
POST /predict_api
```

---

## Example JSON Input

```json
{
  "age": 30,
  "gender": "Male",
  "education_level": "Master's Degree",
  "job_title": "Software Engineer",
  "years_experience": 5
}
```

---

## Example Using Postman

- Method: POST
- URL:

```bash
http://127.0.0.1:5000/predict_api
```

- Body → raw → JSON

---

# Screenshots

You can add screenshots inside the `screenshots/` folder.

Example:

```bash
screenshots/predict_salary_app_preview.png
```

Then display it in README:

```md
![App Preview](screenshots/predict_salary_app_preview.png)
```

---

# Future Improvements

Possible future enhancements:

- User authentication
- Database integration
- Docker deployment
- Cloud deployment
- Model optimization
- Better UI/UX
- Advanced feature engineering

---

# GitHub Upload Guide

## Initialize Git

```bash
git init
```

---

## Add Files

```bash
git add .
```

---

## Commit Files

```bash
git commit -m "Initial commit"
```

---

## Connect GitHub Repository

```bash
git remote add origin https://github.com/eliaskuet/salary-prediction-app.git
```

---

## Push to GitHub

```bash
git push -u origin main
```

---

# License

This project is open-source and available under the MIT License.

---

# Acknowledgements

Special thanks to:

- Flask
- Scikit-learn
- XGBoost
- Pandas
- Open Source Community

# Datasource Reference
**https://www.geeksforgeeks.org/machine-learning/dataset-for-linear-regression/**


---

# Author

## Elias Hossain

- GitHub: [Elias GitHub Profile](https://github.com/eliaskuet)
- LinkedIn: [Elias LinkedIn Profile](https://www.linkedin.com/in/ngelias)
