import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor


# =========================================================
# STEP 1 — LOAD DATA
# =========================================================

df = pd.read_csv("salary_data.csv")

print("Original Data:")
print(df.head())


# =========================================================
# CHANGe COLUMN NAMES TO LOWERCASE
# =========================================================

df.columns = ["age", "gender", "education_level", "job_title", "years_experience", "salary"]

mapping = {
    "education_level": {
        "Bachelor's": "Bachelor's Degree",
        "Master's": "Master's Degree",
        "phD": "PhD"
    },
    "job_title": {
        'Back end Developer': 'Back End Developer',
        'Customer Service Rep': 'Customer Service Representative',
        'Front end Developer': 'Front End Developer',
        'Juniour HR Coordinator': 'Junior HR Coordinator',
        'Juniour HR Generalist': 'Junior HR Generalist',
        'Social Media Man': 'Social Media Manager'
    }
}
df.replace(mapping, inplace=True)

# =========================================================
# STEP XXX — HANDLE MISSING VALUES
# =========================================================
#df.dropna(inplace=True, subset=df[df.isnull().any(axis=1)].columns, how='all')
df.dropna(inplace=True)

#print(sorted(df["job_title"].unique().tolist()))
#print(sorted(df["education_level"].unique().tolist()))
# =========================================================
# STEP 2 — DEFINE FEATURES AND TARGET
# =========================================================

X = df.drop("salary", axis=1)
y = df["salary"]


# =========================================================
# STEP 3 — DEFINE COLUMN TYPES
# =========================================================

categorical_features = [
    'gender',
    'education_level',
    'job_title'
]

numeric_features = [
    'years_experience'
]


# =========================================================
# STEP 4 — CREATE PREPROCESSOR
# =========================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            'cat',
            OneHotEncoder(handle_unknown='ignore'),
            categorical_features
        )
    ],
    remainder='passthrough'
)


# =========================================================
# STEP 5 — CREATE FULL PIPELINE
# =========================================================

## Using Linear Regression
# pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('model', LinearRegression())
# ])

## Using XGBoost
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', XGBRegressor(random_state=42))
])


# =========================================================
# STEP 6 — SPLIT DATA
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

#X_train.info()
#X_train.head()
# =========================================================
# STEP 7 — TRAIN MODEL
# =========================================================

pipeline.fit(X_train, y_train)

print("\nModel training completed.")


# =========================================================
# STEP 8 — TEST MODEL
# =========================================================

y_pred = pipeline.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Squared Error:", mse)
print("R2 Score:", r2)


# =========================================================
# STEP XXX — AGAIN TRAIN WITH FULL DATA (OPTIONAL)
# =========================================================
pipeline.fit(X, y)

train_score = pipeline.score(X_train, y_train)
test_score = pipeline.score(X_test, y_test)

print("Train:", train_score)
print("Test:", test_score)

# =========================================================
# STEP 9 — SAVE FULL PIPELINE
# =========================================================

joblib.dump(pipeline, "salary_prediction_pipeline.pkl")

print("\nPipeline saved successfully.")


# # =========================================================
# # STEP 10 — LOAD SAVED PIPELINE
# # =========================================================

# loaded_pipeline = joblib.load("salary_prediction_pipeline.pkl")

# print("\nPipeline loaded successfully.")


# # =========================================================
# # STEP 11 — USER INPUT (SIMULATING DROPDOWN VALUES)
# # =========================================================

# # Example values from frontend dropdown

# education_level = "Master's"
# job_title = "Software Engineer"
# years_of_experience = 4


# # =========================================================
# # STEP 12 — CREATE INPUT DATAFRAME
# # =========================================================

# # new_data = pd.DataFrame({
# #     'Education Level': [education_level],
# #     'Job Title': [job_title],
# #     'Years of Experience': [years_of_experience]
# # })


# new_data = pd.DataFrame({
#     'age': [30],
#     'job_title': ['Software Engineer'],
#     'education_level': ["Master's"],
#     'years_experience': [5],
#     'gender': ['Male']
# })


# print("\nNew Input Data:")
# print(new_data)


# # =========================================================
# # STEP 13 — PREDICT
# # =========================================================

# prediction = loaded_pipeline.predict(new_data)

# print("\nPredicted Salary:")
# print(prediction[0])