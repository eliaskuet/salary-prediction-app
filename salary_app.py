#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template


# In[2]:


app = Flask(__name__)


# In[3]:


#model = pickle.load(open('model.pkl', 'rb'))
loaded_pipeline = joblib.load("salary_prediction_pipeline.pkl")


# In[4]:


@app.route('/')
def home():
    return render_template('index.html')


# In[5]:


@app.route('/estimate-salary', methods=['POST'])
def predict():
    age = float(request.form['age'])
    gender = request.form['gender']
    education_level = request.form['education_level']    
    job_title = request.form['job_title']
    years_experience = float(request.form['years_experience'])

    input_data = {
    'age': age,
    'gender': gender,
    'education_level': education_level,
    'job_title': job_title,
    'years_experience': years_experience
    }

    new_data = pd.DataFrame([input_data])

    prediction = loaded_pipeline.predict(new_data)
    output = round(float(prediction.item()), 2)
    return render_template('index.html', prediction_text = 'Estimated Salary is ${}'.format(output))


# In[6]:


@app.route('/predict_api', methods = ['POST'])
def predict_api():
    data = request.get_json(force=True)
    prediction = loaded_pipeline.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


# In[ ]:

if __name__ == "__main__":
    app.run(debug=True, port=5000)


