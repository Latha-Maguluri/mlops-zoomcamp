#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('python -V')


# In[2]:


import pandas as pd


# In[3]:


import pickle


# In[4]:


from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error


# In[5]:


import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nyc-taxi-experiment")


# In[6]:


df = pd.read_parquet('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet')


# In[7]:


def read_dataframe():
    df = pd.read_parquet('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet')

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df


# In[8]:


df = read_dataframe()


# In[10]:



categorical = ['PULocationID', 'DOLocationID'] #'PULocationID', 'DOLocationID']
numerical = ['trip_distance']
dv = DictVectorizer()

train_dicts = df[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dicts)


# In[11]:


df.columns


# In[12]:


target = 'duration'
y_train = df['duration'].values


# In[14]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_train)

mean_squared_error(y_train, y_pred)


# In[16]:


from pathlib import Path


# In[17]:


models_folder = Path('models')
models_folder.mkdir(exist_ok=True)


# In[18]:



import mlflow
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle
from pathlib import Path

models_folder = Path('models')
models_folder.mkdir(exist_ok=True)

with mlflow.start_run():
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred)
    
    # Log parameters (example: just default params here)
    mlflow.log_param("fit_intercept", lr.fit_intercept)
    
    # Log metrics
    mlflow.log_metric("mse", mse)
    
    # Save the model locally and log it as artifact
    model_path = models_folder / "linear_regression_model.pkl"
    with open(model_path, "wb") as f_out:
        pickle.dump(lr, f_out)
    mlflow.log_artifact(str(model_path))


# In[ ]:




