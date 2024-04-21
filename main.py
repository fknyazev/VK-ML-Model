import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor
from imblearn.over_sampling import SMOTE

data = pd.read_parquet("part_0.parquet")

data['os'] = np.where(data['user_agent'].str.contains("Windows"), 0,
              np.where(data['user_agent'].str.contains("Android"), 1,
              np.where(data['user_agent'].str.contains("IPhone|Mac OS"), 2,
              np.where(data['user_agent'].str.contains("Linux|LINUX"), 3,
              np.where(data['user_agent'].str.contains("CrOS"), 4, -1)))))

data['browser'] = np.where(data['user_agent'].str.contains("YaBrowser|YaSearchBrowser"), 0,
                   np.where(data['user_agent'].str.contains("Chrome"), 1,
                   np.where(data['user_agent'].str.contains("MiuiBrowser"), 2,
                   np.where(data['user_agent'].str.contains("Firefox"), 3,
                   np.where(data['user_agent'].str.contains("Safari"), 4, -1)))))

data['device'] = np.where(data['user_agent'].str.contains("Mobile"), 1, 0)

data['referer'] = data['referer'].str.extract(r'(domain_\d+)')[0].fillna('unknown_domain')

data.to_csv('output.csv')