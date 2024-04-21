import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor
from imblearn.over_sampling import SMOTE

part0 = pd.read_csv("output.csv")
part_0 = part0.iloc[1000000:1999999]
train_users = pd.read_csv("train_users.csv")
geo_dataframe = pd.read_csv("geo_dataframe.csv")

merged_data = part_0.merge(train_users, on='user_id')
merged_data = merged_data.merge(geo_dataframe, on='geo_id')

le = LabelEncoder()
categorical_features = ['os', 'browser', 'device', 'region_id', 'country_id', 'referer']
for feature in categorical_features:
    merged_data[feature] = le.fit_transform(merged_data[feature])

features = merged_data[categorical_features]
target_gender = le.fit_transform(merged_data['gender'])
target_age = le.fit_transform(merged_data['age'])

smote = SMOTE()
features_balanced, target_gender_balanced = smote.fit_resample(features, target_gender)

X_train, X_test, y_train_gender, y_test_gender = train_test_split(features_balanced, target_gender_balanced, test_size=0.05, random_state=42)
X_train_age, X_test_age, y_train_age, y_test_age = train_test_split(features, target_age, test_size=0.05, random_state=42)

params = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'subsample': [0.7, 0.8, 0.9]
}

gender_model_path = 'gender_classifier.xgb'
age_model_path = 'age_regressor.xgb'

if os.path.exists(gender_model_path):
    gender_classifier = XGBClassifier()
    gender_classifier.load_model(gender_model_path)
    gender_classifier.fit(X_train, y_train_gender, xgb_model='gender_classifier.xgb')
    gender_classifier.save_model(gender_model_path)
else:
    gender_classifier = XGBClassifier()
    grid_search = GridSearchCV(gender_classifier, param_grid=params, scoring='accuracy', cv=3)
    grid_search.fit(X_train, y_train_gender)
    gender_classifier = grid_search.best_estimator_
    gender_classifier.save_model(gender_model_path)

if os.path.exists(age_model_path):
    age_regressor = XGBRegressor()
    age_regressor.load_model(age_model_path)
    age_regressor.fit(X_train_age, y_train_age, xgb_model='age_regressor.xgb')
    age_regressor.save_model(age_model_path)
else:
    age_regressor = XGBRegressor(objective='reg:squarederror')
    age_regressor.fit(X_train_age, y_train_age)
    age_regressor.save_model(age_model_path)


gender_predictions = gender_classifier.predict(X_test)
print("Gender Classification Report:")
print(classification_report(y_test_gender, gender_predictions))
print("Accuracy for Gender Prediction:", accuracy_score(y_test_gender, gender_predictions))

age_predictions = age_regressor.predict(X_test_age)
mse = mean_squared_error(y_test_age, age_predictions)
print("Age Prediction Mean Squared Error:", mse)
