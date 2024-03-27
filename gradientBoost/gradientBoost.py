#Script for Gradient Boost Algorithm

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV


from xgboost import XGBRegressor


def get_metrics (y_real, y_pred): 
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_absolute_percentage_error
    import numpy as np
    mse = mean_squared_error(y_real, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_real, y_pred)
    r2 = r2_score(y_real, y_pred)
    mape = mean_absolute_percentage_error(y_real, y_pred)
    #put it in a df
    metrics = pd.DataFrame({'mse': [mse], 'rmse': [rmse], 'mae': [mae], 'r2': [r2], 'mape': [mape]})
    return metrics

cred = credentials.Certificate('claves.json')
firebase_admin.initialize_app(cred)

db = firestore.client()

houses_ref = db.collection('cleaned_data')
houses = houses_ref.get()

# Create an empty dataframe
data = pd.DataFrame()

# Iterate over the houses and add each house's data to the dataframe
for house in houses:
    house_data = house.to_dict()
    data = data.append(house_data, ignore_index=True)

data_numeric = data.select_dtypes(include=['float64', 'int64'])

data = data.drop(columns=["description", "propertyCode", "distance"], axis=1)
data_categorical = data.select_dtypes(include=['object'])

data_boolean = data.select_dtypes(include=['bool'])

# Create an instance of the OneHotEncoder
encoder = OneHotEncoder()

# Fit and transform the categorical variables
encoded_data = encoder.fit_transform(data_categorical)

# Convert the encoded data to a dataframe
encoded_df = pd.DataFrame(encoded_data.toarray(), columns=encoder.get_feature_names_out(data_categorical.columns))

# Concatenate the encoded dataframe with the numeric dataframe
encoded_data = pd.concat([data_numeric, encoded_df, data_boolean], axis=1)

X_train, X_test, y_train, y_test = train_test_split(encoded_data.drop("price", axis=1), encoded_data["price"] , test_size=0.33, random_state=42)


# Define the hyperparameters grid
param_grid = {
    'n_estimators': [100, 200, 300, 350, 400, 500, 600],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.01, 0.001]
}

# Create an instance of the XGBRegressor
xgb = XGBRegressor()

# Create GridSearchCV instance
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')

# Fit the model on the training data
grid_search.fit(X_train, y_train)

# Get the best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Print the best parameters and best score
print("Best Parameters:", best_params)
print("Best Negative Mean Squared Error:", best_score)

#use best parameters to predict on test 
xgb = XGBRegressor(n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'], learning_rate=best_params['learning_rate'])
xgb.fit(X_train, y_train)

# Predict the target variable
y_pred = xgb.predict(X_test)

# Calculate the metrics
metrics = get_metrics(y_test, y_pred)

metrics.to_csv("gradientBoost//metrics.csv")

# Plot the predicted vs real values
plt.scatter(y_test, y_pred)
plt.xlabel('Real Values')
plt.ylabel('Predicted Values')
plt.title('Real vs Predicted Values')
# Save the plot
plt.savefig('gradientBoost//real_vs_predicted.png')


#Get the importance of each feature
importances = xgb.feature_importances_

# Create a dataframe with the feature importances
feature_importances_df = pd.DataFrame({'feature': X_train.columns, 'importance': importances})

# Sort the dataframe by the importance in descending order
feature_importances_df = feature_importances_df.sort_values('importance', ascending=False)

#get top 15 features
feature_importances_df = feature_importances_df.head(15)

print(feature_importances_df)

#write into a file called infoxgboost.txt
with open('gradientBoost//infoxgboost.txt', 'w') as file:
    file.write(f"Best Parameters: {best_params}\n")
    file.write(feature_importances_df.to_string(index=False))




