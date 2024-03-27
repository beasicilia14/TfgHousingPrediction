import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import pandas as pd
from sklearn.ensemble import RandomForestRegressor 
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

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

#grid search
from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200, 250, 300],
    'max_depth': [10, 20, 30, 40, 50]
}

grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5, scoring='neg_mean_absolute_error')
grid_search.fit(X_train, y_train)

# Print the best parameters
print("best parameters", grid_search.best_params_)
#write into a file called inforandomforest.txt 

rf = RandomForestRegressor(n_estimators=grid_search.best_params_['n_estimators'], max_depth=grid_search.best_params_['max_depth'])
rf.fit(X_train, y_train)

# Predict the target variable
y_pred = rf.predict(X_test)

metrics = get_metrics(y_test, y_pred)

metrics.to_csv('randomForest//metrics.csv', index=False)


grid_metrics = pd.DataFrame(grid_search.cv_results_)
grid_mean_scores = grid_metrics["mean_test_score"]
# absolute values of grid_mean_scores
abs_grid_mean_scores = abs(grid_mean_scores)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(np.array([50, 100, 200, 250, 300]), abs_grid_mean_scores[:5], '-o', label='max_depth = 10')
plt.plot(np.array([50, 100, 200, 250, 300]), abs_grid_mean_scores[5:10], '-o', label='max_depth = 20')
plt.plot(np.array([50, 100, 200, 250, 300]), abs_grid_mean_scores[10:15], '-o', label='max_depth = 30')
plt.plot(np.array([50, 100, 200, 250, 300]), abs_grid_mean_scores[15:20], '-o', label='max_depth = 40')
plt.plot(np.array([50, 100, 200, 250, 300]), abs_grid_mean_scores[20:25], '-o', label='max_depth = 50')

plt.legend()
plt.xlabel('Number of trees')
plt.ylabel('Mean Absolute Error')
plt.title('Mean Absolute Error for different parameters')

#save the plot into an image file
plt.savefig('randomForest//grid_search.png')

# Get the feature importances
feature_importances = rf.feature_importances_

# Create a dataframe with the feature importances
feature_importances_df = pd.DataFrame({'feature': X_train.columns, 'importance': feature_importances})

# Sort the dataframe by the feature importances
feature_importances_df = feature_importances_df.sort_values('importance', ascending=False)

# Display the 10 most important features
feature_importances_df.head(10)

with open('randomForest//inforandomforest.txt', 'w') as f:
    
    print("Best Parameters /n", file=f )
    print(grid_search.best_params_, file=f)

    #write importance 
    print("Feature importances /n", file=f)
    print(feature_importances_df.head(15), file=f)

    



