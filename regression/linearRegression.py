
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

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


data_numeric = data.select_dtypes(include=['float64', 'int64'])

# Normalize the data in data_numeric excluding the target variable
data_normalized_cand = data_numeric.drop('price', axis=1)   
data_normalized = (data_normalized_cand - data_normalized_cand.mean()) / data_normalized_cand.std()

X_train, X_test, y_train, y_test = train_test_split(data_normalized, data_numeric["price"], test_size=0.33, random_state=42)

#Creo modelo 
reg = LinearRegression().fit(X_train, y_train)

#Predigo
y_pred = reg.predict(X_test)

#vif coefficients of data_numeric
X = add_constant(data_numeric.drop("price", axis=1))
pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index=X.columns)

coefficients = reg.coef_
intercept = reg.intercept_

print("Coefficients:", coefficients)
print("Intercept:", intercept)

metrics = get_metrics(y_test, y_pred)

print(metrics)

#write metrics into a csv file 
metrics.to_csv('regression//metrics.csv', index=False)


# plot 
plt.scatter(y_test, y_pred)
plt.xlabel('True Values')
plt.ylabel('Predictions')
#plt.show()

#save the plot into an image file 
plt.savefig('regression//scatter_plot.png')

# Calculate residuals
residuals = y_test - y_pred


#clear the plot
plt.clf()
# Plot histogram of residuals
plt.hist(residuals, bins=20, density=True, alpha=0.6, color='lightblue', label='Residuals')

# Fit a normal distribution to the residuals
mu, std = norm.fit(residuals)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2, label='Distribución Normal')

# Add labels and legend
plt.xlabel('Residuos')
plt.ylabel('Densidad')
plt.title('Distribución de los residuos')
plt.legend()

# Show plot
#plt.show()

plt.savefig('regression//residuals.png')

